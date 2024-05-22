# This is greedy algorithm 3 proposed in our paper
import time
import os
from tqdm import tqdm
import sys, pickle

parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

sys.path.append(parent)

import multiprocessing
from utility import get_structures
from utils import *


def greedy_algorithm(user_prob, distance_matrix):


    def extend_bestEdge_to_path_greedy_MSD(best_nodes):

        sequence = list(best_nodes)
        chosen_node_accum_p = np.prod(user_prob[np.array(sequence)]) #scala

        accum_p_vec = user_prob * chosen_node_accum_p #vec
        accum_p_vec[np.array(sequence)] = np.NINF

        dis_all_2_chosen = np.sum(distance_matrix[:, np.array(sequence)], axis=1) #vec

        all_items = np.arange(num_items)
        while len(sequence) < num_items:

            gain = np.multiply(accum_p_vec,dis_all_2_chosen)

            best_node_index = np.argmax(gain)
            best_node = all_items[best_node_index]
            sequence.append(best_node)

            accum_p_vec = accum_p_vec * user_prob[best_node]
            accum_p_vec[best_node] = np.NINF

            dis_all_2_chosen += distance_matrix[:, best_node]

        return sequence

    def extend_greedy_prefix(best_nodes, threshold):

        sequence = list(best_nodes)
        chosen_node_accum_p = np.prod(user_prob[np.array(sequence)]) #scala

        accum_p_vec = user_prob * chosen_node_accum_p #vec
        accum_p_vec[np.array(sequence)] = np.NINF

        dis_all_2_chosen = np.sum(distance_matrix[:, np.array(sequence)], axis=1) #vec

        all_items = np.arange(num_items)
        while len(sequence) < threshold:

            gain = np.multiply(accum_p_vec,dis_all_2_chosen)

            best_node_index = np.argmax(gain)
            best_node = all_items[best_node_index]
            sequence.append(best_node)

            accum_p_vec = accum_p_vec * user_prob[best_node]
            accum_p_vec[best_node] = np.NINF

            dis_all_2_chosen += distance_matrix[:, best_node]

        return sequence


    num_items = distance_matrix.shape[0]
    all_2_permutations = generate_combinations_and_permutations(np.arange(num_items), 2)
    all_2_exps = expectation_value_MSD_incremental(all_2_permutations, distance_matrix, user_prob)


    best_indices = np.argmax(all_2_exps)
    best_2item = all_2_permutations[best_indices]

    best_candidate = np.array([[best_2item[0], best_2item[1]],[best_2item[1], best_2item[0]]])


    all_extended = []
    for ranking in best_candidate:
        extended_ranking = extend_greedy_prefix(ranking, 100)
        all_extended.append(extended_ranking)

    all_extended = np.array(all_extended)

    all_exp = expectation_value_MSD_incremental(np.array(all_extended), distance_matrix, user_prob)
    best_RANKING_idx = np.argmax(all_exp)
    best_ranking = all_extended[best_RANKING_idx]

    best_full_ranking = extend_bestEdge_to_path_greedy_MSD(best_ranking)


    return best_full_ranking

def get_recommendations_chunk(distance_matrix, ratings_chunk):
    seq_chunk = []
    for idx in tqdm(range(ratings_chunk.shape[0])):
        items_continue_p = ratings_chunk[idx]
        best_sequence = greedy_algorithm(items_continue_p, distance_matrix)
        seq_chunk.append(best_sequence)

    return seq_chunk

def get_recommendation( ratings, items_items_distances, mapping_range, regime, dataset_name):

    distance_matrix = items_items_distances
    distance_matrix = distance_matrix / np.max(distance_matrix)

    num_user, num_items = ratings.shape

    ratings = np.interp(ratings, (1, 5), mapping_range)

    hoped_num_process = min(48, multiprocessing.cpu_count())

    chunk_range = separate_to_chunks(np.arange(num_user), hoped_num_process)
    pool = multiprocessing.Pool()


    print ('num of threads is ', len(chunk_range))
    results = []
    for i in range(len(chunk_range)):
        chunk = chunk_range[i]
        ratings_chunk = ratings[chunk[0]:chunk[1]]
        result = pool.apply_async(get_recommendations_chunk,
                                  args=(distance_matrix, ratings_chunk))

        results.append(result)
    #
    pool.close()
    pool.join()

    result_list = []
    for result in results:
        seq_list = result.get()
        result_list.append(seq_list)

    pool.terminate()

    expectation_list = []
    for i, seq_list in enumerate(result_list):
        chunk = chunk_range[i]
        ratings_chunk_exp = ratings[chunk[0]:chunk[1]]

        seq_chunk_arr = lib.to_object_array(seq_list).astype(int)
        exp_torch_arr = expectation_value_MSD_torch(seq_chunk_arr, distance_matrix, ratings_chunk_exp)
        expectation_list.append(exp_torch_arr.numpy())

        save_ranking_2file(dataset_name, 2, regime, seq_list, 'SimpleGreedy')




    expectation_arr = np.concatenate(expectation_list)
    exp_deriviation = np.std(expectation_arr)
    average_exp = np.average(expectation_arr)

    return average_exp, exp_deriviation



def main(dataset_name):
    jaccard_distances_dict = {"movielens": "genres", "KuaiRec": "users",
                              "coat": "genres", "yahoo": "genres", "netflix": "genres"}

    jaccard_distance = jaccard_distances_dict[dataset_name]

    print(f"Dataset: {dataset_name}")
    print(f"Adopted jaccard distance: {jaccard_distance}")

    # HERE THE OUTPUTS FOLDER
    folder = f"../outputs/jaccard_{jaccard_distance}"

    if not os.path.exists(folder):
        os.makedirs(folder)

    users_dictionary, items_dictionary, items_items_distances = get_structures(dataset_name,
                                                                               jaccard_distance=jaccard_distance,
                                                                               folder="../outputs")

    file = open(f"../ProcessedData/{dataset_name}_rating.pkl", 'rb')
    ratings = pickle.load(file)
    file.close()

    regimes_mapping = {"small": [0.1, 0.3], "medium": [0.4, 0.6], "large": [0.7, 0.9]}
    regimes = ["large", "medium", "small"]


    result_folder = 'Results'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    ranking_folder = 'ranking'  # save the permutation/ranking of items for each user, this is useful for visualization.
    if not os.path.exists(ranking_folder):
        os.makedirs(ranking_folder)


    strategy = 'SimpleGreedy'


    for regime in regimes:
        mapping_range = regimes_mapping[regime]

        start = time.time()

        average_exp, exp_deriviation = get_recommendation(ratings, items_items_distances, mapping_range, regime, dataset_name)


        print('expectation for strategy ', strategy,
              " is ", average_exp, ' std is ', exp_deriviation, ' for dataset ',
              dataset_name, ' for regime ', mapping_range, 'spend time ', time.time()-start)

        rst = (dataset_name, regime, average_exp, exp_deriviation, time.time() - start)

        result_path = f"Results/{strategy}.txt"
        with open(result_path, 'a+') as file:
            row_str = '\t'.join(map(str, rst))
            file.write(row_str + '\n')


if __name__ == '__main__':

    datasets = ["coat", "netflix","movielens", "KuaiRec", "yahoo"]

    dataset_result = []
    for dataset in datasets:
        main(dataset)
        print(dataset)

