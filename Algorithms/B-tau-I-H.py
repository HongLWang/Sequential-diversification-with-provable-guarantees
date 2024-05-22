# This is the B-tau-I heuristic proposed in our paper

import time
import os
import multiprocessing
from tqdm import tqdm
from utility import *
import pickle
from utils import *



def filter_by_greedy_MSD(ratings, distance_matrix, seed_size):

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

    num_user = ratings.shape[0]
    num_items = distance_matrix.shape[0]

    all_2_permutations = generate_combinations_and_permutations(np.arange(num_items), 2)

    Candidate_set = np.zeros((num_user, seed_size))

    for u in range(num_user):
        user_prob = ratings[u]

        all_2_exps = expectation_value_MSD_incremental(all_2_permutations, distance_matrix, user_prob)
        best_indices = np.argmax(all_2_exps)
        best_2item = all_2_permutations[best_indices]

        best_candidate = np.array([[best_2item[0], best_2item[1]],[best_2item[1], best_2item[0]]])
        all_extended = []
        for ranking in best_candidate:
            extended_ranking = extend_greedy_prefix(ranking, seed_size)
            all_extended.append(extended_ranking)

        all_extended = np.array(all_extended)

        all_exp = expectation_value_MSD_incremental(np.array(all_extended), distance_matrix, user_prob)
        best_RANKING_idx = np.argmax(all_exp)
        best_ranking = all_extended[best_RANKING_idx]

        Candidate_set[u] = best_ranking

    return Candidate_set


def search_best_k_nodes_from_Candidate_set(filtered_nodes, k_param, items_continue_p, distance_matrix, num_items):

    def extend_bestEdge_to_path_greedy_MSD(best_nodes):
        # the greedy increment is the MSD increment instead of the MHP increment.
        # given the best k node, extend it to k+1 by maximizign MSD obj value -- case 1
        # find best (k+1) node that optimize the MHP obj value -- case 2
        # MSD_obj(case 1) > MSD(case 2).

        # it is necessary to compare BKI with greedy MSD algorithm.

        sequence = list(best_nodes)
        chosen_node_accum_p = np.prod(items_continue_p[np.array(sequence)])  # scala

        accum_p_vec = items_continue_p * chosen_node_accum_p  # vec
        accum_p_vec[np.array(sequence)] = np.NINF

        dis_all_2_chosen = np.sum(distance_matrix[:, np.array(sequence)], axis=1)  # vec

        all_items = np.arange(num_items)
        while len(sequence) < num_items:
            gain = np.multiply(accum_p_vec, dis_all_2_chosen)

            best_node_index = np.argmax(gain)
            best_node = all_items[best_node_index]
            sequence.append(best_node)

            accum_p_vec = accum_p_vec * items_continue_p[best_node]
            accum_p_vec[best_node] = np.NINF

            dis_all_2_chosen += distance_matrix[:, best_node]

        return sequence

    all_k_permutation = generate_combinations_and_permutations(filtered_nodes, k_param)
    exp_values = expectation_value_MHP(all_k_permutation, distance_matrix, items_continue_p)
    best_permu_idx = np.argmax(exp_values)
    best_permu = all_k_permutation[best_permu_idx]

    sequence = extend_bestEdge_to_path_greedy_MSD(best_permu)

    return sequence

def get_recommendations_chunk(k_param, distance_matrix, Candidate_chunk, ratings_chunk):
    num_items = distance_matrix.shape[0]
    seq_chunk = []

    for idx in tqdm(range(ratings_chunk.shape[0])):

        items_continue_p = ratings_chunk[idx]
        top_node_for_u = Candidate_chunk[idx]
        best_sequence = search_best_k_nodes_from_Candidate_set(top_node_for_u, k_param, items_continue_p, distance_matrix,
                                                        num_items)
        seq_chunk.append(best_sequence)

    return seq_chunk


def get_recommendation(datset_name, k_param, ratings, distance_matrix, mapping_range, regime, top_num, strategy):

    distance_matrix = distance_matrix / np.max(distance_matrix)

    num_user, num_items = ratings.shape

    ratings = np.interp(ratings, (1, 5), mapping_range)

    Candidate_seeds = filter_by_greedy_MSD(ratings, distance_matrix, 100)

    hoped_num_process = min(48, multiprocessing.cpu_count())
    chunk_range = separate_to_chunks(np.arange(num_user), hoped_num_process)
    pool = multiprocessing.Pool()

    results = []
    for i in range(len(chunk_range)):
        chunk = chunk_range[i]
        ratings_chunk = ratings[chunk[0]:chunk[1]]
        Candidate_chunk = Candidate_seeds[chunk[0]:chunk[1]][:,:top_num]

        result = pool.apply_async(get_recommendations_chunk,
                                  args=(k_param, distance_matrix, Candidate_chunk, ratings_chunk))

        results.append(result)

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

        save_ranking_2file(datset_name, k_param, regime, seq_list, strategy)

    expectation_arr = np.concatenate(expectation_list)

    return np.average(expectation_arr), np.std(expectation_arr)


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

    users_dictionary, items_dictionary, distance_matrix = get_structures(dataset_name,
                                                                               jaccard_distance=jaccard_distance,
                                                                               folder="../outputs")


    file = open(f"../ProcessedData/{dataset_name}_rating.pkl", 'rb')
    ratings = pickle.load(file)
    file.close()

    result_folder = 'Results'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    ranking_folder = 'ranking'  # save the permutation/ranking of items for each user, this is useful for visualization.
    if not os.path.exists(ranking_folder):
        os.makedirs(ranking_folder)


    regimes_mapping = {"small": [0.1, 0.3], "medium": [0.4, 0.6], "large": [0.7, 0.9]}
    regimes = [ "large", "medium","small"]


    strategy = 'B-tau-I-H'


    for regime in regimes:
        mapping_range = regimes_mapping[regime]


        for k_param in [3,4]:

            if k_param == 3:
                top_num = 100
            if k_param == 4:
                top_num = 60


            start = time.time()

            exp_avg, exp_std = get_recommendation(dataset_name, k_param, ratings, distance_matrix, mapping_range, regime, top_num, strategy)

            print('expectation for strategy ', strategy, " with k = ", k_param,
                  " is ", exp_avg, exp_std, ' for dataset ',
                  dataset_name, ' for regime ', mapping_range, 'spend time ', time.time()-start)

            rst = (dataset_name, regime, k_param, strategy, exp_avg, exp_std, time.time() - start)

            result_path = f"Results/{strategy}.txt"
            with open(result_path, 'a+') as file:
                row_str = '\t'.join(map(str, rst))
                file.write(row_str + '\n')


if __name__ == '__main__':

    datasets = ["coat","netflix","movielens", "KuaiRec", "yahoo"]

    dataset_result = []
    for dataset in datasets:
        print(dataset)
        results = main(dataset)

