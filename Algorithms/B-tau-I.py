# implement many filtering method, try to improve the bke-H method performance.
# if the filtering is good enough, then bke-H (when k is large) should give very good performance.


import time
import os
import multiprocessing
from tqdm import tqdm
from utility import get_structures, get_jaccard_matrix, get_items_genres_matrix
import random, pickle
from util_expectations import *


def search_best_k_nodes_fromFilters(filtered_nodes, k_param, items_continue_p, distance_matrix, num_items):

    def extend_bestEdge_to_path_arbtrary(best_nodes):
        num_items = distance_matrix.shape[0]
        chosen_nodes = list(best_nodes)
        all_items = np.arange(num_items)
        rest_nodes = list(set(all_items) - set(chosen_nodes))
        random.shuffle(rest_nodes)
        chosen_nodes.extend(rest_nodes)

        return chosen_nodes

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
    # exp_values = expectation_value_MHP_torch(all_k_permutation, distance_matrix, items_continue_p)
    exp_values = expectation_value_MHP(all_k_permutation, distance_matrix, items_continue_p)
    # exp_values = expectation_MHP_incremental_chunk(all_k_permutation, distance_matrix, items_continue_p)
    # exp_values = expectation_MHP_incremental(all_k_permutation, distance_matrix, items_continue_p)
    best_permu_idx = np.argmax(exp_values)
    best_permu = all_k_permutation[best_permu_idx]

    sequence = extend_bestEdge_to_path_greedy_MSD(best_permu)

    return sequence



def get_recommendations_chunk(k_param, distance_matrix, filtered_nodes, ratings_chunk):
    num_items = distance_matrix.shape[0]
    seq_chunk = []

    # print (ratings_chunk.shape)
    # print (filtered_nodes.shape)

    # if filtered_nodes.shape[0] == 0:
    #     print('dimension 1 of filtered nodes is 0')

    for idx in tqdm(range(ratings_chunk.shape[0])):
    # for idx in range(ratings_chunk.shape[0]):
        # print('processing item ' , idx, 'out of total ', ratings_chunk.shape[0] , 'items in thread')
        items_continue_p = ratings_chunk[idx]
        top_node_for_u = filtered_nodes[idx]
        best_sequence = search_best_k_nodes_fromFilters(top_node_for_u, k_param, items_continue_p, distance_matrix,
                                                        num_items)
        seq_chunk.append(best_sequence)

    return seq_chunk


def get_recommendation(datset_name, filter_method, k_param, ratings, items_items_distances, mapping_range, regime, top_num):

    distance_matrix = items_items_distances
    distance_matrix = distance_matrix / np.max(distance_matrix)

    num_user, num_items = ratings.shape

    ratings = np.interp(ratings, (1, 5), mapping_range) # 0.1-0.3 for small regime, 0.4-0.6 for medium and 0.7-0.9 for large


    if filter_method in ['Dist', 'Dist_Prob', 'Matching', 'Relevance', ]:
        filename = 'Filtered_result/' + datset_name + '_' + filter_method + '.pkl'
        if not os.path.exists(filename):
            raise FileNotFoundError(f"The file {filename} does not exist.")
        else:
            with open(filename, 'rb') as file:
                node_filtered = pickle.load(file)[:, :top_num]


    elif filter_method in ['MSD']:
        filename = 'Filtered_result/' + datset_name + '_' + filter_method + '_' + regime + '.pkl'
        if not os.path.exists(filename):
            raise FileNotFoundError(f"The file {filename} does not exist.")
        else:
            with open(filename, 'rb') as file:
                node_filtered = pickle.load(file)[:, :top_num]

    else:
        all_nodes = np.arange(num_items)
        node_filtered = np.tile(all_nodes, (num_user, 1))


    node_filtered = node_filtered.astype(int)
    # print (node_filtered.shape)
    # print( num_user,num_items)
    # return  0

    hoped_num_process = min(48, multiprocessing.cpu_count())
    chunk_range = separate_to_chunks(np.arange(num_user), hoped_num_process)
    # print(num_user, len(chunk_range), chunk_range)
    pool = multiprocessing.Pool()

    results = []
    for i in range(len(chunk_range)):
        chunk = chunk_range[i]
        ratings_chunk = ratings[chunk[0]:chunk[1]]
        node_filtered_chunk = node_filtered[chunk[0]:chunk[1]]
        # print(i, ratings_chunk.shape, node_filtered_chunk.shape)

        result = pool.apply_async(get_recommendations_chunk,
                                  args=(k_param, distance_matrix, node_filtered_chunk, ratings_chunk))

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


        save_ranking_2file(datset_name, k_param, regime, seq_list, 'BKI_2')



    expectation_arr = np.concatenate(expectation_list)

    return np.average(expectation_arr), np.std(expectation_arr)

def get_recommendation_single(datset_name, filter_method, k_param, ratings, items_items_distances, mapping_range, regime, top_num):

    distance_matrix = items_items_distances
    distance_matrix = distance_matrix / np.max(distance_matrix)

    num_user, num_items = ratings.shape

    ratings = np.interp(ratings, (1, 5), mapping_range) # 0.1-0.3 for small regime, 0.4-0.6 for medium and 0.7-0.9 for large

    if filter_method in ['Dist', 'Dist_Prob', 'Matching', 'Relevance', ]:
        filename = 'Filtered_result/' + datset_name + '_' + filter_method + '.pkl'
        if not os.path.exists(filename):
            raise FileNotFoundError(f"The file {filename} does not exist.")
        else:
            with open(filename, 'rb') as file:
                node_filtered = pickle.load(file)[:, :top_num]


    elif filter_method in ['MSD']:
        filename = 'Filtered_result/' + datset_name + '_' + filter_method + '_' + regime + '.pkl'
        if not os.path.exists(filename):
            raise FileNotFoundError(f"The file {filename} does not exist.")
        else:
            with open(filename, 'rb') as file:
                node_filtered = pickle.load(file)[:, :top_num]

    else:
        all_nodes = np.arange(num_items)
        node_filtered = np.tile(all_nodes, (num_user, 1))

    node_filtered = node_filtered.astype(int)

    hoped_num_process = min(5, multiprocessing.cpu_count())
    chunk_range = separate_to_chunks(np.arange(num_user), hoped_num_process)

    results = []
    for i in range(len(chunk_range)):
        chunk = chunk_range[i]
        ratings_chunk = ratings[chunk[0]:chunk[1]]
        node_filtered_chunk = node_filtered[chunk[0]:chunk[1]]
        # print(ratings_chunk.shape, node_filtered_chunk.shape)
        result = get_recommendations_chunk(k_param, distance_matrix, node_filtered_chunk, ratings_chunk, top_num)

        results.append(result)

    expectation_list = []
    for i, seq_list in enumerate(results):
        chunk = chunk_range[i]
        ratings_chunk_exp = ratings[chunk[0]:chunk[1]]

        seq_chunk_arr = lib.to_object_array(seq_list).astype(int)
        exp_torch_arr = expectation_value_MSD_torch(seq_chunk_arr, distance_matrix, ratings_chunk_exp)
        expectation_list.append(exp_torch_arr.numpy())

        save_ranking_2file(datset_name, k_param, regime, ranking=seq_list)

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

    users_dictionary, items_dictionary, items_items_distances = get_structures(dataset_name,
                                                                               jaccard_distance=jaccard_distance,
                                                                               folder="../outputs")

    # chosen_users = np.random.choice(np.arange(n_users), 40, replace=False)
    # chosen_items = np.random.choice(np.arange(n_items), 50, replace=False)
    # ratings = np.load(f"../OMSD/rating_{dataset_name}.npy")[:len(chosen_users)]
    # distance_matrix = items_items_distances[chosen_items,:][:, chosen_items]



    ratings = np.load(f"../OMSD/rating_{dataset_name}.npy")
    distance_matrix = items_items_distances



    regimes_mapping = {"small": [0.1, 0.3], "medium": [0.4, 0.6], "large": [0.7, 0.9]}
    regimes = [ "large", "medium","small"]

    filter_method_list = ['NoFilter']

    strategy = 'CIKM_bkI_2'

    for regime in regimes:
        mapping_range = regimes_mapping[regime]

        for filter_method in filter_method_list:

            for k_param in [2]:


                top_num = 60

                start = time.time()

                exp_avg, exp_std = get_recommendation(dataset_name, filter_method, k_param, ratings, distance_matrix, mapping_range, regime, top_num)

                print('expectation for strategy ', filter_method, " with k = ", k_param,
                      " is ", exp_avg, exp_std, ' for dataset ',
                      dataset_name, ' for regime ', mapping_range, ' filter method ', filter_method, 'spend time ', time.time()-start)

                rst = (dataset_name, regime, k_param, filter_method, exp_avg, exp_std, time.time() - start)

                result_path = f"../results_new/{strategy}.txt"
                with open(result_path, 'a+') as file:
                    row_str = '\t'.join(map(str, rst))
                    file.write(row_str + '\n')


if __name__ == '__main__':

    datasets = {"movielens", "KuaiRec", "coat", "yahoo", "netflix"}
    datasets = ["coat", "netflix","movielens", "KuaiRec", "yahoo"]

    dataset_result = []
    for dataset in datasets:
        print(dataset)
        main(dataset)

