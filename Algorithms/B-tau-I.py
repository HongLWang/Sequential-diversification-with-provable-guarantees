# This is the "BtauI" algorithm in our paper, this code runs with \tau=2

import time
import os
import multiprocessing
import numpy as np
from tqdm import tqdm
from utility import *
import random, pickle
from utils import *


def Find_Best_ranking(k_param, items_continue_p, distance_matrix):

    def extend_bestEdge_to_path_greedy_MSD(best_nodes):

        sequence = list(best_nodes)
        chosen_node_accum_p = np.prod(items_continue_p[np.array(sequence)])  # scala

        accum_p_vec = items_continue_p * chosen_node_accum_p  # vec
        accum_p_vec[np.array(sequence)] = np.NINF

        dis_all_2_chosen = np.sum(distance_matrix[:, np.array(sequence)], axis=1)  # vec

        all_items = np.arange(num_item)
        while len(sequence) < num_item:
            gain = np.multiply(accum_p_vec, dis_all_2_chosen)

            best_node_index = np.argmax(gain)
            best_node = all_items[best_node_index]
            sequence.append(best_node)

            accum_p_vec = accum_p_vec * items_continue_p[best_node]
            accum_p_vec[best_node] = np.NINF

            dis_all_2_chosen += distance_matrix[:, best_node]

        return sequence

    num_item = distance_matrix.shape[0]
    all_k_permutation = generate_combinations_and_permutations(np.arange(num_item), k_param)
    exp_values = expectation_value_MHP(all_k_permutation, distance_matrix, items_continue_p)
    best_permu_idx = np.argmax(exp_values)
    best_permu = all_k_permutation[best_permu_idx]

    sequence = extend_bestEdge_to_path_greedy_MSD(best_permu)

    return sequence



def get_recommendations_chunk(k_param, distance_matrix, ratings_chunk):
    seq_chunk = []

    for idx in tqdm(range(ratings_chunk.shape[0])):
        items_continue_p = ratings_chunk[idx]
        best_sequence = Find_Best_ranking(k_param, items_continue_p, distance_matrix)
        seq_chunk.append(best_sequence)

    return seq_chunk

def get_recommendation(datset_name, k_param, ratings, items_items_distances, mapping_range, regime, strategy):

    distance_matrix = items_items_distances
    distance_matrix = distance_matrix / np.max(distance_matrix)

    num_user, num_items = ratings.shape

    ratings = np.interp(ratings, (1, 5), mapping_range)

    hoped_num_process = min(48, multiprocessing.cpu_count())
    chunk_range = separate_to_chunks(np.arange(num_user), hoped_num_process)
    pool = multiprocessing.Pool()

    results = []
    for i in range(len(chunk_range)):
        chunk = chunk_range[i]
        ratings_chunk = ratings[chunk[0]:chunk[1]]

        result = pool.apply_async(get_recommendations_chunk,
                                  args=(k_param, distance_matrix, ratings_chunk))

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


    regimes_mapping = {"small": [0.1, 0.3], "medium": [0.4, 0.6], "large": [0.7, 0.9]}
    regimes = [ "large", "medium","small"]


    strategy = 'B-tau-I'
    k_param = 2


    result_folder = 'Results'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    ranking_folder = 'ranking'  # save the permutation/ranking of items for each user, this is useful for visualization.
    if not os.path.exists(ranking_folder):
        os.makedirs(ranking_folder)

    for regime in regimes:
        mapping_range = regimes_mapping[regime]

        start = time.time()

        exp_avg, exp_std = get_recommendation(dataset_name, k_param, ratings, distance_matrix, mapping_range, regime, strategy)

        print('expectation for strategy ', strategy, " with k = ", k_param,
              " is ", exp_avg, exp_std, ' for dataset ',
              dataset_name, ' for regime ', mapping_range, 'spend time ', time.time()-start)

        rst = (dataset_name, regime, k_param, exp_avg, exp_std, time.time() - start)

        result_path = f"{result_folder}/{strategy}.txt"
        with open(result_path, 'a+') as file:
            row_str = '\t'.join(map(str, rst))
            file.write(row_str + '\n')


if __name__ == '__main__':

    datasets = ["coat", "netflix","movielens", "KuaiRec", "yahoo"]
    datasets = ["coat"]

    dataset_result = []
    for dataset in datasets:
        print(dataset)
        results = main(dataset)

