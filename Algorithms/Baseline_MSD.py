import pandas as pd
import time
import os
import sys
parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

sys.path.append(parent)


import multiprocessing
# import torch
from tqdm import tqdm
import numpy as np
import torch
# parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
# sys.path.append(parent)
import  pandas._libs.lib as lib

from utility import get_structures
from utility import get_jaccard_matrix, get_items_genres_matrix

from util_expectations import separate_to_chunks, save_ranking_2file

def expectation_value_MSD_torch(permutations, distance_matrix, items_continue_p):
    permutations = torch.tensor(permutations)
    distance_matrix = torch.tensor(distance_matrix)
    items_continue_p = torch.tensor(items_continue_p)

    num_permutation, num_items = permutations.shape
    matrix_p = items_continue_p.gather(1, permutations)  # size is num of permutations * k_param

    # [p1p2, p1p2p3, p1p2p3p4, \cdots, p1...p_n]
    accumulate_p = torch.cumprod(matrix_p, dim=1)[:, 1:]  # size is num of permutations * k_param-1

    # [d(1,2), d(3,{1,2}), d(4,{1,2,3}), ...d(n, {1,2,...,n-1})], the i-th element is d(i+1, {1,2,...i})
    distances = torch.zeros((num_permutation, num_items-1), dtype=torch.double)
    for i in range(1, num_items):
        new_node = permutations[:, i]
        existing_nodes = permutations[:, :i]

        matrix_row_slices = distance_matrix[new_node]
        matrix_column_slices = matrix_row_slices.gather(1, existing_nodes)

        distance_new2exist = torch.sum(matrix_column_slices, dim=1)
        distances[:, i-1] = distance_new2exist

    expectations = torch.einsum('ij,ij->i', accumulate_p, distances)

    return expectations


def expectation_value_MHP(permutations, distance_matrix, items_continue_p):
    matrix_p = items_continue_p[permutations]  # size is num of permutations * k_param
    accumulate_p = np.cumprod(matrix_p, axis=1)[:, 1:]  # size is num of permutations * k_param-1
    weights = np.cumsum(accumulate_p[:, ::-1], axis=1)[:, ::-1]  # wi = p1p2..p_{i+1} + ...+ p1p2..p_n

    indices = (permutations[:, :-1], permutations[:, 1:])
    distances = distance_matrix[indices]  # size is num of permutations * k_param-1

    expectations = np.sum(weights * distances, axis=1, keepdims=True)

    return expectations

def greedy_msd_vectorized(items_continue_p_, distance_matrix_, lambda_factor):
    n_items = distance_matrix_.shape[0]
    U = set(np.arange(n_items))
    S = []
    ss = np.zeros(n_items)

    phi_u_S = items_continue_p_
    best_u = np.argmax(phi_u_S)
    S.append(best_u)
    U -= {best_u}
    items_continue_p_[best_u] = 0
    distance_matrix_[best_u, :] = np.zeros(n_items)

    while len(U) > 0:
        ss = ss + np.squeeze(distance_matrix_[:,best_u])
        ss[best_u] = 0
        phi_u_S = items_continue_p_/2 + lambda_factor * ss
        best_u = np.argmax(phi_u_S)
        S.append(best_u)
        U -= {best_u}

        # UPDATE THE P AND D SO YOU DONT CHOSE THE PREVIOUS ITEMS AGAIN
        items_continue_p_[best_u] = 0
        distance_matrix_[best_u,:] = np.zeros(n_items)

    return S

def get_single_recommendation(distance_matrix, ratings_chunk, k_param, lambda_factor):
    # distance_matrix = items_items_distances
    seq_chunk = []
    # for i in range(len(ratings_chunk)):
    for items_continue_p in ratings_chunk:
        # candidates_relevance = ratings_chunk[i]

        distance_matrix = distance_matrix / np.max(distance_matrix)
        copy_distance_matrix = distance_matrix.copy()
        copy_items_continue_p = items_continue_p.copy()

        msd_sequence = greedy_msd_vectorized(copy_items_continue_p, copy_distance_matrix, lambda_factor)

        seq_chunk.append(msd_sequence)

    return seq_chunk


def get_recommendations(k_param, lambda_factor, ratings, items_items_distances, mapping_range, dataset_name , regime):
    num_user, num_items = ratings.shape

    results = []
    distance_matrix = items_items_distances
    distance_matrix = distance_matrix / np.max(distance_matrix)

    hoped_num_process = min(48, multiprocessing.cpu_count())
    chunk_range = separate_to_chunks(np.arange(num_user), hoped_num_process)
    pool = multiprocessing.Pool()

    ratings = np.interp(ratings, (1, 5), mapping_range)  # 0.1-0.3 for small regime, 0.4-0.6 for medium and 0.7-0.9 for large


    for i in range(len(chunk_range)):
        chunk = chunk_range[i]
        ratings_chunk = ratings[chunk[0]:chunk[1]]
        result = pool.apply_async(get_single_recommendation,
                                  args=(distance_matrix, ratings_chunk, k_param, lambda_factor))
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

        # print(len(seq_list))
        seq_chunk_arr = lib.to_object_array(seq_list).astype(int)
        exp_torch_arr = expectation_value_MSD_torch(seq_chunk_arr, distance_matrix, ratings_chunk_exp)
        expectation_list.append(exp_torch_arr)

        save_ranking_2file(dataset_name, lambda_factor, regime, seq_chunk_arr, 'MSD')

    expectation_list = np.concatenate(expectation_list)

    return np.average(expectation_list), np.std(expectation_list)


def main(dataset_name):

    jaccard_distances_dict = {"movielens": "genres", "KuaiRec": "users",
                              "coat": "genres", "yahoo": "genres", "netflix": "genres"}

    jaccard_distance = jaccard_distances_dict[dataset_name]

    folder = f"../outputs/jaccard_{jaccard_distance}"

    if not os.path.exists(folder):
        os.makedirs(folder)

    model_folder = "../models/checkpoints/"

    users_dictionary, items_dictionary, items_items_distances = get_structures(dataset_name,
                                                                               jaccard_distance=jaccard_distance,
                                                                               folder="../outputs")


    ratings = np.load(f"../OMSD/rating_{dataset_name}.npy")


    strategies = [
        "CIKM_MSD"
    ]

    regimes_mapping = {"small": [0.1, 0.3], "medium": [0.4, 0.6], "large": [0.7, 0.9]}
    regimes = ["small", "medium", "large"]


    for regime in regimes:
        mapping_range = regimes_mapping[regime]

        for strategy in strategies:
            k_param = 2

            lambda_factor_dict = [known_best_param[data_name][regime]]
            # for lambda_factor in list(np.arange(0, 1.1, 0.1)):
            for lambda_factor in lambda_factor_dict:

                start = time.time()
                exp_avg, exp_std = get_recommendations(k_param, lambda_factor, ratings, items_items_distances, mapping_range, dataset_name, regime)

                rst = (dataset_name, regime, exp_avg, exp_std, lambda_factor, time.time() - start)

                result_path = f"../results_new/{strategy}.txt"
                with open(result_path, 'a+') as file:
                    row_str = '\t'.join(map(str, rst))
                    file.write(row_str + '\n')

                print('MSD  expectation ', exp_avg, exp_std, ' for dataset ', dataset_name, ' regime ', regime,
                      ' spend time ',
                      time.time() - start)


if __name__ == '__main__':
    strategy = "msd"
    datasets = {"movielens", "KuaiRec", "coat", "yahoo", "netflix"}


    known_best_param = {}
    for data_name in datasets:
        known_best_param[data_name] = {}

    known_best_param['coat']['small'] = 0.1
    known_best_param['coat']['medium'] = 0.1
    known_best_param['coat']['large'] = 0

    known_best_param['netflix']['small'] = 0.5
    known_best_param['netflix']['medium'] = 0.1
    known_best_param['netflix']['large'] = 0

    known_best_param['movielens']['small'] = 0.5
    known_best_param['movielens']['medium'] = 0.1
    known_best_param['movielens']['large'] = 0.1

    known_best_param['yahoo']['small'] = 0.5
    known_best_param['yahoo']['medium'] = 0.5
    known_best_param['yahoo']['large'] = 0.5

    known_best_param['KuaiRec']['small'] = 0.1
    known_best_param['KuaiRec']['medium'] = 0.2
    known_best_param['KuaiRec']['large'] = 0.1

    # datasets = {"coat"}

    dataset_result = []
    for dataset in datasets:
        print (dataset)
        main(dataset)
