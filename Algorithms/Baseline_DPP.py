import pandas as pd

import time, random
import os, math
import sys
import  pandas._libs.lib as lib
import torch, multiprocessing
import numpy as np
parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

sys.path.append(parent)

from utility import get_structures

from util_expectations import separate_to_chunks, save_ranking_2file

def get_recommendations(ratings, sim_matrix, distance_matrix, theta_factor, mapping_range, regime, dataset_name):  # this is multi-thread version.

    num_user, num_items = ratings.shape

    candidates_relevance = np.interp(ratings, (1, 5),
                                     mapping_range)  # 0.1-0.3 for small regime, 0.4-0.6 for medium and 0.7-0.9 for large

    similarity = sim_matrix


    for test_poolsize in [128]:
        each_pool_size = test_poolsize
        actual_process = int(np.ceil(num_user / each_pool_size))
        chunk_range = separate_to_chunks(np.arange(num_user), actual_process)


        expectation_list = []
        for i in range(len(chunk_range)):

            if i % 50 == 0:
                print(i, '-th of ', actual_process, ' chunck processing')
            chunk = chunk_range[i]
            ratings_chunk = candidates_relevance[chunk[0]:chunk[1]]

            seq_list = get_recommendations_chunk(ratings_chunk, similarity, theta_factor)

            if len(seq_list) == 0 :
                debug = 'stop here'
                print (debug)


            seq_chunk_arr = lib.to_object_array(seq_list).astype(int)

            exp_torch_arr = expectation_value_MSD_torch(seq_chunk_arr, distance_matrix, ratings_chunk).numpy()
            expectation_list.append(exp_torch_arr)

            save_ranking_2file(dataset_name, 0, regime, seq_chunk_arr, 'dpp')

        expectation_list = np.concatenate(expectation_list)

        return np.average(expectation_list), np.std(expectation_list)

def get_recommendations_chunk(relevance_chunk, similarity, theta_factor):
    seq_list = []
    num_items = relevance_chunk.shape[1]

    for i in range(len(relevance_chunk)):
        relevance = relevance_chunk[i]

        L = get_kernel_matrix(relevance, similarity, theta_factor)

        user_recommendation_list = dpp(L, num_items)

        seq_list.append(user_recommendation_list)

    return seq_list

def get_kernel_matrix(relevance, similarity, theta_factor):  # kernel matrix

    alpha = theta_factor / 2 / (1 - theta_factor)
    relevance = math.e ** (alpha * relevance)
    item_size = len(relevance)
    # kernel_matrix = relevance.reshape((item_size, 1)) * similarity * relevance.reshape((1, item_size))
    kernel_matrix = relevance.reshape((item_size, 1)) * similarity * relevance.reshape((1, item_size))
    # check_positive(kernel_matrix)
    return kernel_matrix

def check_positive(matrix): # check whether matrix is positive semi-definite
    L = np.linalg.cholesky(matrix)
    return L


def dpp(L, k, epsilon=1e-8):

    def extend_bestEdge_to_path_arbtrary(best_nodes):
        num_items = L.shape[0]
        chosen_nodes = list(best_nodes)
        all_items = np.arange(num_items)
        rest_nodes = list(set(all_items) - set(chosen_nodes))
        random.shuffle(rest_nodes)
        chosen_nodes.extend(rest_nodes)

        return chosen_nodes

    """
    Our proposed fast implementation of the greedy algorithm
    :param kernel_matrix: 2-d array
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list
    """
    item_size = L.shape[0]
    cis = np.zeros((k, item_size))
    di2s = np.copy(np.diag(L))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    while len(selected_items) < k:
        s = len(selected_items) - 1
        ci_optimal = cis[:s, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = L[selected_item, :]

        if not di_optimal:
            eis = 0
        else:
            eis = (elements - np.dot(ci_optimal, cis[:s, :])) / di_optimal

        cis[s, :] = eis
        di2s -= np.square(eis)
        di2s[selected_item] = -np.inf
        selected_item = np.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break


        if len(selected_items)> 50: # only the top few items are important for calculating expectations.
            break

        selected_items.append(selected_item)

    extendend_list =  extend_bestEdge_to_path_arbtrary(selected_items)

    # extend the rest items arbitrarily to selected_items such that len(selected_items) = num_items.
    return extendend_list




def expectation_value_MSD_torch(permutations, distance_matrix, items_continue_p):

    permutations = torch.tensor(permutations)
    distance_matrix = torch.tensor(distance_matrix)
    items_continue_p = torch.tensor(items_continue_p)

    num_permutation, num_items = permutations.shape
    matrix_p = items_continue_p.gather(1, permutations)  # size is num of permutations * k_param

    # [p1p2, p1p2p3, p1p2p3p4, \cdots, p1...p_n]
    double_type_matrix_p = matrix_p.double()
    accumulate_p = torch.cumprod(double_type_matrix_p, dim=1)[:, 1:]  # size is num of permutations * k_param-1

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




def main(dataset_name):

    jaccard_distances_dict = {"movielens": "genres", "KuaiRec": "users",
                              "coat": "genres", "yahoo": "genres", "netflix": "genres"}

    jaccard_distance = jaccard_distances_dict[dataset_name]

    # HERE THE OUTPUTS FOLDER
    folder = f"../outputs/jaccard_{jaccard_distance}"

    if not os.path.exists(folder):
        os.makedirs(folder)


    users_dictionary, items_dictionary, items_items_distances = get_structures(dataset_name,
                                                                               jaccard_distance=jaccard_distance,
                                                                               folder="../outputs")
    sim_matrix = 1 - items_items_distances


    ratings = np.load(f"../OMSD/rating_{dataset_name}.npy")


    regimes_mapping = {"small": [0.1, 0.3], "medium": [0.4, 0.6], "large": [0.7, 0.9]}
    regimes = ["small", "medium", "large"]


    distance_matrix = items_items_distances / np.max(items_items_distances)

    strategies = [ "CIKM_DPP"]


    for regim in regimes:
        mapping_range = regimes_mapping[regim]

        for strategy in strategies:

            
            thetas = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]

            for theta_factor in [thetas]:

                start = time.time()
                average_exp, std_exp  = get_recommendations(ratings, sim_matrix, distance_matrix, theta_factor, mapping_range, dataset_name, regim)

                rst = (dataset_name, theta_factor, regim, average_exp, std_exp, time.time()-start)

                result_path = f"../results_new/{strategy}.txt"
                with open(result_path, 'a+') as file:
                    row_str = '\t'.join(map(str, rst))
                    file.write(row_str + '\n')

                print(' expectation ', average_exp, ' for dataset ', dataset_name, ' regime ', regim, ' spend time ', time.time()-start)

    # return  results

if __name__ == '__main__':

    strategy = "dpp"
    datasets = ["coat",  "netflix", "movielens", "KuaiRec" , "yahoo"]


    # regimes = ["small", "medium", "large"]


    known_best_param = {}
    for data_name in datasets:
        known_best_param[data_name] = {}

    known_best_param['coat']['small'] = 0.6
    known_best_param['coat']['medium'] = 0.8
    known_best_param['coat']['large'] = 0.99


    known_best_param['netflix']['small'] = 0.6
    known_best_param['netflix']['medium'] = 0.8
    known_best_param['netflix']['large'] = 0.99


    known_best_param['movielens']['small'] = 0.9
    known_best_param['movielens']['medium'] = 0.9
    known_best_param['movielens']['large'] = 0.99


    known_best_param['yahoo']['small'] = 0.5
    known_best_param['yahoo']['medium'] = 0.7
    known_best_param['yahoo']['large'] = 0.99


    known_best_param['KuaiRec']['small'] = 0.8
    known_best_param['KuaiRec']['medium'] = 0.8
    known_best_param['KuaiRec']['large'] = 0.9


    # datasets = ["coat"]


    dataset_result = []
    for dataset in datasets:
        print (dataset)
        main(dataset)


