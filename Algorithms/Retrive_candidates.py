# implement many filtering method, try to improve the bke-H method performance.
# if the filtering is good enough, then bke-H (when k is large) should give very good performance.

import numpy as np
import torch
import time
import os
from tqdm import tqdm
import sys
import multiprocessing
from itertools import combinations, permutations
import copy
parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent)


from utility import get_structures, get_jaccard_matrix, get_items_genres_matrix
from util_expectations import *
import random
import pickle as pkl

def filter_by_relevance(rating_matrix, top_num_for_filtering):
    '''
    calculate the distance from each node to all other node, and return the top items
    '''

    top_items = np.argsort(rating_matrix, axis=1)[:, -top_num_for_filtering:]

    return top_items

def filter_by_SumOfDist(num_user, distance_matrix, top_num_for_filtering):
    '''
    calculate the distance from each node to all other node, and return the top items
    '''
    sum_distance = np.sum(distance_matrix, axis = 1)
    top_items = np.argsort(-sum_distance)[:top_num_for_filtering]


    return_arr = np.tile(top_items, (num_user, 1))
    return return_arr


def filter_by_SumOfDist_2HighProb( distance_matrix, rating_matrix, top_num_for_prob, top_num_for_filtering):
    '''
    first filter out top few high probability items F1
    calculate sum of distance from each node to F1, rank them, return top items.
    '''
    num_user = rating_matrix.shape[0]
    num_item = distance_matrix.shape[0]

    top_k_indices_per_row_4highest_prob = np.argsort(rating_matrix, axis=1)[:, -top_num_for_prob:]

    top_distance_items = np.zeros((num_user, top_num_for_filtering))
    for u in range(num_user):
        distance_matrix_copy = copy.deepcopy(distance_matrix)
        high_prob_items = top_k_indices_per_row_4highest_prob[u]

        mask = np.zeros(num_item, dtype=bool)
        mask[high_prob_items] = True
        distance_matrix_copy[~mask, :] = 0
        distance_matrix_copy[:, ~mask] = 0

        sum_distance = np.sum(distance_matrix_copy, axis=1)
        top_items = np.argsort(sum_distance)[-top_num_for_filtering:]

        top_distance_items[u,:] = top_items

    return top_distance_items

def filter_by_greedy_MHP(rating_matrix, distance_matrix, Thres4Greedy):
    num_user = rating_matrix.shape[0]
    num_items = distance_matrix.shape[0]

    Filtered_items = np.zeros((num_user, Thres4Greedy))
    for u in range(num_user):
    # for u in tqdm(range(num_user)):
        user_prob = rating_matrix[u]

        S = []
        U = set(np.arange(num_items))

        max_exp = 0
        best_seq = []
        for idx in range(num_items):
            # start item set as idx, and the rest added greedily.
            S.append(idx)
            U.remove(idx)

            while not len(S) >= Thres4Greedy:
                Candidate_seqs = np.zeros((len(U), len(S)+1), dtype=int)
                Candidate_seqs[:, :len(S)] = np.array(S)
                Candidate_seqs[:,-1] = np.array(list(U))

                # rank all candidate by its expectation.
                Candidat_exps = expectation_MHP_incremental(Candidate_seqs, distance_matrix, user_prob)

                best_candidate_idx = np.argmax(Candidat_exps)
                best_candidate = Candidate_seqs[best_candidate_idx][-1]

                S.append(best_candidate)
                U.remove(best_candidate)

            seq_exp = expectation_MHP_incremental(np.array([S]), distance_matrix, user_prob)
            if seq_exp > max_exp:
                max_exp = seq_exp
                best_seq = S

            # reset S and U for next candidate sequence.
            S = []
            U = set(np.arange(num_items))

        assert len(best_seq) != 0
        Filtered_items[u] = best_seq


    return Filtered_items


def filter_by_greedy_MSD(ratings, distance_matrix, threshold):

    def extend_greedy_prefix(best_nodes, threshold):
        # the greedy increment is the MSD increment instead of the MHP increment.
        # given the best k node, extend it to k+1 by maximizign MSD obj value -- case 1
        # find best (k+1) node that optimize the MHP obj value -- case 2
        # MSD_obj(case 1) > MSD(case 2).

        # it is necessary to compare BKI with greedy MSD algorithm.

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

    filtered_items = np.zeros((num_user, threshold))

    for u in tqdm(range(num_user)):
        user_prob = ratings[u]

        all_2_exps = expectation_value_MSD_incremental(all_2_permutations, distance_matrix, user_prob)
        best_indices = np.argmax(all_2_exps)
        best_2item = all_2_permutations[best_indices]

        best_candidate = np.array([[best_2item[0], best_2item[1]],[best_2item[1], best_2item[0]]])
        all_extended = []
        for ranking in best_candidate:
            extended_ranking = extend_greedy_prefix(ranking, threshold)
            all_extended.append(extended_ranking)

        all_extended = np.array(all_extended)

        all_exp = expectation_value_MSD_incremental(np.array(all_extended), distance_matrix, user_prob)
        best_RANKING_idx = np.argmax(all_exp)
        best_ranking = all_extended[best_RANKING_idx]

        filtered_items[u] = best_ranking

    return filtered_items



def filter_by_greedy_matching(num_user, adj_matrix, filtering_threshold): # adj_matrix is the distance_matrix

    n = len(adj_matrix)

    # List to store edges and their weights
    edges = []

    # Collect all edges with their weights
    for i in range(n):
        for j in range(i + 1, n):
            if adj_matrix[i][j] > 0:
                edges.append((adj_matrix[i][j], i, j))

    # Sort edges by weight in decreasing order
    edges.sort(reverse=True, key=lambda x: x[0])

    # To keep track of vertices already in the matching
    matched = set()
    matched_list = []
    matching = []

    # Form a maximal matching
    for weight, u, v in edges:
        if u not in matched and v not in matched:
            matching.append((u, v))
            matched.add(u)
            matched.add(v)
            matched_list.append(u)
            matched_list.append(v)


    return_arr = np.tile(matched_list[:filtering_threshold], (num_user, 1))
    return return_arr

###############################################################################################




def search_best_k_nodes_fromFilters(filtered_nodes, k_param, items_continue_p, distance_matrix, num_items):

    def extend_bestEdge_to_path_arbtrary(best_nodes):
        num_items = distance_matrix.shape[0]
        chosen_nodes = list(best_nodes)
        all_items = np.arange(num_items)
        rest_nodes = list(set(all_items) - set(chosen_nodes))
        random.shuffle(rest_nodes)
        chosen_nodes.extend(rest_nodes)

        return chosen_nodes

    def extend_bestEdge_to_path_greedy(best_nodes):

        sequence = list(best_nodes)
        chosen_node_accum_p = np.prod(items_continue_p[np.array(sequence)]) #scala

        accum_p_vec = items_continue_p * chosen_node_accum_p #vec
        # accum_p_vec[np.array(sequence)] = np.NINF
        accum_p_vec[np.array(sequence)] = -10000

        dis_all_2_chosen = np.sum(distance_matrix[:, np.array(sequence)], axis=1) #vec

        all_items = np.arange(num_items)
        while len(sequence) < num_items:


            gain = np.multiply(accum_p_vec,dis_all_2_chosen)

            best_node_index = np.argmax(gain)
            best_node = all_items[best_node_index]
            sequence.append(best_node)

            accum_p_vec = accum_p_vec * items_continue_p[best_node]
            # accum_p_vec[best_node] = np.NINF
            accum_p_vec[best_node] = -10000

            dis_all_2_chosen += distance_matrix[:, best_node]

        return sequence

    all_k_permutation = generate_combinations_and_permutations(filtered_nodes, k_param)
    # exp_values = expectation_value_MHP_torch(all_k_permutation, distance_matrix, items_continue_p)
    exp_values = expectation_value_MHP(all_k_permutation, distance_matrix, items_continue_p)

    best_permu_idx = np.argmax(exp_values)
    best_permu = all_k_permutation[best_permu_idx]


    sequence = extend_bestEdge_to_path_greedy(best_permu)
    # sequence = extend_bestEdge_to_path_arbtrary(best_permu)

    return sequence



def get_recommendations_chunk(k_param, distance_matrix, filtered_nodes, ratings_chunk):
    num_items = distance_matrix.shape[0]
    seq_chunk = []
    for idx in range(ratings_chunk.shape[0]):
        items_continue_p = ratings_chunk[idx]
        top_node_for_u = filtered_nodes[idx]
        best_sequence = search_best_k_nodes_fromFilters(top_node_for_u, k_param, items_continue_p, distance_matrix,
                                                        num_items)
        seq_chunk.append(best_sequence)

    return seq_chunk


def save_filtering2file_1(dataset_name, ratings, items_items_distances, mapping_range):

    distance_matrix = items_items_distances
    distance_matrix = distance_matrix / np.max(distance_matrix)

    ratings = np.interp(ratings, (1, 5), mapping_range) # 0.1-0.3 for small regime, 0.4-0.6 for medium and 0.7-0.9 for large


    folder_path = 'Filtered_result/'+ dataset_name + '_'

    node_filtered = filter_by_relevance(ratings, 100)
    filename = 'Relevance.pkl'
    with open(folder_path+filename, 'wb') as file:
        pkl.dump(node_filtered, file)

    node_filtered = filter_by_SumOfDist(ratings.shape[0],distance_matrix, 100)
    filename = 'Dist.pkl'
    with open(folder_path+filename, 'wb') as file:
        pkl.dump(node_filtered, file)

    node_filtered = filter_by_SumOfDist_2HighProb(distance_matrix, ratings, 500, 100)
    filename = 'Dist_Prob.pkl'
    with open(folder_path+filename, 'wb') as file:
        pkl.dump(node_filtered, file)

    node_filtered = filter_by_greedy_matching(ratings.shape[0], distance_matrix, 100)
    filename = 'Matching.pkl'
    with open(folder_path+filename, 'wb') as file:
        pkl.dump(node_filtered, file)

# ####################################################################################
#
#     print('processing MHP')
#
#     num_user = ratings.shape[0]
#
#     hoped_num_process = multiprocessing.cpu_count()
#     each_pool_size = int(np.floor(num_user / hoped_num_process))
#     actual_process = int(np.ceil(num_user / each_pool_size))
#
#     chunk_range = [np.array([0, each_pool_size - 1]) + each_pool_size * a for a in range(actual_process - 1)]
#     if chunk_range[-1][1] < num_user - 1:
#         chunk_range.append([chunk_range[-1][1] + 1, num_user - 1])
#
#
#
#     print ('num_of_thread = ', len(chunk_range), 'each trunk size is ', each_pool_size)
#
#     pool = multiprocessing.Pool()
#
#     results = []
#     for i in range(len(chunk_range)):
#         chunk = chunk_range[i]
#         ratings_chunk = ratings[chunk[0]:chunk[1] + 1]
#         result = pool.apply_async(filter_by_greedy_MHP,
#                                  args=(ratings_chunk, distance_matrix, 100))
#         results.append(result)
#
#     pool.close()
#     pool.join()
#
#     result_list = []
#     for result in results:
#         filtered_chunk = result.get()
#         result_list.append(filtered_chunk)
#
#     node_filtered = np.concatenate(result_list, axis=0)
#
#     pool.terminate()
#
#
#     filename = 'MHP.pkl'
#     with open(folder_path+filename, 'wb') as file:
#         pkl.dump(node_filtered, file)
#
#
# ####################################################################################




def save_filtering2file_2(dataset_name, ratings, items_items_distances, mapping_range, regime):

    distance_matrix = items_items_distances
    distance_matrix = distance_matrix / np.max(distance_matrix)

    ratings = np.interp(ratings, (1, 5), mapping_range) # 0.1-0.3 for small regime, 0.4-0.6 for medium and 0.7-0.9 for large


    folder_path = 'Filtered_result/'+ dataset_name + '_'



    num_user, num_items = ratings.shape


    print('processing MSD')

    hoped_num_process = min(48, multiprocessing.cpu_count())
    chunk_range = separate_to_chunks(np.arange(num_user),hoped_num_process)
    pool = multiprocessing.Pool()

    results = []
    for i in range(len(chunk_range)):
        chunk = chunk_range[i]
        ratings_chunk = ratings[chunk[0]:chunk[1]]
        result = pool.apply_async(filter_by_greedy_MSD,
                                  args=(ratings_chunk, distance_matrix, 100))
        results.append(result)

    pool.close()
    pool.join()

    result_list = []
    for result in results:
        filtered_chunk = result.get()
        result_list.append(filtered_chunk)

    node_filtered = np.concatenate(result_list, axis=0)

    pool.terminate()

    filename = 'MSD_'+ regime + '.pkl'
    with open(folder_path+filename, 'wb') as file:
        pkl.dump(node_filtered, file)

####################################################################################
def main1(dataset_name):
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

    items_genres_path = f"../outputs/items_genres_matrix_{dataset_name}.npy"
    items_genres_matrix = get_items_genres_matrix(items_dictionary, dataset_name, None, items_genres_path)

    users, items = list(users_dictionary.values()), list(items_dictionary.values())
    n_users, n_items = len(users), len(items)

    ratings = np.load(f"../OMSD/rating_{dataset_name}.npy")

    regimes_mapping = {"small": [0.1, 0.3], "medium": [0.4, 0.6], "large": [0.7, 0.9]}
    regimes = ["medium"]

    for regime in regimes:
        mapping_range = regimes_mapping[regime]
        save_filtering2file_1(dataset_name, ratings, items_items_distances, mapping_range)


def main2(dataset_name):
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

    items_genres_path = f"../outputs/items_genres_matrix_{dataset_name}.npy"
    items_genres_matrix = get_items_genres_matrix(items_dictionary, dataset_name, None, items_genres_path)

    users, items = list(users_dictionary.values()), list(items_dictionary.values())
    n_users, n_items = len(users), len(items)

    ratings = np.load(f"../OMSD/rating_{dataset_name}.npy")

    regimes_mapping = {"small": [0.1, 0.3], "medium": [0.4, 0.6], "large": [0.7, 0.9]}
    regimes = ["medium","small", "large"]


    for regime in regimes:
        mapping_range = regimes_mapping[regime]
        save_filtering2file_2(dataset_name, ratings, items_items_distances, mapping_range, regime)

if __name__ == '__main__':

    strategy = 'bke_filter_arbtray_extend'
    datasets = {"movielens", "KuaiRec", "coat", "yahoo", "netflix"}
    datasets = ["movielens", "KuaiRec", "coat", "yahoo", "netflix"]
    datasets = [ "coat", "netflix","movielens","yahoo", "KuaiRec"]
    # datasets = ["movielens","yahoo", "KuaiRec"]
    # datasets = [  "netflix","movielens","yahoo", "KuaiRec"]

    # dataset_result = []
    # for dataset in datasets:
    #     print(dataset)
    #     results = main1(dataset)

    for dataset in datasets:
        print(dataset)
        results = main2(dataset)
