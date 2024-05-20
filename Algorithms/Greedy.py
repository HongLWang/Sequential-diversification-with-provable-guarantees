import numpy as np
import torch
import time
import os
from tqdm import tqdm
import sys

parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

sys.path.append(parent)



import multiprocessing
from itertools import combinations, permutations
from utility import get_structures, get_jaccard_matrix, get_items_genres_matrix
import  pandas._libs.lib as lib
import torch, pickle
import random
from util_expectations import save_ranking_2file
#######################################################################################

def expectation_value_MSD_torch(permutations, distance_matrix, items_continue_p):

    permutations = torch.tensor(permutations)
    permutations = permutations.to(torch.int64)
    distance_matrix = torch.tensor(distance_matrix)
    items_continue_p = torch.tensor(items_continue_p)


    num_permutation, num_items = permutations.shape
    matrix_p = items_continue_p.gather(1, permutations)  # size is num of permutations * k_param

    # [p1p2, p1p2p3, p1p2p3p4, \cdots, p1...p_n]
    double_matrix_p = matrix_p.double()
    accumulate_p = torch.cumprod(double_matrix_p, dim=1)[:, 1:]  # size is num of permutations * k_param-1

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

def expectation_value_MSD_incremental_mine(permutations, distance_matrix, items_continue_p):

    num_permutation, num_items = permutations.shape

    first_two_column = permutations[:, :2]


    first_2_p = items_continue_p[first_two_column]
    accum_p = np.prod(first_2_p, axis=1)

    dist = distance_matrix[(first_two_column[:,0], first_two_column[:,1])]

    first_2_expectation = np.multiply(accum_p, dist)

    for idx in range(2, num_items):
        new_item = permutations[:, idx]
        new_p = items_continue_p[new_item]
        accum_p = np.multiply(accum_p, new_p)

        # indices_row = new_item[:, np.newaxis]  # Shape n*1

        dist_new2exist = distance_matrix[new_item[:, np.newaxis], first_two_column]  # indices_column = indices, indices_row = new_item
        dist_increment = np.sum(dist_new2exist, axis = 1)

        first_2_expectation +=  np.multiply(accum_p, dist_increment)  # this amount is the distance increment

        first_two_column = np.concatenate((first_two_column, new_item[:, np.newaxis]), axis=1)

    return first_2_expectation




# this is accelarated by chatgpt
def expectation_value_MSD_incremental(permutations, distance_matrix, items_continue_p):
    num_permutation, num_items = permutations.shape

    # Initialize results and accumulate initial probabilities
    first_2_p = items_continue_p[permutations[:, 0]] * items_continue_p[permutations[:, 1]]
    dist = distance_matrix[permutations[:, 0], permutations[:, 1]]
    expectation = first_2_p * dist

    # For storing accumulated probabilities
    accum_p = first_2_p

    for idx in range(2, num_items):
        new_item = permutations[:, idx]
        new_p = items_continue_p[new_item]
        accum_p *= new_p  # Update the accumulated probabilities

        # Compute the sum of distances from the new item to all previously considered items
        dist_increment = np.zeros(num_permutation)
        for prev_idx in range(idx):  # Loop over all previous items
            prev_item = permutations[:, prev_idx]
            dist_increment += distance_matrix[new_item, prev_item]

        # Update expectation
        expectation += accum_p * dist_increment

    return expectation


def expectation_value_MSD(permutations, distance_matrix, items_continue_p):

    num_permutation, num_items = permutations.shape
    matrix_p = items_continue_p[permutations]  # size is num of permutations * k_param

    # [p1p2, p1p2p3, p1p2p3p4, \cdots, p1...p_n]
    accumulate_p = np.cumprod(matrix_p, axis=1)[:, 1:]  # size is num of permutations * k_param-1

    # [d(1,2), d(3,{1,2}), d(4,{1,2,3}), ...d(n, {1,2,...,n-1})], the i-th element is d(i+1, {1,2,...i})
    distances = np.zeros((num_permutation, num_items - 1))

    for i in range(1, num_items):
        new_node = permutations[:, i]
        existing_nodes = permutations[:, :i]

        matrix_row_slices = distance_matrix[new_node]
        row_indices = np.arange(existing_nodes.shape[0])[:, None]  # Extend dimensions to align with B for broadcasting
        matrix_column_slices = matrix_row_slices[row_indices, existing_nodes]

        distance_new2exist = np.sum(matrix_column_slices, axis=1)
        distances[:, i - 1] = distance_new2exist # size is num of permutations * k_param-1


    expectations = np.einsum('ij,ij->i', accumulate_p, distances)


    return expectations


def generate_combinations_and_permutations(all_items, k_param):
    result = []
    # print ('items before permutation', all_items)
    for c in combinations(all_items, k_param):
        result.extend(permutations(c))

    all_permutations_array =  lib.to_object_array(result).astype(int)
    return all_permutations_array


def greedy_algotithm(user_prob, distance_matrix):


    def extend_bestEdge_to_path_greedy_MSD(best_nodes):
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


    num_items = distance_matrix.shape[0]
    all_2_permutations = generate_combinations_and_permutations(np.arange(num_items), 2)
    all_2_exps = expectation_value_MSD_incremental(all_2_permutations, distance_matrix, user_prob)

    # max_exp_value = np.max(all_2_exps)
    # indices = np.where(all_2_exps == max_exp_value)[0]
    # best_2_rankings = all_2_permutations[indices]

    best_indices = np.argmax(all_2_exps)
    best_2item = all_2_permutations[best_indices]

    best_candidate = np.array([[best_2item[0], best_2item[1]],[best_2item[1], best_2item[0]]])

##################################################################################################################
    # # extend to a full ranking and pick the best
    # all_extended = []
    # for ranking in best_2_rankings:
    #     extended_ranking = extend_bestEdge_to_path_greedy_MSD(ranking)
    #     all_extended.append(extended_ranking)
    #
    # all_extended = np.array(all_extended)
    #
    # all_exp = expectation_value_MSD(np.array(all_extended), distance_matrix, user_prob)
    # best_RANKING_idx = np.argmax(all_exp)
    # best_ranking = all_extended[best_RANKING_idx]

##################################################################################################################

##################################################################################################################

    # extend to a  ranking of len(threashold) and pick the best, to make it faster.
    all_extended = []
    for ranking in best_candidate:
        extended_ranking = extend_greedy_prefix(ranking, 40)
        all_extended.append(extended_ranking)

    all_extended = np.array(all_extended)

    all_exp = expectation_value_MSD_incremental(np.array(all_extended), distance_matrix, user_prob)
    best_RANKING_idx = np.argmax(all_exp)
    best_ranking = all_extended[best_RANKING_idx]

    best_full_ranking = extend_bestEdge_to_path_greedy_MSD(best_ranking)

##################################################################################################################

    return best_full_ranking

def get_recommendations_chunk(distance_matrix, ratings_chunk):
    seq_chunk = []
    for idx in tqdm(range(ratings_chunk.shape[0])):
        items_continue_p = ratings_chunk[idx]
        best_sequence = greedy_algotithm(items_continue_p, distance_matrix)
        seq_chunk.append(best_sequence)

    return seq_chunk

def separate_to_chunks(lst, k):
    chunk_size = len(lst) // k
    remainder = len(lst) % k

    start_index = 0
    end_index = 0
    chunk_indices = []
    for i in range(k):
        if i < remainder:
            end_index += chunk_size + 1
        else:
            end_index += chunk_size

        if end_index > len(lst):
            end_index = len(lst)

        chunk_indices.append((start_index, end_index))
        start_index = end_index

    return chunk_indices

def get_recommendation_single_thred( ratings, items_items_distances, mapping_range):

    distance_matrix = items_items_distances
    distance_matrix = distance_matrix / np.max(distance_matrix)

    num_user, num_items = ratings.shape

    ratings = np.interp(ratings, (1, 5), mapping_range) # 0.1-0.3 for small regime, 0.4-0.6 for medium and 0.7-0.9 for large



    hoped_num_process = min(4, multiprocessing.cpu_count())
    chunk_range = separate_to_chunks(np.arange(num_user), hoped_num_process)



    results = []
    for i in range(len(chunk_range)):
        print('processing chunk ', i , 'out of total ', len(chunk_range))
        chunk = chunk_range[i]
        ratings_chunk = ratings[chunk[0]:chunk[1]]
        result = get_recommendations_chunk(distance_matrix, ratings_chunk)

        results.append(result)


    expectation_list = []
    for i, seq_list in enumerate(results):
        chunk = chunk_range[i]
        ratings_chunk_exp = ratings[chunk[0]:chunk[1]]

        seq_chunk_arr = lib.to_object_array(seq_list).astype(int)
        exp_torch_arr = expectation_value_MSD_torch(seq_chunk_arr, distance_matrix, ratings_chunk_exp)
        expectation_list.append(exp_torch_arr.numpy())

    expectation_arr = np.concatenate(expectation_list)
    exp_deriviation = np.std(expectation_arr)
    average_exp = np.average(expectation_arr)

    return average_exp, exp_deriviation

def get_recommendation( ratings, items_items_distances, mapping_range, regime, dataset_name):

    distance_matrix = items_items_distances
    distance_matrix = distance_matrix / np.max(distance_matrix)

    num_user, num_items = ratings.shape

    ratings = np.interp(ratings, (1, 5), mapping_range) # 0.1-0.3 for small regime, 0.4-0.6 for medium and 0.7-0.9 for large

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


    ratings = np.load(f"../OMSD/rating_{dataset_name}.npy")


    regimes_mapping = {"small": [0.1, 0.3], "medium": [0.4, 0.6], "large": [0.7, 0.9]}
    regimes = ["small", "medium", "large"]
    # regimes = ["medium"]


    strategy = 'simple_greedy'


    for regime in regimes:
        mapping_range = regimes_mapping[regime]

        start = time.time()

        # average_exp, exp_deriviation = get_recommendation_single_thred(ratings, items_items_distances, mapping_range)
        average_exp, exp_deriviation = get_recommendation(ratings, items_items_distances, mapping_range, regime, dataset_name)


        print('expectation for strategy ', strategy,
              " is ", average_exp, ' std is ', exp_deriviation, ' for dataset ',
              dataset_name, ' for regime ', mapping_range, 'spend time ', time.time()-start)

        rst = (dataset_name, regime, average_exp, exp_deriviation, time.time() - start)

        result_path = f"../results_new/{strategy}.txt"
        with open(result_path, 'a+') as file:
            row_str = '\t'.join(map(str, rst))
            file.write(row_str + '\n')


if __name__ == '__main__':

    datasets = {"movielens", "KuaiRec", "coat", "yahoo", "netflix"}
    datasets = ["coat", "netflix","movielens", "KuaiRec", "yahoo"]
    # datasets = ["movielens", "KuaiRec", "yahoo"]
    datasets = ["coat", "netflix"]

    dataset_result = []
    for dataset in datasets:
        main(dataset)
        print(dataset)


'''
multi thread version
'''
