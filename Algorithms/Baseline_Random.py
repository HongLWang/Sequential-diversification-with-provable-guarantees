import random
import time
import os
import sys
import multiprocessing
from tqdm import tqdm
import numpy as np

parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

sys.path.append(parent)



from itertools import combinations, permutations
from utility import get_structures, get_jaccard_matrix, get_items_genres_matrix
import  pandas._libs.lib as lib
import torch
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

    # wi = p1p2..p_{i+1} + ...+ p1p2..p_n
    weights = np.cumsum(accumulate_p[:, ::-1], axis=1)[:, ::-1]

    indices = (permutations[:, :-1], permutations[:, 1:])

    # size is num of permutations * k_param-1
    distances = distance_matrix[indices]

    expectations = np.sum(weights * distances, axis=1, keepdims=True)

    return expectations


def Filter_Candidate_Nodes(item_continue_p, node_filtered_from_distance, num_items, filter_percentage):

    all_items = np.arange(num_items)

    num_after_fliter = int(num_items * filter_percentage)

    # filter nodes that have high probability to be checked
    rank_p_index = np.argsort(item_continue_p)[::-1] # reversed sort
    chosen_nodes = all_items[rank_p_index[:num_after_fliter]]
    filter_result = node_filtered_from_distance.intersection(chosen_nodes)

    return filter_result


def filter_by_distance(num_item, distance_matrix, filter_percentage):

    all_items = np.arange(num_item)

    node_pairs = []
    for c in combinations(all_items, 2):
        node_pairs.append(c)
    node_pairs = lib.to_object_array(node_pairs).astype(int)

    num_pair_to_choose = int(node_pairs.shape[0] * filter_percentage)
    distances = distance_matrix[(node_pairs[:,0], node_pairs[:,1])]
    node_pair_rank_index = np.argsort(distances)[::-1]
    chosen_node_pairs = node_pairs[node_pair_rank_index][:num_pair_to_choose]
    flattened_arr = [element for row in chosen_node_pairs for element in row]
    unique_nodes_in_node_pairs = set(flattened_arr)


    return unique_nodes_in_node_pairs

def generate_combinations_and_permutations(all_items, k_param):
    result = []
    for c in combinations(all_items, k_param):
        result.extend(permutations(c))

    all_permutations_array =  lib.to_object_array(result).astype(int)
    return all_permutations_array



def search_best_k_nodes(all_k_permutation, items_continue_p, distance_matrix, num_items):

    def extend_bestEdge_to_path_greedy(best_nodes):

        sequence = list(best_nodes)
        chosen_node_accum_p = np.prod(items_continue_p[np.array(sequence)]) #scala

        accum_p_vec = items_continue_p * chosen_node_accum_p #vec
        accum_p_vec[np.array(sequence)] = np.NINF

        dis_all_2_chosen = np.sum(distance_matrix[:, np.array(sequence)], axis=1) #vec

        all_items = np.arange(num_items)
        while len(sequence) < num_items:

            gain = np.multiply(accum_p_vec,dis_all_2_chosen)

            best_node_index = np.argmax(gain)
            best_node = all_items[best_node_index]
            sequence.append(best_node)

            accum_p_vec = accum_p_vec * items_continue_p[best_node]
            accum_p_vec[best_node] = np.NINF

            dis_all_2_chosen += distance_matrix[:, best_node]

        return sequence

    exp_values = expectation_value_MHP(all_k_permutation, distance_matrix, items_continue_p)

    best_permu_idx = np.argmax(exp_values)
    best_permu = all_k_permutation[best_permu_idx]


    sequence = extend_bestEdge_to_path_greedy(best_permu)

    return sequence


def get_recommendations_chunk(num_this_chunk, num_item):
    seq_chunk = []
    candidates = list(np.arange(num_item))
    for i in range(num_this_chunk):
        random.shuffle(candidates)
        seq_chunk.append(candidates)
    return seq_chunk




def get_recommendation(k_param, ratings, items_items_distances, mapping_range, dataset_name, regime):

    sim_matrix = 1- items_items_distances
    distance_matrix = items_items_distances
    distance_matrix = distance_matrix / np.max(distance_matrix)
    similarity = 1- distance_matrix

    num_user, num_items = ratings.shape

    ratings = np.interp(ratings, (1, 5), mapping_range) # 0.1-0.3 for small regime, 0.4-0.6 for medium and 0.7-0.9 for large

    results = []
    hoped_num_process = min(48, multiprocessing.cpu_count())
    chunk_range = separate_to_chunks(np.arange(num_user), hoped_num_process)
    pool = multiprocessing.Pool()

    for i in range(len(chunk_range)):
        chunk = chunk_range[i]
        ratings_chunk = ratings[chunk[0]:chunk[1]]
        num_this_chunk, num_item = ratings_chunk.shape
        result = pool.apply_async(get_recommendations_chunk,
                                  args=(num_this_chunk, num_item))

        results.append(result)

    pool.close()
    pool.join()

    result_list = []
    for result in results:
        seq_list = result.get()
        result_list.append(seq_list)

    # expecation = sum(result_list) / num_user

    pool.terminate()

    expectation_list = []
    for i, seq_list in enumerate(result_list):
        chunk = chunk_range[i]
        ratings_chunk_exp = ratings[chunk[0]:chunk[1]]

        # print(len(seq_list))
        seq_chunk_arr = lib.to_object_array(seq_list).astype(int)
        exp_torch_arr = expectation_value_MSD_torch(seq_chunk_arr, distance_matrix, ratings_chunk_exp)
        expectation_list.append(exp_torch_arr)

        save_ranking_2file(dataset_name, k_param, regime, seq_chunk_arr, 'Random')

    expectation_list = np.concatenate(expectation_list)

    return np.average(expectation_list), np.std(expectation_list)


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

    for regime in regimes:
        mapping_range = regimes_mapping[regime]
        k_param = 2
        start = time.time()
        exp_avg, exp_std = get_recommendation(k_param, ratings, items_items_distances, mapping_range, dataset_name, regime)

        rst = (dataset_name, regime, exp_avg, exp_std, time.time() - start)

        result_path = f"../results_new/CIKM_Random.txt"
        with open(result_path, 'a+') as file:
            row_str = '\t'.join(map(str, rst))
            file.write(row_str + '\n')

        print('Ramdom  expectation ', exp_avg, exp_std, ' for dataset ', dataset_name, ' regime ', regime,
              ' spend time ',
              time.time() - start)



if __name__ == '__main__':

    for strategy in ['random']:
        datasets = {"movielens", "KuaiRec", "coat", "yahoo", "netflix"}
        # datasets = [ 'coat']

        dataset_result = []
        for dataset in datasets:
            print (dataset)
            main(dataset)
