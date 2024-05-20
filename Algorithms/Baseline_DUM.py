import random
import time
import os
import sys

parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

sys.path.append(parent)


import multiprocessing
from tqdm import tqdm
import numpy as np
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








def get_recommendations_dum_chunk(ratings_chunck, items_genres_matrix,mapping_range):

    num_items = ratings_chunck.shape[1]

    all_item = np.arange(num_items)

    seq_chunk = []
    for i, user_weights in enumerate(ratings_chunck):

        final_recommendation_list = []
        user_weights = ratings_chunck[i]

        ordered_by_weights = all_item[np.argsort(user_weights)[::-1]]

        final_recommendation_list.append(ordered_by_weights[0])

        for c in ordered_by_weights[1:]:

            coverage_vector_without_c = items_genres_matrix[final_recommendation_list].sum(axis=0)
            coverage_without_c = (coverage_vector_without_c > 0).astype(int).sum()

            coverage_vector_with_c = items_genres_matrix[(final_recommendation_list + [c])].sum(axis=0)
            coverage_with_c = (coverage_vector_with_c > 0).astype(int).sum()

            if coverage_with_c > coverage_without_c:
                final_recommendation_list.append(c)

            if len(final_recommendation_list) == num_items:
                break


        unselected_items = list(set(all_item) - set(final_recommendation_list))
        random.shuffle(unselected_items)
        final_recommendation_list.extend(unselected_items)
        seq_chunk.append(final_recommendation_list)

    return seq_chunk



def get_recommendation(k_param, ratings, items_items_distances, items_genres_matrix,mapping_range, dataset_name, regime):

    distance_matrix = items_items_distances
    distance_matrix = distance_matrix / np.max(distance_matrix)

    num_user, num_items = ratings.shape

    ratings = np.interp(ratings, (1, 5), mapping_range) # 0.1-0.3 for small regime, 0.4-0.6 for medium and 0.7-0.9 for large

    hoped_num_process = min(48, multiprocessing.cpu_count())
    chunk_range = separate_to_chunks(np.arange(num_user), hoped_num_process)
    pool = multiprocessing.Pool()

    results = []
    for i in range(len(chunk_range)):
        chunk = chunk_range[i]
        ratings_chunk = ratings[chunk[0]:chunk[1]]
        result = pool.apply_async(get_recommendations_dum_chunk,
                                  args=(ratings_chunk, items_genres_matrix,mapping_range))

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

        # print (len(seq_list))
        seq_chunk_arr = lib.to_object_array(seq_list).astype(int)
        exp_torch_arr = expectation_value_MSD_torch(seq_chunk_arr, distance_matrix, ratings_chunk_exp)
        expectation_list.append(exp_torch_arr)


        save_ranking_2file(dataset_name, k_param, regime, seq_chunk_arr, 'DUM')

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

    items_genres_path = f"../outputs/items_genres_matrix_{dataset_name}.npy"
    items_genres_matrix = get_items_genres_matrix(items_dictionary, dataset_name, None, items_genres_path)

    users, items = list(users_dictionary.values()), list(items_dictionary.values())
    n_users, n_items = len(users), len(items)

    ratings = np.load(f"../OMSD/rating_{dataset_name}.npy")



    regimes_mapping = {"small": [0.1, 0.3], "medium": [0.4, 0.6], "large": [0.7, 0.9]}
    regimes = ["small", "medium", "large"]

    strategy = 'CIKM_DUM'


    for regime in regimes:
        mapping_range = regimes_mapping[regime]
        k_param = 2
        start = time.time()
        exp_avg, exp_std = get_recommendation(k_param, ratings, items_items_distances, items_genres_matrix, mapping_range,dataset_name, regime)

        rst = (dataset_name, regime, exp_avg, exp_std, time.time() - start)

        result_path = f"../results_new/{strategy}.txt"
        with open(result_path, 'a+') as file:
            row_str = '\t'.join(map(str, rst))
            file.write(row_str + '\n')

        print('DUM  expectation ', exp_avg, exp_std, ' for dataset ', dataset_name, ' regime ', regime, ' spend time ',
              time.time() - start)



    # return results



if __name__ == '__main__':

    for strategy in ['dum']:
        datasets = {"movielens", "KuaiRec", "coat", "yahoo", "netflix"}
        # datasets = {"coat"}
        # datasets = ["coat"]

        dataset_result = []
        for dataset in datasets:
            print (dataset)
            main(dataset)
