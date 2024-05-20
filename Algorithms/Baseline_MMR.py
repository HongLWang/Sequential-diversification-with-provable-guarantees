
'''
所有的结果， ranking都删掉了。但是mac本地还有。
'''
import time
import os
import sys

parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

sys.path.append(parent)



import multiprocessing
from tqdm import tqdm
import numpy as np
import torch
from utility import get_structures, get_jaccard_matrix, get_items_genres_matrix
import warnings
import  pandas._libs.lib as lib
from util_expectations import separate_to_chunks, save_ranking_2file

warnings.filterwarnings('ignore')

def expectation_value_MSD(permutations, distance_matrix, items_continue_p):


    num_items  = len(permutations)
    matrix_p = items_continue_p[permutations]  # size is num of permutations * k_param

    # [p1p2, p1p2p3, p1p2p3p4, \cdots, p1...p_n]
    accumulate_p = np.cumprod(matrix_p)[1:]  # size is num of permutations * k_param-1

    # [d(1,2), d(3,{1,2}), d(4,{1,2,3}), ...d(n, {1,2,...,n-1})], the i-th element is d(i+1, {1,2,...i})
    distances = np.zeros(num_items-1)
    for i in range(1, num_items):
        new_node = permutations[ i]
        existing_nodes = permutations[:i]

        matrix_row_slices = distance_matrix[new_node][existing_nodes]

        distance_new2exist = np.sum(matrix_row_slices)
        distances[i-1] = distance_new2exist

    expectations = np.dot(accumulate_p, distances)

    return expectations # a scalar

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

def expectation_value_MHP_torch2(permutations, distance_matrix, items_continue_p):
    # Convert NumPy arrays to PyTorch tensors
    permutations_tensor = torch.tensor(permutations)
    distance_matrix_tensor = torch.tensor(distance_matrix)
    items_continue_p_tensor = torch.tensor(items_continue_p)


    accumulate_p = torch.flip(torch.cumprod(items_continue_p_tensor[permutations_tensor], dim=1)[:, 1:], dims = [1]) # size is num of permutations * k_param-1


    # wi = p1p2..p_{i+1} + ...+ p1p2..p_n
    weights = torch.flip(torch.cumsum(accumulate_p, dim=1), dims=[1])

    # size is num of permutations * k_param-1
    distances = distance_matrix_tensor[(permutations_tensor[:, :-1], permutations_tensor[:, 1:])]

    expectations = torch.sum(weights * distances, dim=1, keepdim=True)

    # Convert the result back to a NumPy array if needed
    return expectations.numpy()


def expectation_value_upgraded(permutations, distance_matrix, items_continue_p):
    matrix_p = items_continue_p[permutations]  # size is num of permutations * k_param
    accumulate_p = np.cumprod(matrix_p, axis=1)[:, 1:]  # size is num of permutations * k_param-1
    weights = np.cumsum(accumulate_p[:, ::-1], axis=1)[:, ::-1]  # wi = p1p2..p_{i+1} + ...+ p1p2..p_n

    indices = (permutations[:, :-1], permutations[:, 1:])
    distances = distance_matrix[indices]  # size is num of permutations * k_param-1

    expectations = np.sum(weights * distances, axis=1, keepdims=True)

    return expectations


def get_single_exp(num_user, num_items, distance_matrix, similarity, ratings_chunk_algo, ratings_chunk_exp, lambda_factor):

    seq_chunk = []

    for i, relevance in enumerate(ratings_chunk_algo):

        S = []
        U = set(np.arange(num_items))

        best_u = np.argmax(relevance)
        S.append(best_u)
        U -= {best_u}

        relevance[S] = np.NINF
        second_part = similarity[best_u]

        for i in list(np.arange(1, num_items, 1)):
            second_part = np.maximum(second_part, similarity[best_u])
            to_maximize = lambda_factor * relevance - (1 - lambda_factor) * second_part
            best_u = np.argmax(to_maximize)
            S.append(best_u)
            relevance[S[i]] = np.NINF

        seq_chunk.append(S)


    return seq_chunk





def get_recommendation(k_param, lambda_factor, ratings, items_items_distances, mapping_range, dataset_name, regime):

    sim_matrix = 1- items_items_distances
    distance_matrix = items_items_distances
    distance_matrix = distance_matrix / np.max(distance_matrix)
    similarity = 1- distance_matrix

    num_user, num_items = ratings.shape
    results = []

    hoped_num_process = min(48, multiprocessing.cpu_count())
    chunk_range = separate_to_chunks(np.arange(num_user), hoped_num_process)
    pool = multiprocessing.Pool()


    num_user, num_item = ratings.shape
    rating_for_algo = np.interp(ratings, (1, 5), mapping_range)  # 0.1-0.3 for small regime, 0.4-0.6 for medium and 0.7-0.9 for large
    rating_for_exp = np.interp(ratings, (1, 5), mapping_range)  # 0.1-0.3 for small regime, 0.4-0.6 for medium and 0.7-0.9 for large

    for i in range(len(chunk_range)):
        chunk = chunk_range[i]
        ratings_chunk_algo = rating_for_algo[chunk[0]:chunk[1]]
        ratings_chunk_exp = rating_for_exp[chunk[0]:chunk[1]]

        result = pool.apply_async(get_single_exp,
                                  args=(num_user, num_item, distance_matrix, similarity, ratings_chunk_algo, ratings_chunk_exp, lambda_factor))

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
    for i, seq_list in enumerate (result_list):

        chunk = chunk_range[i]
        ratings_chunk_exp = rating_for_exp[chunk[0]:chunk[1]]


        seq_chunk_arr = lib.to_object_array(seq_list).astype(int)
        exp_torch_arr = expectation_value_MSD_torch(seq_chunk_arr, distance_matrix, ratings_chunk_exp)
        expectation_list.append(exp_torch_arr)

        save_ranking_2file(dataset_name, lambda_factor, regime, seq_chunk_arr, 'MMR')

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
    # ratings = ratings[:500]
    # ratings_small = ratings[:1000]


    strategies = [
        "CIKM_MMR"
    ]


    regimes_mapping = {"small": [0.1, 0.3], "medium": [0.4, 0.6], "large": [0.7, 0.9]}
    regimes = ["small", "medium", "large"]

    result = []
    for regime in regimes:
        mapping_range = regimes_mapping[regime]

        for strategy in strategies:
            k_param = 2

            # for lambda_factor in list(np.arange(0, 1.1, 0.1)):
            for lambda_factor in [known_best_param[dataset_name][regime]]:
                start = time.time()
                exp_avg, exp_std = get_recommendation(k_param,lambda_factor, ratings, items_items_distances, mapping_range, dataset_name, regime)

                rst = (dataset_name, regime, exp_avg, exp_std, lambda_factor, time.time() - start)

                result_path = f"../results_new/{strategy}.txt"
                with open(result_path, 'a+') as file:
                    row_str = '\t'.join(map(str, rst))
                    file.write(row_str + '\n')

                print('MMR  expectation ', exp_avg, exp_std, ' for dataset ', dataset_name, ' regime ', regime,
                      ' spend time ',
                      time.time() - start)


if __name__ == '__main__':

    strategy = "mmr"
    datasets = {"movielens", "KuaiRec", "coat", "yahoo", "netflix"}


    known_best_param = {}
    for data_name in datasets:
        known_best_param[data_name] = {}

    known_best_param['coat']['small'] = 0.9
    known_best_param['coat']['medium'] = 0.9
    known_best_param['coat']['large'] = 0.9

    known_best_param['netflix']['small'] = 0.9
    known_best_param['netflix']['medium'] = 0.9
    known_best_param['netflix']['large'] = 1

    known_best_param['movielens']['small'] = 0.5
    known_best_param['movielens']['medium'] = 0.9
    known_best_param['movielens']['large'] = 0.9

    known_best_param['yahoo']['small'] = 0.5
    known_best_param['yahoo']['medium'] = 0.9
    known_best_param['yahoo']['large'] = 0.9

    known_best_param['KuaiRec']['small'] = 0.8
    known_best_param['KuaiRec']['medium'] = 0.7
    known_best_param['KuaiRec']['large'] = 0.8


    # datasets = {"coat"}


    for dataset in datasets:
        print (dataset)
        main(dataset)

