import time
import os
import sys, pickle
parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent)

import multiprocessing
from utility import get_structures
from utils import *


def get_single_recommendation(distance_matrix, ratings_chunk, k_param, lambda_factor):
    seq_chunk = []
    for items_continue_p in ratings_chunk:

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

        save_ranking_2file(dataset_name, round(lambda_factor,1), regime, seq_chunk_arr, 'MSD')

    expectation_list = np.concatenate(expectation_list)

    return np.average(expectation_list), np.std(expectation_list)


def main(dataset_name):

    jaccard_distances_dict = {"movielens": "genres", "KuaiRec": "users",
                              "coat": "genres", "yahoo": "genres", "netflix": "genres"}

    jaccard_distance = jaccard_distances_dict[dataset_name]

    folder = f"../outputs/jaccard_{jaccard_distance}"

    if not os.path.exists(folder):
        os.makedirs(folder)


    users_dictionary, items_dictionary, items_items_distances = get_structures(dataset_name,
                                                                               jaccard_distance=jaccard_distance,
                                                                               folder="../outputs")

    file = open(f"../ProcessedData/{dataset_name}_rating.pkl", 'rb')
    ratings = pickle.load(file)
    file.close()

    regimes_mapping = {"small": [0.1, 0.3], "medium": [0.4, 0.6], "large": [0.7, 0.9]}
    regimes = ["large", "medium", "small"]

    strategy = 'MSD'

    result_folder = 'Results'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    ranking_folder = 'ranking'  # save the permutation/ranking of items for each user, this is useful for visualization.
    if not os.path.exists(ranking_folder):
        os.makedirs(ranking_folder)


    for regime in regimes:
        mapping_range = regimes_mapping[regime]

        k_param = 2

        for lambda_factor in list(np.arange(0, 1.1, 0.1)):

            start = time.time()
            exp_avg, exp_std = get_recommendations(k_param, lambda_factor, ratings, items_items_distances, mapping_range, dataset_name, regime)

            rst = (dataset_name, regime, exp_avg, exp_std, round(lambda_factor, 1), time.time() - start)

            result_path = f"Results/{strategy}.txt"
            with open(result_path, 'a+') as file:
                row_str = '\t'.join(map(str, rst))
                file.write(row_str + '\n')

            print('MSD  expectation ', exp_avg, exp_std, ' for dataset ', dataset_name, ' regime ', regime, 'strategy', strategy,
                  ' spend time ',
                  time.time() - start)


if __name__ == '__main__':

    datasets = ["coat",  "netflix", "movielens", "KuaiRec" , "yahoo"]
    for dataset in datasets:
        print (dataset)
        main(dataset)
