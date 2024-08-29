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


def get_recommendations(k_param, lambda_factor, ratings, items_items_distances, dataset_name , regime):
    num_user, num_items = ratings.shape

    results = []
    distance_matrix = items_items_distances
    distance_matrix = distance_matrix / np.max(distance_matrix)

    hoped_num_process = min(48, multiprocessing.cpu_count())
    chunk_range = separate_to_chunks(np.arange(num_user), hoped_num_process)
    pool = multiprocessing.Pool()

    ratings = map2range(dataset_name, regime, ratings, regimes_mapping)



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
    recommendation_list = []
    for i, seq_list in enumerate(result_list):
        recommendation_list.extend(seq_list)
        chunk = chunk_range[i]
        ratings_chunk_exp = ratings[chunk[0]:chunk[1]]

        # print(len(seq_list))
        seq_chunk_arr = lib.to_object_array(seq_list).astype(int)
        exp_torch_arr = expectation_value_MSD_torch(seq_chunk_arr, distance_matrix, ratings_chunk_exp)
        expectation_list.append(exp_torch_arr)


    expectation_list = np.concatenate(expectation_list)

    return np.average(expectation_list), np.std(expectation_list),recommendation_list


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

    filepath = f"../ProcessedData/{dataset_name}_rating.npy"
    ratings = np.load(filepath)

    regimes_mapping = {"small": [0.1, 0.3], "medium": [0.4, 0.6], "large": [0.7, 0.9]}
    regimes = ["large", "medium", "small",'full']
    regimes = ['full']

    strategy = 'MSD'


    for regime in regimes:

        k_param = 2

        max_exp = 0
        for lambda_factor in list(np.arange(0, 1.1, 0.1)):

            start = time.time()
            exp_avg, exp_std, recommendation_list = get_recommendations(k_param, lambda_factor, ratings, items_items_distances, dataset_name, regime)
            print('MSD  expectation ', exp_avg, exp_std, ' for dataset ', dataset_name, ' regime ', regime, 'strategy',
                  strategy, ' lambda=', lambda_factor,
                  ' spend time ',
                  time.time() - start)

            spentime = time.time() - start

            if exp_avg > max_exp:
                max_exp = exp_avg
                best_std = exp_std
                best_ranking = recommendation_list
                best_lambda = lambda_factor

        rst = (strategy, dataset_name, round(best_lambda, 2), regime, max_exp, best_std, spentime)

        result_path = f"{result_folder}/{dataset_name}_{regime}.txt"
        with open(result_path, 'a+') as file:
            row_str = '\t'.join(map(str, rst))
            file.write(row_str + '\n')

        print('MSD  expectation ', max_exp, best_std, ' for dataset ', dataset_name, ' regime ', regime, 'strategy', strategy,
              ' spend time ',
              time.time() - start)

        save_ranking_2file(ranking_folder, dataset_name, best_lambda, regime, best_ranking, strategy)


regimes_mapping = {"small": [0.1, 0.3], "medium": [0.4, 0.6], "large": [0.7, 0.9], 'full':[0.1,0.9]}

if __name__ == '__main__':

    result_folder = '../results'
    ensure_folder_exists(result_folder)


    ranking_folder = 'ranking'  # save the permutation/ranking of items for each user, this is useful for visualization.


    datasets = ["coat", "netflix","movielens", "KuaiRec", "yahoo"]
    datasets = ["coat"]

    dataset_result = []
    for dataset in datasets:
        print(dataset)
        results = main(dataset)