import random
import time
import os
import sys
import pickle
parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent)

import multiprocessing
from utility import get_structures, get_items_genres_matrix
from utils import *



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

    ratings = np.interp(ratings, (1, 5), mapping_range)

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

    file = open(f"../ProcessedData/{dataset_name}_rating.pkl", 'rb')
    ratings = pickle.load(file)
    file.close()

    regimes_mapping = {"small": [0.1, 0.3], "medium": [0.4, 0.6], "large": [0.7, 0.9]}
    regimes = ["large", "medium", "small"]

    strategy = 'DUM'

    result_folder = 'Results'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    ranking_folder = 'ranking'  # save the permutation/ranking of items for each user, this is useful for visualization.
    if not os.path.exists(ranking_folder):
        os.makedirs(ranking_folder)


    for regime in regimes:
        mapping_range = regimes_mapping[regime]
        k_param = 2
        start = time.time()
        exp_avg, exp_std = get_recommendation(k_param, ratings, items_items_distances, items_genres_matrix, mapping_range,dataset_name, regime)

        rst = (dataset_name, regime, exp_avg, exp_std, time.time() - start)

        result_path = f"Results/{strategy}.txt"
        with open(result_path, 'a+') as file:
            row_str = '\t'.join(map(str, rst))
            file.write(row_str + '\n')

        print('DUM  expectation ', exp_avg, exp_std, ' for dataset ', dataset_name, ' regime ', regime, ' spend time ',
              time.time() - start)



if __name__ == '__main__':

    datasets = ["coat",  "netflix", "movielens", "KuaiRec" , "yahoo"]

    dataset_result = []
    for dataset in datasets:
        print (dataset)
        main(dataset)
