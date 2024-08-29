from utility import *
from utils import *
import time, random
import math

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def get_recommendations(ratings, sim_matrix, distance_matrix, theta_factor,  dataset_name, regime):

    num_user, num_items = ratings.shape

    candidates_relevance = map2range(dataset_name, regime, ratings, regimes_mapping)

    similarity = sim_matrix

    each_pool_size = 64
    actual_process = int(np.ceil(num_user / each_pool_size))
    chunk_range = separate_to_chunks(np.arange(num_user), actual_process)

    recommendation_list = []
    expectation_list = []
    for i in range(len(chunk_range)):
        if i % 50 == 0:
            print(i, '-th of ', actual_process, ' chunck processing')
        chunk = chunk_range[i]
        ratings_chunk = candidates_relevance[chunk[0]:chunk[1]]

        seq_list = get_recommendations_chunk(ratings_chunk, similarity, theta_factor)
        recommendation_list.extend(seq_list)

        seq_chunk_arr = lib.to_object_array(seq_list).astype(int)

        exp_torch_arr = expectation_value_MSD_torch(seq_chunk_arr, distance_matrix, ratings_chunk).numpy()
        expectation_list.append(exp_torch_arr)

    expectation_list = np.concatenate(expectation_list)

    return np.average(expectation_list), np.std(expectation_list), recommendation_list


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
    kernel_matrix = relevance.reshape((item_size, 1)) * similarity * relevance.reshape((1, item_size))
    return kernel_matrix


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


        # the larger the parameter the longer it takes to run
        # only top few items are important for the MaxSSD objective
        if len(selected_items)> 100:
            break

        selected_items.append(selected_item)

    extendend_list =  extend_bestEdge_to_path_arbtrary(selected_items)

    return extendend_list



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
    distance_matrix = items_items_distances / np.max(items_items_distances)


    filepath = f"../ProcessedData/{dataset_name}_rating.npy"
    ratings = np.load(filepath)

    regimes = ["large", "medium", "small", 'full']
    regimes = ['full']

    strategy = 'DPP'

    for regim in regimes:

        thetas = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]
        thetas = [0.4,0.9]

        max_exp = 0
        for theta_factor in thetas:

            start = time.time()

            average_exp, std_exp, recommendation_list  = get_recommendations(ratings, sim_matrix, distance_matrix, theta_factor, dataset_name, regim)

            spentime = time.time() - start

            if average_exp > max_exp:
                max_exp = average_exp
                best_std = std_exp
                best_ranking = recommendation_list
                best_theta = theta_factor


        rst = (strategy, dataset_name, best_theta, regim, max_exp, best_std, spentime)

        result_path = f"{result_folder}/{dataset_name}_{regim}.txt"
        with open(result_path, 'a+') as file:
            row_str = '\t'.join(map(str, rst))
            file.write(row_str + '\n')

        print('Strategy' , strategy, ' average_exp, ', average_exp , 'for dataset ', dataset_name, ' regime ', regim, ' spend time ', time.time()-start)

        save_ranking_2file(ranking_folder, dataset_name, best_theta, regim, best_ranking, strategy)



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
