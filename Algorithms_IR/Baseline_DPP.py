import time
import random
import math, pickle
from tqdm import tqdm
from utils import *



def get_recommendation(dataset_name, qid, theta_factor, regime, relevance, items_items_distances, mapping_range):

    distance_matrix = items_items_distances
    distance_matrix = distance_matrix / np.max(distance_matrix)
    similarity_matrix = 1 - distance_matrix


    items_continue_p = map2range(dataset_name, regime, relevance,regimes_mapping)
    num_items = len(relevance)

    L = get_kernel_matrix(items_continue_p, similarity_matrix, theta_factor)

    best_ranking = dpp(L, num_items)
    expectation = expectation_value_MSD(np.array(best_ranking), distance_matrix, items_continue_p)

    return expectation,best_ranking


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


def main(dataset_name, regimes):

    print(f"Dataset: {dataset_name}")

    if dataset_name == 'LETOR' or dataset_name == 'letor':
        num_user = 1691
    elif dataset_name == 'LTRC':
        num_user = 1195
    elif dataset_name == 'LTRCB':
        num_user = 5154
    else:
        raise ValueError(f'Dataset {dataset_name} not supported')


    for regime in regimes:

        best_res = 0
        best_tuple = ()
        best_ranking = []
        best_theta_factor = 0

        # for theta_factor in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
        for theta_factor in [0.3, 0.1 ]:

            mapping_range = regimes_mapping[regime]
            start = time.time()
            expectation_list = []
            ranking_list = []


            folder_path = f'../ProcessedData/{dataset_name}/'
            for qid in tqdm(range(num_user)):
                items_items_distances = np.load(folder_path + f'jaccard_genres_distances_{qid}.npy')
                rating = np.load( folder_path + f'rating_{qid}.npy')
                rating = rating.astype(int)

                expectation, ranking = get_recommendation(dataset_name, qid, theta_factor, regime, rating, items_items_distances,
                                                 mapping_range)
                expectation_list.append(expectation)
                ranking_list.append((qid, ranking))

            exp_avg, exp_std = np.average(expectation_list), np.std(expectation_list)

            rst = (strategy, dataset_name, theta_factor, regime, exp_avg, exp_std, time.time() - start)
            if exp_avg > best_res:
                best_tuple = rst
                best_res = exp_avg
                best_ranking = ranking_list
                best_theta_factor = theta_factor

            # param_folder_path = "../result_KDD25/parameter_tunning/"
            # ensure_folder_exists(param_folder_path)
            # param_result_path = param_folder_path + f"{strategy}.txt"
            # with open(param_result_path, 'a+') as file:
            #     row_str = '\t'.join(map(str, rst))
            #     file.write(row_str + '\n')


            print('DPP  expectation ', exp_avg, exp_std, ' for dataset ', dataset_name, ' regime ', regime,
                  'theta_factor = ', theta_factor , ' spend time ',
                  time.time() - start)

        result_path = f"{result_folder}/{dataset_name}_{regime}.txt"

        with open(result_path, 'a+') as file:
            row_str = '\t'.join(map(str, best_tuple))
            file.write(row_str + '\n')

        save_qid_ranking_2file(ranking_folder, dataset_name, best_theta_factor, regime, best_ranking, strategy)


# ranking 不管parameter是多少都存下来，后面手动选parameter最好的单独放到一个文件夹里面
regimes_mapping = {"small": [0.1, 0.3], "medium": [0.4, 0.6], "large": [0.7, 0.9], 'full':[0,1]}

if __name__ == '__main__':


    result_folder = '../results'
    ensure_folder_exists(result_folder)

    ranking_folder = 'ranking'  # save the permutation/ranking of items for each user, this is useful for visualization.



    strategy = 'DPP'

    # for new dataset, run all mapping. first finish collecting results for new datasets.
    datasets = ['LETOR','LTRCB']
    datasets = ['LETOR']
    regimes = ["small", "medium", "large", "full"]
    regimes = [ "full"]


    dataset_result = []
    for dataset in datasets:
        print (dataset)
        main(dataset, regimes)

