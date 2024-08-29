import time
from tqdm import tqdm
from utils import *
import copy



def search_best_k_nodes(item_continue_prob, distance_matrix):

    num_items = distance_matrix.shape[0]
    all_pair = generate_combinations_and_permutations(np.arange(num_items), 2)
    all_pair_indices = (all_pair[:,0], all_pair[:, 1])
    all_pair_distance = distance_matrix[all_pair_indices]

    all_pair_prob = np.cumprod(item_continue_prob[all_pair], axis=1)[:,-1]
    all_pair_gain = np.multiply(all_pair_distance, all_pair_prob)

    best_ranking = []

    best_pair = all_pair[np.argmax(all_pair_gain)]
    best_ranking.extend(list(best_pair))

    item_prob = copy.deepcopy(item_continue_prob)
    item_prob[best_pair[0]] = 0
    item_prob[best_pair[1]] = 0

    while len(best_ranking) < num_items:

        distance_gain_vec = np.sum(distance_matrix[:, best_pair], axis = 1)
        exp_gain = np.multiply(item_prob, distance_gain_vec)
        best_item = np.argmax(exp_gain)
        item_prob[best_item] = 0
        distance_gain_vec += distance_matrix[:, best_item]
        best_ranking.append(best_item)


    return best_ranking



def get_recommendation(dataset_name, qid, regime, k_param, ratings, items_items_distances, mapping_range):

    distance_matrix = items_items_distances
    distance_matrix = distance_matrix / np.max(distance_matrix)


    items_continue_p =  map2range(dataset_name,regime,ratings,regimes_mapping)

    best_sequence = search_best_k_nodes(items_continue_p, distance_matrix)
    expectation = expectation_value_MSD(np.array(best_sequence), distance_matrix, items_continue_p)

    return expectation, best_sequence



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

        # for lambda_factor in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
        for k_param in [2]:

            mapping_range = regimes_mapping[regime]
            start = time.time()
            expectation_list = []
            ranking_list = []


            folder_path = f'../ProcessedData/{dataset_name}/'
            for qid in tqdm(range(num_user)):
                items_items_distances = np.load(folder_path + f'jaccard_genres_distances_{qid}.npy')
                rating = np.load(folder_path + f'rating_{qid}.npy')
                rating = rating.astype(int)

                expectation, ranking = get_recommendation(dataset_name, qid, regime, k_param, rating,
                                                 items_items_distances, mapping_range)

                expectation_list.append(expectation)
                ranking_list.append((qid, ranking))

            exp_avg, exp_std = np.average(expectation_list), np.std(expectation_list)

            rst = (strategy, dataset_name, k_param, regime, exp_avg, exp_std, time.time() - start)
            if exp_avg > best_res:
                best_tuple = rst
                best_res = exp_avg
                best_ranking = ranking_list
                best_theta_factor = k_param



            print('DUM  expectation ', exp_avg, exp_std, ' for dataset ', dataset_name, ' regime ', regime,
                   ' spend time ',
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


    strategy = 'DUM'

    # for new dataset, run all mapping. first finish collecting results for new datasets.
    datasets = ['LETOR','LTRC','LTRCB']
    regimes = ["small", "medium", "large", "full"]

    # #
    datasets = ['LETOR']
    regimes = ["full"]

    dataset_result = []
    for dataset in datasets:
        print (dataset)
        main(dataset, regimes)

