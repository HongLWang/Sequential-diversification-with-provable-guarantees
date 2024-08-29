import time
import random
import multiprocessing
from tqdm import tqdm
from utils import *



def search_best_k_nodes(all_k_permutation, items_continue_p, distance_matrix, num_items):

    def extend_bestEdge_to_path_arbtrary(best_nodes):
        num_items = distance_matrix.shape[0]
        chosen_nodes = list(best_nodes)
        all_items = np.arange(num_items)
        rest_nodes = list(set(all_items) - set(chosen_nodes))
        random.shuffle(rest_nodes)
        chosen_nodes.extend(rest_nodes)

        return chosen_nodes

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

    # sequence = extend_bestEdge_to_path_greedy(best_permu)
    sequence = extend_bestEdge_to_path_greedy(best_permu)

    return sequence


def get_recommendation(dataset_name, regime, k_param, ratings, items_items_distances, mapping_range):

    distance_matrix = items_items_distances
    distance_matrix = distance_matrix / np.max(distance_matrix)

    num_items = len(ratings)

    items_continue_p = map2range(dataset_name, regime, ratings, regimes_mapping)
    all_k_permutation = generate_combinations_and_permutations(np.arange(num_items), k_param)
    best_sequence = search_best_k_nodes(all_k_permutation, items_continue_p, distance_matrix, num_items)
    expectation = expectation_value_MSD(np.array(best_sequence), distance_matrix, items_continue_p)

    return expectation,best_sequence


def get_recommendation_chunk(dataset_name, qid_chunk, regime, k_param, mapping_range):

    expectation_chunk = []
    ranking_chunk = []
    folder_path = f'../ProcessedData/{dataset_name}/'
    for qid in tqdm(range(qid_chunk[0],qid_chunk[1])):
        items_items_distances = np.load(folder_path+ f'jaccard_genres_distances_{qid}.npy')
        rating = np.load(folder_path + f'rating_{qid}.npy')
        rating = rating.astype(int)

        expectation, ranking = get_recommendation(dataset_name, regime, k_param, rating, items_items_distances,
                                         mapping_range)

        expectation_chunk.append(expectation)
        ranking_chunk.append((qid,ranking))

    return [expectation_chunk, ranking_chunk]




def main(dataset_name, regimes, k_param_list):

    print(f"Dataset: {dataset_name}")

    for k_param in k_param_list:
        strategy = f'B{k_param}I'
        for regime in regimes:

            mapping_range = regimes_mapping[regime]
            start = time.time()

            if dataset_name == 'LETOR' or dataset_name == 'letor':
                num_user = 1691
            elif dataset_name == 'LTRC':
                num_user = 1195
            elif dataset_name == 'LTRCB':
                num_user = 5154
            else:
                raise ValueError('please provide a valid dataset')


            hoped_num_process = min(4, multiprocessing.cpu_count())
            chunk_range = separate_to_chunks(np.arange(num_user), hoped_num_process)
            pool = multiprocessing.Pool()

            results = []
            for i in range(len(chunk_range)):
                qid_chunk = chunk_range[i]
                print(qid_chunk)

                result = pool.apply_async(get_recommendation_chunk,
                                          args=(dataset_name, qid_chunk, regime, k_param, mapping_range))

                results.append(result)

            pool.close()
            pool.join()

            exp_list = []
            rank_list = []
            for result in results:
                [exp,ranking] = result.get()
                exp_list.extend(exp)
                rank_list.extend(ranking)

            pool.terminate()

            exp_avg, exp_std = np.average(exp_list), np.std(exp_list)
            save_qid_ranking_2file(ranking_folder, dataset_name, k_param, regime, rank_list, strategy)  # this should be after time was calculated

            rst = (strategy, dataset_name, k_param, regime, exp_avg, exp_std, time.time() - start)

            result_path = f"{result_folder}/{dataset_name}_{regime}.txt"
            with open(result_path, 'a+') as file:
                row_str = '\t'.join(map(str, rst))
                file.write(row_str + '\n')

            print('bke exact  expectation ', exp_avg, exp_std, ' for dataset ', dataset_name, ' regime ', regime,
                  'k = ', k_param , ' spend time ',
                  time.time() - start)



regimes_mapping = {"small": [0.1, 0.3], "medium": [0.4, 0.6], "large": [0.7, 0.9], 'full':[0,1]}
if __name__ == '__main__':

    result_folder = '../results'
    ensure_folder_exists(result_folder)

    ranking_folder = 'ranking'  # save the permutation/ranking of items for each user, this is useful for visualization.



    # for new dataset, run all mapping. first finish collecting results for new datasets.
    datasets = ['LETOR','LTRCB']
    datasets = ['LETOR']

    regimes = ["full","small", "medium", "large"]
    regimes = ["full"]

    k_param_list = [2,3,4]
    k_param_list = [2]


    dataset_result = []

    for dataset in datasets:
        print (dataset)
        main(dataset, regimes, k_param_list)
