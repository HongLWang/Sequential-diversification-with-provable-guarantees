import glob
from tqdm import tqdm
import multiprocessing
from utils import *

def obtain_filepath(directory, dataset_name, method, regime):  # both old data and new data are of the same saving format

    pattern =  directory + '/'+ dataset_name + '_*_' + regime + f'_{method}.txt'
    print(pattern)
    pattern = os.path.join(directory, pattern)
    matching_files = glob.glob(pattern)

    if matching_files:
        return matching_files[0]
    else:
        raise ValueError("No matching files found.")

def prob_check_list(ranking, uid, rating_interploted_mapped):

    num_user = rating_interploted_mapped.shape[0]

    item_continue_p = rating_interploted_mapped[uid]
    ranking_prob = item_continue_p[ranking]
    accu_continue_prob = np.cumprod(ranking_prob)
    next_reject_prob = np.ones(len(ranking_prob))
    next_reject_prob[:-1] = 1 - ranking_prob[1:]
    prob_acc_list = np.multiply(accu_continue_prob, next_reject_prob)
    return prob_acc_list

def EXP_serendipity_for_single_user(uid, ranking, prob_acc_list, item_genre_matrix, covered_genre_matrix, rating_interploted_mapped):
    genre_history = covered_genre_matrix[uid]
    serendipity = 0
    serendipity_list = []
    for item in ranking:
        this_genre = item_genre_matrix[int(item)].astype(int)
        genre_merged = np.bitwise_or(genre_history, this_genre)
        increment = np.sum(genre_merged - genre_history)
        if increment > 0:
            serendipity += rating_interploted_mapped[uid][int(item)]
            serendipity_list.append(serendipity)
        else:
            serendipity_list.append(serendipity)
    expectation = np.sum(np.multiply(prob_acc_list, serendipity_list))
    return expectation

def EXP_num_accepted_for_single_user(prob_acc_list):
    acc_num_arr = np.arange(len(prob_acc_list)) + 1
    acc_num_exp = np.sum(np.multiply(prob_acc_list, acc_num_arr))
    return acc_num_exp

def EXP_average_sum_distance_for_single_user(ranking, prob_acc_list, distance_matrix):
    num_items = len(ranking)
    distances = np.zeros(num_items)
    for i in range(1, num_items):
        new_node = ranking[i]
        existing_nodes = ranking[:i]

        matrix_row_slices = distance_matrix[new_node][existing_nodes]

        distance_new2exist = np.sum(matrix_row_slices)
        distances[i] = distance_new2exist

    top_k_sum_diversity_arr = np.cumsum(distances)

    # Multiply with the accumulated probability list and sum the result
    exp_sum_diversity = np.dot(prob_acc_list, top_k_sum_diversity_arr)

    return exp_sum_diversity

def dcg_for_single_user(uid, ranking, prob_acc_list, rating_interploted_mapped):
    n = len(ranking)
    item_continue_p = rating_interploted_mapped[uid]
    ranking_prob_list = item_continue_p[ranking]
    denom = np.log2(np.arange(2, n + 2))
    discounted_matrix = ranking_prob_list / denom
    dcg_topk = np.cumsum(discounted_matrix)
    EXP_dcg = np.sum(np.multiply(prob_acc_list, dcg_topk))
    return EXP_dcg

def process_user(uid_chunk, all_ranking, rating_interploted_mapped, distance_matrix, item_genre_matrix, covered_genre_matrix):

    serendipity_list = []
    expnum_list = []
    diversity_list = []
    dcg_list = []
    num_users = rating_interploted_mapped.shape[0]
    for uid in tqdm(uid_chunk):

        if uid <= num_users-1:

            ranking = all_ranking[uid]
            seq = np.array(ranking.split(',')).astype(int)
            prob_acc_list = prob_check_list(seq, uid, rating_interploted_mapped)
            exp_single_serendipity = EXP_serendipity_for_single_user(uid, seq, prob_acc_list, item_genre_matrix, covered_genre_matrix, rating_interploted_mapped)
            exp_single_diversity = EXP_average_sum_distance_for_single_user(seq, prob_acc_list, distance_matrix)
            exp_single_expnum = EXP_num_accepted_for_single_user(prob_acc_list)
            exp_single_dcg = dcg_for_single_user(uid, seq, prob_acc_list, rating_interploted_mapped)


            serendipity_list.append(exp_single_serendipity)
            expnum_list.append(exp_single_expnum)
            diversity_list.append(exp_single_diversity)
            dcg_list.append(exp_single_dcg)

        elif uid >= num_users:
            print('last user processed')
            continue

    return serendipity_list, expnum_list, diversity_list, dcg_list

def Save_ExpMetric(dataset_name, method_list, regime_list):

    rating_interploted_fp = f'../ProcessedData/{dataset_name}_rating.npy'
    rating_interploted = np.load(rating_interploted_fp)
    item_genre_fp = f'../outputs/items_genres_matrix_{dataset_name}.npy'
    item_genre_matrix = np.load(item_genre_fp)
    covered_genre_fp = f'../outputs/covered_genre/{dataset_name}.npy'
    covered_genre_matrix = np.load(covered_genre_fp)

    if not dataset_name == 'KuaiRec':
        jaccard_distances_fp = f"../outputs/jaccard_genres_distances_{dataset_name}.npy"
    else:
        jaccard_distances_fp = f"../outputs/jaccard_users_distances_{dataset_name}.npy"
    distance_matrix = np.load(jaccard_distances_fp)
    distance_matrix = distance_matrix / np.max(distance_matrix)


    ensure_folder_exists('../expected_metrics/')

    for method in method_list:
        for regime in regime_list:
            rating_interploted_mapped = map2range(dataset_name, regime, rating_interploted, regimes_mapping_vocab)
            print(dataset_name, method, regime)
            ranking_path = obtain_filepath(ranking_folder, dataset_name, method, regime)
            if ranking_path is None:
                print(f'no ranking file found for {dataset_name} + {method} + {regime}')
                continue
            else:
                print(ranking_path)
            with open(ranking_path, 'r') as f:
                all_ranking = f.readlines()

                hoped_num_process = min(32, multiprocessing.cpu_count())
                chunk_range = separate_to_chunks(np.arange(len(all_ranking)), hoped_num_process)
                pool = multiprocessing.Pool()

                results = []
                for i in range(len(chunk_range)):
                    chunk = chunk_range[i]
                    uid_chunk = np.arange(len(all_ranking))[chunk[0]:chunk[1]]
                    # print(chunk)
                    #
                    result = pool.apply_async(process_user,
                                              args=(uid_chunk, all_ranking, rating_interploted_mapped, distance_matrix, item_genre_matrix, covered_genre_matrix))

                    results.append(result)

                pool.close()
                pool.join()

                serendipity_list, expnum_list, diversity_list, dcg_list = [],[],[],[]
                for result in results:
                    serendipity_list_chunk, expnum_list_chunk, diversity_list_chunk, dcg_list_chunk = result.get()
                    serendipity_list.extend(serendipity_list_chunk)
                    expnum_list.extend(expnum_list_chunk)
                    diversity_list.extend(diversity_list_chunk)
                    dcg_list.extend(dcg_list_chunk)

            seren_avg, seren_std = np.average(serendipity_list), np.std(serendipity_list)
            expnum_avg, expnum_std = np.average(expnum_list), np.std(expnum_list)
            diversity_avg, diversity_std = np.average(diversity_list), np.std(diversity_list)
            dcg_avg, dcg_std = np.average(dcg_list), np.std(dcg_list)

            rst = (dataset_name, method, regime, seren_avg, expnum_avg, diversity_avg, dcg_avg, seren_std, expnum_std, diversity_std, dcg_std)

            with open(output_path, 'a+') as file:
                row_str = '\t'.join(map(str, rst))
                file.write(row_str + '\n')

            print(method)
            print(expnum_avg, expnum_std)
            print(dcg_avg, dcg_std)
            print(diversity_avg, diversity_std)
            print(seren_avg, seren_std)


def get_genre_history_dictionary(dataset_list):

    def get_genre_history(uid): # save the genres the users has covered
        item_vec = rating_history_matrix[uid]
        selected_rows = item_genre_matrix[item_vec > 0].astype(int)
        uid_covered_genres = np.bitwise_or.reduce(selected_rows, axis=0)
        return uid_covered_genres


    for dataset_name in dataset_list:

        rating_history_fp = '../outputs/items_users_matrix_' + dataset_name + '.npy'
        item_user_matrix = np.load(rating_history_fp)
        rating_history_matrix = item_user_matrix.T
        [num_usr,num_item] = rating_history_matrix.shape

        item_genre_fp = '../outputs/items_genres_matrix_' + dataset_name + '.npy'
        item_genre_matrix = np.load(item_genre_fp)

        num_genre = item_genre_matrix.shape[1]
        covered_genre_matrix = np.zeros((num_usr, num_genre)).astype(int)

        covered_genre_fp = f'../outputs/covered_genre/{dataset_name}.npy'
        ensure_folder_exists('../outputs/covered_genre/')

        for uid in range(num_usr):
            covered_genre = get_genre_history(uid)
            covered_genre_matrix[uid] = covered_genre

        np.save(covered_genre_fp, covered_genre_matrix)


regimes_mapping_vocab = {"small": [0.1, 0.3], "medium": [0.4, 0.6], "large": [0.7, 0.9], 'full': [0.1, 0.9]}

if __name__ == '__main__':

    ranking_folder = '../ranking'  # save the permutation/ranking of items for each user, this is useful for visualization.
    output_path = '../expected_metrics/metrics.txt'

    dataset_list = {"movielens", "KuaiRec", "coat", "yahoo", "netflix"}
    dataset_list = {"coat"}

    get_genre_history_dictionary(dataset_list)

    method_list = ['EXPLORE','Random', 'DUM', 'MSD', 'MMR', 'DPP', 'B2I', 'B3I-H']
    method_list = ['EXPLORE']

    regime_list = ['large', 'medium', 'small', 'full']
    regime_list = ['full']


    for dataset_name in dataset_list:
        Save_ExpMetric(dataset_name, method_list, regime_list)
