import glob
from tqdm import tqdm
from utils import *
import os

def obtain_filepath(dataset_name, method, regime):  # both old data and new data are of the same saving format

    directory = "../ranking"
    if method == 'EXPLORE':
        pattern =  dataset_name + '_*_*_' + regime + f'_{method}.txt'
    else:
        pattern =  dataset_name + '_*_' + regime + f'_{method}.txt'

    print(pattern)
    pattern = os.path.join(directory, pattern)
    matching_files = glob.glob(pattern)

    if matching_files:
        return matching_files[0]
    else:
        raise ValueError("No matching files found.")

def prob_check_list(ranking, item_continue_p):

    ranking_prob = item_continue_p[ranking]
    accu_continue_prob = np.cumprod(ranking_prob)
    next_reject_prob = np.ones(len(ranking_prob))
    next_reject_prob[:-1] = 1 - ranking_prob[1:]
    prob_acc_list = np.multiply(accu_continue_prob, next_reject_prob)
    return prob_acc_list

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

def dcg_for_single_user(item_continue_p, ranking, prob_acc_list):
    n = len(ranking)
    ranking_prob_list = item_continue_p[ranking]
    denom = np.log2(np.arange(2, n + 2))
    discounted_matrix = ranking_prob_list / denom
    dcg_topk = np.cumsum(discounted_matrix)
    EXP_dcg = np.sum(np.multiply(prob_acc_list, dcg_topk))
    return EXP_dcg

def Save_ExpMetric_IR(dataset_name, method_list, regime_list):



    if dataset_name == 'LETOR' or dataset_name == 'letor':
        num_user = 1691
    elif dataset_name == 'LTRC':
        num_user = 1195
    elif dataset_name == 'LTRCB':
        num_user = 5154
    else:
        raise ValueError(f'Dataset {dataset_name} not supported')

    output_path = f'../expected_metrics/metrics.txt'
    ensure_folder_exists('../expected_metrics/')

    for method in method_list:
        for regime in regime_list:
            mathching_file = obtain_filepath(dataset_name, method, regime)

            with open(mathching_file) as reader:  # ranking file reading
                print(mathching_file)

                expnum_list = []
                diversity_list = []
                dcg_list = []

                cnted_user = set()

                all_lines = reader.readlines()
                for linecnt in tqdm(range(len(all_lines))):
                    line = all_lines[linecnt]
                    qid, seq = line.split()

                    if qid in cnted_user:  # some rankings are saved for multiple times.
                        continue
                    else:
                        cnted_user.add(qid)

                    ranking = np.array(seq.split(',')).astype(int)

                    rating_fp = f'../ProcessedData/{dataset_name}/rating_{qid}.npy'
                    distance_fp = f'../ProcessedData/{dataset_name}/jaccard_genres_distances_{qid}.npy'

                    rating_arr = np.load(rating_fp)
                    distance_matrix = np.load(distance_fp)
                    distance_matrix = distance_matrix / np.max(distance_matrix)

                    item_probs = map2range(dataset_name, regime, rating_arr, regimes_mapping_vocab)

                    prob_acc_list = prob_check_list(ranking, item_probs)
                    exp_single_diversity = EXP_average_sum_distance_for_single_user(ranking, prob_acc_list, distance_matrix)
                    exp_single_expnum = EXP_num_accepted_for_single_user(prob_acc_list)
                    exp_single_dcg = dcg_for_single_user(item_probs, ranking, prob_acc_list)

                    expnum_list.append(exp_single_expnum)
                    diversity_list.append(exp_single_diversity)
                    dcg_list.append(exp_single_dcg)


            expnum_avg, expnum_std = np.average(expnum_list), np.std(expnum_list)
            diversity_avg, diversity_std = np.average(diversity_list), np.std(diversity_list)
            dcg_avg, dcg_std = np.average(dcg_list), np.std(dcg_list)



            rst = (dataset_name, method, regime, expnum_avg, diversity_avg, dcg_avg, expnum_std,
            diversity_std, dcg_std)

            with open(output_path, 'a+') as file:
                row_str = '\t'.join(map(str, rst))
                file.write(row_str + '\n')


            print(method)
            print(expnum_avg, expnum_std)
            print(dcg_avg, dcg_std)
            print(diversity_avg, diversity_std)

regimes_mapping_vocab = {"small": [0.1, 0.3], "medium": [0.4, 0.6], "large": [0.7, 0.9], 'full':[0.1,0.9]}

if __name__ == '__main__':
    #
    dataset_list = {"LETOR",'LTRCB'}
    dataset_list = {"LETOR"}

    method_list = ['EXPLORE','Random', 'DUM', 'MSD', 'MMR', 'DPP', 'B2I']

    regime_list = ['large','medium','small', 'full']
    regime_list = ['full']


    for dataset_name in dataset_list:
        Save_ExpMetric_IR(dataset_name, method_list, regime_list)

