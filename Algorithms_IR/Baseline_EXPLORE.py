import random
from joblib import Parallel, delayed
import time
import sys

from Strategy.utils import  *
from Strategy.strategies import *


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from utility import get_structures, get_items_genres_matrix


def save_ranking_2file_budget(dataset_name, k_param, regime, budget, ranking, strategy):
    folderpath = '../ranking/'
    ensure_folder_exists(folderpath)
    filename = folderpath + dataset_name + '_'+ str(k_param) + '_'+ str(budget) + '_'+ regime + '_'+ strategy + '.txt'
    with open(filename, 'a+') as writer:
        for row in ranking:
            row_string = ','.join(map(str, row))
            writer.write(row_string + '\n')


def save_qid_ranking_2file(dataset_name, k_param, budget,regime, ranking_list, strategy):
    filename = '../ranking/' + dataset_name + '_'+ str(k_param) + '_'+ str(budget) + '_'+ regime + f'_{strategy}.txt'
    ensure_folder_exists('../ranking')
    with open(filename, 'a+') as writer:
        for (qid, ranking) in ranking_list:
            row_string = ','.join(map(str, ranking))
            writer.write(str(qid) + ' ' +  row_string + '\n')



def main_recommendation_datasets(dataset_name, regime, k_length, user_budget, num_trials):

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

    items_items_distances = items_items_distances/np.max(items_items_distances)

    items_genres_path = f"../outputs/items_genres_matrix_{dataset_name}.npy"
    items_genres_matrix = get_items_genres_matrix(items_dictionary, dataset_name, None, items_genres_path)

    users, items = list(users_dictionary.values()), list(items_dictionary.values())
    n_users, n_items = len(users), len(items)

    if items_genres_matrix is not None:
        n_genres = items_genres_matrix.shape[1]

    device_id = 1

    if torch.cuda.is_available():
        device = f"cuda:{device_id}"
        torch.cuda.set_device(device_id)
    else:
        device = "cpu"

    # print(f"Device: {device}")


    users = np.array(users)

    clip_min, clip_max = 1, 5



    strategy = "k_means_with_relevance"
    strategies = ["k_means_with_relevance"]

    copula = "clayton"

    USE_WEIBULL = True

    COMPUTE_EXPECTATION = False

    EVALUATE = False

    alpha = 0.5
    g = 2

    lambdas_dict = {5: 6.2, 10: 11.85, 20: 23.21, 30: 30 / 0.886227, 40: 40 / 0.886227, 50: 50 / 0.886227}

    if copula == "clayton":
        default_alphas = [0.0001, 0.001, 0.01, 0.5, 0.1, 1, 2, 3, 4]  # current experiments are with alpha=0.5


    whole_rating_predicted = np.load(f'../ProcessedData/{dataset_name}_rating.npy')
    whole_rating_predicted = map2range(dataset_name, regime, whole_rating_predicted, regimes_mapping_vocab)
    models_dict = instantiate_models(strategies, n_users, n_items, items_items_distances, items_genres_matrix, whole_rating_predicted,
                                     device, clip_min, clip_max, EVALUATE, dataset_name, jaccard_distance)

    def process():

        users_steps = np.zeros(n_users)


        print("Starting trial...")
        start_trial = time.time()

        users_items_matrix = np.zeros((n_users, n_items))
        users_quitting_vector = np.zeros(len(users))
        users_quitting_vector2 = np.zeros(len(users))

        recommendation_list = [[] for i in range(n_users)]

        start = time.time()

        while 0 in users_quitting_vector:

            active_users1 = users[users_quitting_vector[users] == 0]
            active_users2 = users[users_quitting_vector2[users] == 0]
            active_users = list(set.intersection(set(active_users1), set(active_users2)))

            if len(active_users) == 0:
                break


            actual_model = models_dict[strategy]

            if dataset_name in ["coat", "netflix","movielens", "KuaiRec", "yahoo"]:


                # this is only one step recomendation. each time there are k_length num of items recommend, loop until user quit.
                final_users_items = actual_model.get_recommendations(active_users, k_length, users_items_matrix, alpha, copula,
                                                                     use_relevance=True)

                for idx, uid in enumerate(active_users):
                    already_recom_list = recommendation_list[uid]
                    new_recom_list = final_users_items[idx]
                    new_recom_items = [item for item in new_recom_list if item not in set(already_recom_list)]
                    recommendation_list[uid].extend(new_recom_items)


                final_users_items = np.array(final_users_items)
                active_users = np.array(active_users)
                ratings = whole_rating_predicted[active_users[:, np.newaxis], final_users_items]


                if USE_WEIBULL:
                    l = lambdas_dict[user_budget]
                    q = np.exp(-1 / (l ** g))
                    u_budget = 1 - q ** ((users_steps[active_users] + 1) ** g - users_steps[active_users] ** g)
                else:
                    u_budget = users_steps[active_users] / user_budget

                theta = np.repeat(1 - u_budget, k_length).reshape(len(active_users), k_length)

                quitting_probability = np.full(len(active_users), 1 - theta[:, 0])

                if not COMPUTE_EXPECTATION:
                    for i in range(k_length - 1):
                        quitting_probability += np.prod(theta[:, :i + 1], axis=-1) * np.prod(1 - ratings[:, :i + 1], axis=-1) * (
                                1 - theta[:, i + 1])

                quitting_bernoulli = np.random.binomial(1, p=quitting_probability)

                p_u = ratings / np.sum(ratings, axis=-1, keepdims=True)

                for i in range(len(active_users)):

                    if quitting_bernoulli[i]:  # that user quits
                        users_quitting_vector[active_users[i]] = 1
                    else:
                        user_choice = np.random.choice(final_users_items[i], p=p_u[i])
                        users_items_matrix[active_users[i]][user_choice] = 1
                        users_steps[active_users[i]] += 1

            users_quitting_vector2 = np.all(users_items_matrix != 0, axis=1).astype(int)



        # after the trial end, check what are in the ranking,
        # if the ranking are not complete, extend the rest elements
        # in the dataset to the ranking arbitrarily.

        expectation_value_list = []
        length_generated = []
        final_output = [[] for j in range(n_users)]
        for uid in range(n_users):
            ranking_u = recommendation_list[uid]
            length_generated.append(len(ranking_u))
            missing_items = np.setdiff1d(np.arange(n_items), ranking_u)
            random.shuffle(missing_items)
            ranking_u.extend(missing_items)
            final_output[uid] =ranking_u

            items_continue_p = whole_rating_predicted[uid]
            expvalue = expectation_value_MSD(np.array(ranking_u), items_items_distances, items_continue_p)

            expectation_value_list.append(expvalue)

            timespent = time.time() - start

        return expectation_value_list,final_output, timespent


    results = Parallel(n_jobs=10, prefer="processes")(
                            delayed(process)() for t in range(num_trials))

    exp_all_trials = []
    max_exp = 0
    for exp_list, ranking, timespent in results:
        exp_all_trials.extend(exp_list)
        avg_exp = np.average(exp_list)
        if avg_exp > max_exp:
            max_exp = avg_exp
            best_ranking = ranking

    return best_ranking, exp_all_trials, timespent


def main_IR_dataset(dataset_name, regime, k_length, users_budget, num_trials):


# ********build vocab for the new dataset****************************************
    if dataset_name == 'LETOR' or dataset_name == 'letor':
        num_user = 1691
    elif dataset_name == 'LTRC':
        num_user = 1195
    elif dataset_name == 'LTRCB':
        num_user = 5154
    else:
        raise ValueError('please provide a valid dataset')

    folder_path = f'../ProcessedData/{dataset_name}/'

    ratings_vocab = {}
    distances_vocab = {}
    users_items_matrix = {}

    for uid in range(num_user):
        items_items_distances = np.load(folder_path + f'jaccard_genres_distances_{uid}.npy')
        items_items_distances = items_items_distances/np.max(items_items_distances)

        rating = np.load(folder_path + f'rating_{uid}.npy')
        rating = map2range(dataset_name, regime, rating, regimes_mapping_vocab)

        ratings_vocab[uid] = rating
        distances_vocab[uid] = items_items_distances
        users_items_matrix[uid] = np.zeros(len(rating))

        assert len(rating) == items_items_distances.shape[0]

    # ********finish building vocab ****************************************


    device_id = 1

    if torch.cuda.is_available():
        device = f"cuda:{device_id}"
        torch.cuda.set_device(device_id)
    else:
        device = "cpu"

    print(f"Device: {device}")

    default_n_jobs = 10


    copula = "clayton"

    USE_WEIBULL = True
    COMPUTE_EXPECTATION = False
    g = 2

    lambdas_dict = {5: 6.2, 10: 11.85, 20: 23.21, 30: 30 / 0.886227, 40: 40 / 0.886227, 50: 50 / 0.886227}


    if copula == "clayton":
        default_alphas = [0.0001, 0.001, 0.01, 0.5, 0.1, 1, 2, 3, 4]  # current experiments are with alpha=0.5


    def process():

        users_steps = np.zeros(num_user)

        print("Starting trial...")
        start_trial = time.time()

        users = np.arange(num_user)
        users_quitting_vector = np.zeros(num_user).astype(int)
        users_quitting_vector2 = np.zeros(num_user).astype(int)

        recommendation_list = [[] for i in range(num_user)]

        start = time.time()


        cnt_in_loop = 0
        while 0 in np.bitwise_or(users_quitting_vector, users_quitting_vector2):

            active_users_indicator = np.bitwise_or(users_quitting_vector, users_quitting_vector2)
            active_users = users[active_users_indicator[users] == 0]

            if len(active_users) == 0 :
                break

            final_users_items = get_recommendations_newdataset(ratings_vocab, distances_vocab, active_users, k_length, users_items_matrix, alpha=0.5, copula="clayton",
                        use_relevance=True)

            assert len(final_users_items) != 0

            for idx, uid in enumerate(active_users):
                already_recom_list = recommendation_list[uid]
                new_recom_list = final_users_items[idx]
                new_recom_items = [item for item in new_recom_list if item not in set(already_recom_list)]
                recommendation_list[uid].extend(new_recom_items)
                assert len(recommendation_list[uid])<= len(ratings_vocab[uid])


            rating_chunk = [[] for j in range(len(active_users))]
            for idx, uid in enumerate(active_users):
                rating_chunk[idx] = ratings_vocab[uid][final_users_items[idx]]

# ******************************************************************************************************************
            if USE_WEIBULL:
                l = lambdas_dict[users_budget]
                q = np.exp(-1 / (l ** g))
                u_budget = 1 - q ** ((users_steps[active_users] + 1) ** g - users_steps[active_users] ** g)
            else:
                u_budget = users_steps[active_users] / users_budget

            theta = np.repeat(1 - u_budget, k_length).reshape(len(active_users), k_length)
            # for now theta is constant in the examination of the recommendation list

            quitting_probability = np.full(len(active_users), 1 - theta[:, 0])

            if not COMPUTE_EXPECTATION:
                for i in range(k_length - 1):
                    # Handle list of lists for rating_chunk
                    theta_product = np.prod(theta[:, :i + 1], axis=-1)
                    rating_product = [np.prod(1-np.array(rating_chunk[u][:i + 1])) for u in range(len(active_users))]
                    quitting_probability += theta_product * np.array(rating_product) * (1 - theta[:, i + 1])

            quitting_bernoulli = np.random.binomial(1, p=quitting_probability)

            # Handle list of lists for rating_chunk
            p_u = [rating_chunk[idx] / np.sum(rating_chunk[idx]) for idx, uid in enumerate(active_users)]

            for idx, uid in enumerate(active_users):
                if quitting_bernoulli[idx]:  # that user quits
                    users_quitting_vector[uid] = 1
                else:
                    if not len(final_users_items[idx]) == len(p_u[idx]):
                        print(final_users_items[idx], len(final_users_items[idx]))
                        print(p_u[idx], len(p_u[idx]))
                        print(rating_chunk[idx])
                        print(final_users_items[idx])
                        print(ratings_vocab[uid][final_users_items[idx]])

                        raise ValueError('a and p  must have same size')
                    user_choice = np.random.choice(final_users_items[idx], p=p_u[idx])
                    users_items_matrix[uid][user_choice] = 1
                    users_steps[uid] += 1



            for uid in range(num_user):
                selected_items = users_items_matrix[uid]
                if np.all(selected_items):
                    users_quitting_vector2[uid] = 1




        expectation_value_list = []
        length_generated = []
        final_recommendation_list_output = []
        for uid in range(num_user):
            n_items = len(ratings_vocab[uid])
            ranking_u = recommendation_list[uid]
            length_generated.append(len(ranking_u))
            missing_items = np.setdiff1d(np.arange(n_items), ranking_u)
            random.shuffle(missing_items)
            ranking_u.extend(missing_items)


            items_continue_p = ratings_vocab[uid]
            distance_matrix = distances_vocab[uid]
            expvalue = expectation_value_MSD(np.array(ranking_u), distance_matrix, items_continue_p)

            expectation_value_list.append(expvalue)
            final_recommendation_list_output.append((uid, ranking_u))

        # cnt_in_loop += 1
        # if cnt_in_loop == 1:
        #     print(f'avg num of recommendation is {np.average(length_generated)}')

        timespent = time.time() - start

        return expectation_value_list, final_recommendation_list_output, timespent


    results = Parallel(n_jobs=10, prefer="processes")(
        delayed(process)() for t in range(num_trials))

    exp_all_trials = []
    max_exp = 0
    for exp_list, ranking, timespent in results:
        exp_all_trials.extend(exp_list)
        avg_exp = np.average(exp_list)
        if avg_exp > max_exp:
            max_exp = avg_exp
            best_ranking = ranking

    return best_ranking, exp_all_trials, timespent



regimes_mapping_vocab= {"small": [0.1, 0.3], "medium": [0.4, 0.6], "large": [0.7, 0.9], 'full':[0,1]}
if __name__ == '__main__':

    datasets = ["coat", "netflix","movielens", "KuaiRec", "yahoo", 'LETOR','LTRCB']
    datasets = ['LETOR']

    k_length_list = [1, 3, 5, 10]
    num_trials = 10
    budget_list = [5, 10, 20, 30, 40]


    for dataset in datasets:
        if dataset not in ['LETOR','LTRCB']:
            # for regime in ['small', 'medium', 'large', 'full']:
            for regime in ['full']:
                max_exp_value = 0
                for budget in budget_list:
                    for k_length in k_length_list:

                        print(dataset, regime, budget, k_length)
                        ranking, exp_all_trials, spendtime = main_recommendation_datasets(dataset, regime, k_length, budget,
                                                                                 num_trials)
                        expvalue = np.average(exp_all_trials)

                        if expvalue > max_exp_value:
                            max_exp_value = expvalue
                            max_exp_std = np.std(exp_all_trials)
                            best_ranking_list = ranking
                            best_k_length = k_length
                            best_budgets = budget

                rst = (
                    'EXPLORE', dataset, best_k_length, best_budgets, regime,max_exp_value, max_exp_std, spendtime)

                print(rst)
                result_path = f"../results/{dataset}_{regime}.txt"
                ensure_folder_exists('../results')

                with open(result_path, 'a+') as file:
                    row_str = '\t'.join(map(str, rst))
                    file.write(row_str + '\n')

                save_ranking_2file_budget(dataset, best_k_length, regime, best_budgets, best_ranking_list, 'EXPLORE')

        else:
            # for regime in ['small', 'medium', 'large', 'full']:
            for regime in ['full']:

                max_exp_value = 0

                for budget in budget_list:
                    for k_length in k_length_list:

                        # for regime in ['large']:
                        print(dataset, regime, budget, k_length)
                        ranking, exp_all_trials, spendtime = main_IR_dataset(dataset, regime, k_length, budget,
                                                                                 num_trials)
                        expvalue = np.average(exp_all_trials)

                        if expvalue > max_exp_value:
                            max_exp_value = expvalue
                            max_exp_std = np.std(exp_all_trials)
                            best_ranking_list = ranking
                            best_k_length = k_length
                            best_budgets = budget

                rst = (
                    'EXPLORE', dataset, best_k_length, best_budgets, regime, max_exp_value, max_exp_std, spendtime)

                print(rst)
                result_path = f"../results/{dataset}_{regime}.txt"
                ensure_folder_exists('../results')

                with open(result_path, 'a+') as file:
                    row_str = '\t'.join(map(str, rst))
                    file.write(row_str + '\n')

                save_qid_ranking_2file(dataset, best_k_length, best_budgets, regime, best_ranking_list, 'EXPLORE')
