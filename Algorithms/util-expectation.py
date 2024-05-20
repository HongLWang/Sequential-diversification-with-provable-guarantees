
import numpy as np
import torch
from itertools import combinations, permutations
import  pandas._libs.lib as lib


def expectation_value_MHP(permutations, distance_matrix, items_continue_p):
    matrix_p = items_continue_p[permutations]  # size is num of permutations * k_param

    accumulate_p = np.cumprod(matrix_p, axis=1)[:, 1:]  # size is num of permutations * k_param-1

    # wi = p1p2..p_{i+1} + ...+ p1p2..p_n
    weights = np.cumsum(accumulate_p[:, ::-1], axis=1)[:, ::-1]

    indices = (permutations[:, :-1], permutations[:, 1:])

    # size is num of permutations * k_param-1
    distances = distance_matrix[indices]

    expectations = np.sum(weights * distances, axis=1, keepdims=True)

    return expectations

def expectation_MHP_incremental(permutations, distance_matrix, items_continue_p):
    expectations = 0

    accu_p= 0
    accu_dist = 0

    for idx in range(1, permutations.shape[1]):

        if idx == 1:  # start of incrementation
            ######################### incrementally calculate MHP expectation ################################
            top2 = permutations[:,:2] # maybe these two columns can be retrived seperately in later optimization
            matrix_p = items_continue_p[top2]   # check this does what you want

            accu_p = np.multiply(matrix_p[:,0], matrix_p[:,1])

            indices = (permutations[:,0], permutations[:,1])
            # size is num of permutations * k_param-1
            accu_dist = distance_matrix[indices]

            expectations = np.multiply(accu_p , accu_dist) # num_permu * 1

            ########################################################################################

        else:  # when seq_len is >=3, do incrementation

            new_item = permutations[:,idx]
            last_item_p = items_continue_p[new_item]
            accu_p = np.multiply(last_item_p, accu_p)

            indices = (permutations[:,idx-1], permutations[:,idx])
            increment_dist = distance_matrix[indices]
            accu_dist += increment_dist

            increment_exp = np.multiply(accu_p, accu_dist)
            expectations += increment_exp

    return expectations
def separate_to_chunks(lst, k):
    chunk_size = len(lst) // k
    remainder = len(lst) % k

    start_index = 0
    end_index = 0
    chunk_indices = []
    for i in range(k):
        if i < remainder:
            end_index += chunk_size + 1
        else:
            end_index += chunk_size

        if end_index > len(lst):
            end_index = len(lst)

        chunk_indices.append((start_index, end_index))
        start_index = end_index

    return chunk_indices

def expectation_MHP_incremental_chunk(permutations, distance_matrix, items_continue_p, num_thred=5000):

    def chunk_computation(permutations_chunk):
        expectations = 0

        accu_p = 0
        accu_dist = 0

        for idx in range(1, permutations_chunk.shape[1]):

            if idx == 1:  # start of incrementation
                ######################### incrementally calculate MHP expectation ################################
                top2 = permutations_chunk[:,
                       :2]  # maybe these two columns can be retrived seperately in later optimization
                matrix_p = items_continue_p[top2]  # check this does what you want

                accu_p = np.multiply(matrix_p[:, 0], matrix_p[:, 1])

                indices = (permutations_chunk[:, 0], permutations_chunk[:, 1])
                # size is num of permutations_chunk * k_param-1
                accu_dist = distance_matrix[indices]

                expectations = np.multiply(accu_p, accu_dist)  # num_permu * 1

                ########################################################################################

            else:  # when seq_len is >=3, do incrementation

                new_item = permutations_chunk[:, idx]
                last_item_p = items_continue_p[new_item]
                accu_p = np.multiply(last_item_p, accu_p)

                indices = (permutations_chunk[:, idx - 1], permutations_chunk[:, idx])
                increment_dist = distance_matrix[indices]
                accu_dist += increment_dist

                increment_exp = np.multiply(accu_p, accu_dist)
                expectations += increment_exp

        return expectations


    n_thread = num_thred
    num_permu = permutations.shape[0]
    chunk_range = separate_to_chunks(np.arange(num_permu), n_thread)

    results = []
    for i in range(len(chunk_range)):
        chunk = chunk_range[i]
        chunk_permutations = permutations[chunk[0]:chunk[1]]
        result = chunk_computation(chunk_permutations)

        results.append(result)

    expectation_concat = np.concatenate(results)

    assert len(expectation_concat) == permutations.shape[0]

    return expectation_concat

def generate_combinations_and_permutations(all_items, k_param):
    result = []
    # print ('items before permutation', all_items)
    for c in combinations(all_items, k_param):
        result.extend(permutations(c))

    all_permutations_array =  lib.to_object_array(result).astype(int)
    return all_permutations_array


def expectation_value_MSD_torch(permutations, distance_matrix, items_continue_p):

    permutations = torch.tensor(permutations)
    permutations = permutations.to(torch.int64)
    distance_matrix = torch.tensor(distance_matrix)
    items_continue_p = torch.tensor(items_continue_p)


    num_permutation, num_items = permutations.shape
    matrix_p = items_continue_p.gather(1, permutations)  # size is num of permutations * k_param

    # [p1p2, p1p2p3, p1p2p3p4, \cdots, p1...p_n]
    double_matrix_p = matrix_p.double()
    accumulate_p = torch.cumprod(double_matrix_p, dim=1)[:, 1:]  # size is num of permutations * k_param-1

    # [d(1,2), d(3,{1,2}), d(4,{1,2,3}), ...d(n, {1,2,...,n-1})], the i-th element is d(i+1, {1,2,...i})
    distances = torch.zeros((num_permutation, num_items-1), dtype=torch.double)
    for i in range(1, num_items):
        new_node = permutations[:, i]
        existing_nodes = permutations[:, :i]

        matrix_row_slices = distance_matrix[new_node]
        matrix_column_slices = matrix_row_slices.gather(1, existing_nodes)

        distance_new2exist = torch.sum(matrix_column_slices, dim=1)
        distances[:, i-1] = distance_new2exist

    expectations = torch.einsum('ij,ij->i', accumulate_p, distances)

    return expectations

# this is accelarated by chatgpt
def expectation_value_MSD_incremental(permutations, distance_matrix, items_continue_p):
    num_permutation, num_items = permutations.shape

    # Initialize results and accumulate initial probabilities
    first_2_p = items_continue_p[permutations[:, 0]] * items_continue_p[permutations[:, 1]]
    dist = distance_matrix[permutations[:, 0], permutations[:, 1]]
    expectation = first_2_p * dist

    # For storing accumulated probabilities
    accum_p = first_2_p

    for idx in range(2, num_items):
        new_item = permutations[:, idx]
        new_p = items_continue_p[new_item]
        accum_p *= new_p  # Update the accumulated probabilities

        # Compute the sum of distances from the new item to all previously considered items
        dist_increment = np.zeros(num_permutation)
        for prev_idx in range(idx):  # Loop over all previous items
            prev_item = permutations[:, prev_idx]
            dist_increment += distance_matrix[new_item, prev_item]

        # Update expectation
        expectation += accum_p * dist_increment

    return expectation


# version for greedy and other strategy.
def save_ranking_2file(dataset_name, k_param, regime, ranking, strategy):
    filename = 'ranking/' + dataset_name + '_'+ str(k_param) + '_'+ regime + '_'+ strategy + '.txt'
    with open(filename, 'a+') as writer:
        for row in ranking:
            row_string = ','.join(map(str, row))
            writer.write(row_string + '\n')


# def save_ranking_2file(dataset_name, k_param, regime, ranking, strategy):
#     filename = 'debug/' + dataset_name + '_'+ str(k_param) + '_'+ regime + '_'+ strategy + '.txt'
#     with open(filename, 'a+') as writer:
#         for row in ranking:
#             row_string = ','.join(map(str, row))
#             writer.write(row_string + '\n')


# do not delete
# version used in bke_filtering, does not consider strategy, so in the end all filtering method get to be saved in the same file.
def save_ranking_2file_old(dataset_name, k_param, regime, ranking):
    filename = 'ranking/' + dataset_name + '_'+ str(k_param) + '_'+ regime + '_large_k.txt'
    with open(filename, 'a+') as writer:
        for row in ranking:
            row_string = ','.join(map(str, row))
            writer.write(row_string + '\n')