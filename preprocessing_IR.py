from sklearn.metrics.pairwise import cosine_distances
import numpy as np
import os
from tqdm import tqdm
import pandas as pd





def ensure_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        pass


def preprocessing_LETOR():
    def build_vocab():
        doc_vocab = {}
        query_vocab = {}
        query_doc_pair = set()

        with open(file_path, 'r') as file:

            for line in file:

                if line == '\n':
                    continue

                parts = line.split()
                relevance = int(parts[0])
                qid = int(parts[1].split(':')[1])
                feature_vector = [float(feature.split(':')[1]) for feature in parts[2:48]]
                docid = parts[50]
                if sum(feature_vector) == 0:
                    print('all 0 features encountered')
                    continue

                pair = (qid, docid)
                if pair not in query_doc_pair:
                    query_doc_pair.add(pair)
                else:
                    continue

                # Construct document vocabulary
                if docid not in doc_vocab:
                    doc_vocab[docid] = feature_vector

                # Construct query vocabulary
                if qid not in query_vocab:
                    query_vocab[qid] = []

                query_vocab[qid].append([docid, relevance])

        filtered_query_vocab = {k: v for k, v in query_vocab.items() if len(v) >= 10}

        return filtered_query_vocab, doc_vocab


    def save_processed_data(query_vocab, doc_vocab):

        qid_mapping = {old_qid: new_qid for new_qid, old_qid in enumerate(query_vocab.keys())}

        for old_qid, docs in query_vocab.items():
            new_qid = qid_mapping[old_qid]

            # Reindex docids for the current qid
            docid_mapping = {old_docid: new_docid for new_docid, (old_docid, _) in enumerate(docs)}

            # Initialize A and collect feature vectors
            A = np.zeros(len(docs))
            feature_matrix = np.zeros((len(docs), feature_dimension))

            for old_docid, relevance in docs:
                new_docid = docid_mapping[old_docid]
                A[new_docid] = relevance
                feature_matrix[new_docid] = doc_vocab[old_docid]

            M = cosine_distances(feature_matrix)


            ensure_folder_exists('ProcessedData/LETOR')

            distance_file_path = f'ProcessedData/LETOR/jaccard_genres_distances_{new_qid}.npy'
            np.save(distance_file_path, M)

            rating_file_path = f'ProcessedData/LETOR/rating_{new_qid}.npy'
            np.save(rating_file_path, A)


    file_path = 'RawData/LETOR/letor.txt'

    feature_dimension = 46

    filtered_query_vocab, doc_vocab = build_vocab()

    save_processed_data(filtered_query_vocab, doc_vocab)

def preprocessing_LTRC():
    def build_feature_vec(feature_string_list, dim):
        vec = np.zeros(dim)
        for index_value in feature_string_list:
            index = int(index_value.split(':')[0]) - 1
            value = float(index_value.split(':')[1])
            vec[index] = value
        return vec

    def preprocessing(dataset_name):
        query_vec_vocab = {}
        qurey_relevance_vocab = {}
        feat_dime = 700

        with open(file_path, 'r') as reader:

            lines = reader.readlines()
            for idx in tqdm(range(len(lines))):
                line = lines[idx]

                if line == '\n':
                    continue

                parts = line.split()
                relevance = int(parts[0])
                qid = int(parts[1].split(':')[1])
                feature_vector = build_feature_vec(parts[2:], feat_dime)
                if sum(feature_vector) == 0:
                    print('all 0 features encountered')
                    continue

                # Construct query vocabulary
                if qid not in query_vec_vocab:
                    query_vec_vocab[qid] = []
                    qurey_relevance_vocab[qid] = []

                query_vec_vocab[qid].append(feature_vector)
                qurey_relevance_vocab[qid].append(relevance)

        # after finished reading the data, filter out identical vectors
        # check if folder exist, else mkdir
        outputfilepath = f'../../ProcessedData/{dataset_name}'
        if os.path.exists(outputfilepath):
            pass
        else:
            os.makedirs(outputfilepath)

        final_vocab = {}
        new_qid = 0


        for qid, vector_list in tqdm(query_vec_vocab.items()):


            vec_matrix = np.array(vector_list)
            relevance_arr = np.array(qurey_relevance_vocab[qid])

            df_M = pd.DataFrame(vec_matrix)
            series_R = pd.Series(relevance_arr)

            # Drop duplicate rows in M while keeping the first occurrence
            df_M_unique = df_M.drop_duplicates(keep='first')

            # Get the indices of the unique rows
            unique_indices = df_M_unique.index

            # Filter the Series R based on the unique indices
            series_R_unique = series_R.iloc[unique_indices]

            # Convert back to numpy arrays if needed
            M_unique = df_M_unique.to_numpy()
            R_unique = series_R_unique.to_numpy()

            if len(R_unique) < 10:
                continue
            else:
                M = cosine_distances(M_unique)
                assert M.shape[0] == len(R_unique)
                #
                # final_vocab[new_qid] = (R_unique, M)

                distance_file_path = outputfilepath + f'/jaccard_genres_distances_{new_qid}.npy'
                np.save(distance_file_path, M)

                rating_file_path = outputfilepath + f'/rating_{new_qid}.npy'
                np.save(rating_file_path, R_unique)

                new_qid += 1



    dataset_name = 'LTRCB'
    file_path = 'RawData/YahooLERC/set1.test.txt'
    preprocessing(dataset_name)

if __name__ == '__main__':


    dataset_list = ['LETOR','LTRCB']

    # preprocessing_LETOR() # processing LETOR dataset
    preprocessing_LTRC()




