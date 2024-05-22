from joblib import Parallel, delayed

from utility import *

'''
The structure of your SPECIFY_DATA_FOLDER should be like the following tree to run properly

.
├── KuaiRec
│         ├── LICENSE
│         ├── Statistics_KuaiRec.ipynb
│         ├── data
│         │         ├── big_matrix.csv
│         │         ├── item_categories.csv
│         │         ├── item_daily_features.csv
│         │         ├── small_matrix.csv
│         │         ├── social_network.csv
│         │         └── user_features.csv
│         ├── figs
│         │         ├── KuaiRec.png
│         │         └── colab-badge.svg
│         └── loaddata.py
├── coat
│         ├── README.txt
│         ├── propensities.ascii
│         ├── test.ascii
│         ├── train.ascii
│         └── user_item_features
│             ├── item_features.ascii
│             ├── item_features_map.txt
│             ├── user_features.ascii
│             └── user_features_map.txt
├── movielens
│         ├── README
│         ├── movies.dat
│         ├── ratings.dat
│         └── users.dat
├── netflix
│         ├── Netflix_Dataset_Rating.csv
│         └── netflix_genres.csv
└── yahoo
    ├── genre-hierarchy.txt
    ├── readme.txt
    ├── song-attributes.txt
    ├── test_0.txt
    ├── test_1.txt
    ├── test_2.txt
    ├── test_3.txt
    ├── test_4.txt
    ├── test_5.txt
    ├── test_6.txt
    ├── test_7.txt
    ├── test_8.txt
    ├── test_9.txt
    ├── train_0.txt
    ├── train_1.txt
    ├── train_2.txt
    ├── train_3.txt
    └── train_4.txt

'''
def main():

    def preprocess_dataset(dataset_name):
        print(f"DATASET: {dataset_name}")
        dataset_folder = os.path.join("RawData/", dataset_name)
        # dataset_folder = os.path.join(SPECIFY_DATA_FOLDER, dataset_name)

        outputs_folder = "outputs/"
        if not os.path.exists(outputs_folder):
            os.makedirs(outputs_folder)


        ratings = get_ratings(dataset_name, dataset_folder)

        # users_dictionary: {string_user_name, reindexed_user_name}
        # items_dictionary: {string_item_name, reindexed_item_name}
        users_dictionary, items_dictionary = get_dictionaries(ratings, outputs_folder, dataset_name)

        print("#Users", len(users_dictionary))
        print("#Items", len(items_dictionary))
        print("#Ratings", len(ratings))

        items_users_matrix_path = f"outputs/items_users_matrix_{dataset_name}.npy"

        # reindexed user-item-rating matrix.
        items_users_matrix = get_items_users_matrix(ratings, items_dictionary, users_dictionary,
                                                    items_users_matrix_path)
        users_jaccard_path = f"outputs/jaccard_users_distances_{dataset_name}.npy"


        users_jaccard_matrix = get_jaccard_matrix(items_users_matrix, users_jaccard_path)

        print("Jaccard-Users Matrix Statistics:")
        print_matrix_statistics(users_jaccard_matrix)

        items_genres_matrix_path = f"outputs/items_genres_matrix_{dataset_name}.npy"

        items_genres_matrix = get_items_genres_matrix(items_dictionary, dataset_name, dataset_folder,
                                                      items_genres_matrix_path)

        # genres for all dataset are finite set, and genres are 0-1 vector.
        # thus weighted jaccard distance is equal to jaccard distance, which is a metric.
        genres_jaccard_path = f"outputs/jaccard_genres_distances_{dataset_name}.npy"
        genres_jaccard_matrix = get_jaccard_matrix(items_genres_matrix, genres_jaccard_path)

        print("Jaccard-Genres Matrix Statistics:")
        print_matrix_statistics(genres_jaccard_matrix)


    datasets = ["coat", "KuaiRec", "netflix", "yahoo","movielens"]

    for dataset_name in datasets:
        preprocess_dataset(dataset_name)


if __name__ == '__main__':
    main()

