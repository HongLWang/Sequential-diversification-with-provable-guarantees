# Relevance-aware sequential diversification

### USAGE
In order to reproduce the experiments, please follow the steps:

### 1. Install the requirements
   Run `$ pip install -r requirements.txt`
### 2. Download Raw Data
- Coat dataset: https://www.cs.cornell.edu/~schnabts/mnar/
- Netflix: https://www.kaggle.com/datasets/rishitjavia/netflix-movie-rating-dataset?select=Netflix_Dataset_Rating.csv, the genre information for Netflix dataset is avaliable at https://github.com/tommasocarraro/netflix-prize-with-genres
- Movielenss: https://grouplens.org/datasets/movielens/1m/
- Yahoo: https://webscope.sandbox.yahoo.com/catalog.php?datatype=i&did=67
- KuaiRec: https://kuairec.com/

### 3. Preprocess the datasets 
- Specify the path folder in which you store the raw data in "RawData" folder, or specify your own location.
- In `preprocessing.py`, you can choose which dataset to preprocess. The processed data will be available under `output` folder.

### 4. Obtain relevance score (item continuation probability in our paper)
- Run `MF_main.py` to train the Matrix Factorization model that provides the relevance scores. The completed rating matrix will be saved under folder `ProcessedData`.

### 5. Run the algorithms 
#### Note that you can skip step 3 and 4 because the processed data for the "coat" dataset is provided. You can run the following code on "coat" dataset.
- In folder `Algorithms` you can find our proposed algorithms `B-tau-I.py`, `B-tau-I-H.py` and `Greedy.py`.
- Also in folder `Algorithms` you can find baseline algorithms `Baseline_DPP.py`, `Baseline_DUM.py`, `Baseline_MSD.py`, `Baseline_MMR.py`, `Baseline_Random.py`.
- The average MaxSSD objective values as long as their standard derivation for each dataset are saved in folder `Result`.
- The actual rankings of items returned by each algorithm are saved in folder `ranking`.


