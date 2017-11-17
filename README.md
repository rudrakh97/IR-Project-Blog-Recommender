# Instructions to run
1. Download the dataset in `.csv` format from here - https://www.kaggle.com/rtatman/blog-authorship-corpus  
2. Save the csv file in the same directory and run `cluster_2.0_train.py` to train and save the models. For example, `python cluster_2.0_train.py 25000 4` to train `25000` sample blogs and having `4` clusters.  
3. Run `cluster_2.0_test.py` to test on query. For example, `python cluster_2.0_test.py president bush` for getting results for the query `president bush`.
