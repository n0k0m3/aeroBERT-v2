import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))


# use dataset split given from the author of the paper for classification task
import pandas as pd
from datasets import load_dataset, Dataset

if __name__ == "__main__":
    train_df = pd.read_csv("./classifier/train_data_classification.csv")
    train_df.columns = ["requirements","label"]
    test_df = pd.read_csv("./classifier/test_data_classification.csv")
    test_df.columns = ["requirements","label"]
    # convert integer labels to string labels
    # Design requirements: 0
    # Functional requirements: 1
    # Performance requirements: 2
    train_df["label"] = train_df["label"].map({0: "Design", 1: "Functional", 2: "Performance"})
    test_df["label"] = test_df["label"].map({0: "Design", 1: "Functional", 2: "Performance"})

    # make sure df is in correct format for hf datasets
    train = Dataset.from_pandas(train_df)
    test = Dataset.from_pandas(test_df)

    # save to file
    train.to_json("./classifier/train.json")
    test.to_json("./classifier/test.json")