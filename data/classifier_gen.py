import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))


# download archanatikayatray/aeroBERT-NER huggingface dataset
# https://huggingface.co/datasets/archanatikayatray/aeroBERT-NER
import pandas as pd
from datasets import load_dataset, Dataset

if __name__ == "__main__":
    dataset = load_dataset("archanatikayatray/aeroBERT-classification")["train"]["text"]
    # load dataset as pandas dataframe, header is first line, delimiter is *
    df = pd.DataFrame([l.split("*") for l in dataset[1:]], columns=dataset[0].split("*"))

    # convert integer labels to string labels
    # Design requirements: 0
    # Functional requirements: 1
    # Performance requirements: 2
    df["label"] = df["label"].map({"0": "Design", "1": "Functional", "2": "Performance"})

    # make sure df is in correct format for hf datasets
    dataset = Dataset.from_pandas(df)

    # train/test split
    data = dataset.train_test_split(test_size=0.1, seed=42)

    # save to file
    data["train"].to_json("./classifier/train.json")
    data["test"].to_json("./classifier/test.json")