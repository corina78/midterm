import pandas as pd
import numpy as np
import os
import ast

def load_datasets(dataset):

    "Argument: loads train, test or validation dataset"
    "Returns: a pandas dataframe"

    file_path = "/home/corina/Documents/Math_Machine_Learning/midtermExam/"

    df = pd.read_csv(os.path.join(file_path, dataset))
    print(df)
    print(type(df["pos_tags"][0]))

    # Convert the string representation of the vectors into lists
    df['pos_tags'] = df['pos_tags'].apply(ast.literal_eval)
    df['tokens'] = df['tokens'].apply(ast.literal_eval)

    return df

if __name__ == "__main__":

    data = load_datasets("valid_data.csv")
    print(data)
    print(type(data["tokens"][0]))