import math
from re import split
import torch
import torch.nn as nn
import torch.nn.functional as func
import requests
from dotenv import load_dotenv
from torch.serialization import load
import os
import pandas
import pickle


def retrieve_data(ticker: str):
    """
    Output: dataframe output of output
    """

    if os.path.exists("data.pkl"):
        with open("data.pkl", "rb") as f:
            df = pickle.load(f)

    else:
        load_dotenv()
        av_key = os.getenv("AV_API_KEY")
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={av_key}&outputsize=full"
        r = requests.get(url)
        data = r.json()
        df = pandas.DataFrame(data["Time Series (Daily)"])

        with open("data.pkl", "wb") as f:
            pickle.dump(df, f)

    return df


def tensor_convert(df: pandas.DataFrame):
    t_list = []
    df_list = split_dataframe(df)
    for df in df_list:
        df_numeric = df.apply(pandas.to_numeric, errors="coerce")
        tensor = torch.tensor(df_numeric.values, dtype=torch.float32)
        tensor = tensor.T
        tensor = tensor.unsqueeze(0)
        t_list.append(tensor)

    batch_tensor = torch.cat(t_list[:-1], dim=0)
    print("Batch tensor shape: ", batch_tensor.shape)
    print(batch_tensor)


def split_dataframe(df, chunk_size=365):
    return [df.iloc[:, i : i + chunk_size] for i in range(0, df.shape[1], chunk_size)]
