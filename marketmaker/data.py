import torch
import requests
from pydantic_settings import BaseSettings
import os
import pandas
import pickle


class Settings(BaseSettings):
    av_key: str = "default_api_key"
    debug: bool = False

    class Config:
        env_file = ".env"


def retrieve_data(ticker: str):
    """
    Output: dataframe output of output
    """
    settings = Settings()

    if os.path.exists("data.pkl"):
        with open("data.pkl", "rb") as f:
            df = pickle.load(f)

    else:
        av_key = settings.av_key
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
    return batch_tensor


def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    bsize, T_max, D = tensor.shape
    flattened = tensor.reshape(-1, D).float()

    mean = flattened.mean(dim=0)
    std = flattened.std(dim=0)
    eps = 1e-8
    std[std < eps] = 1.0

    flattened_normalized = (flattened - mean) / std

    tensor_norm = flattened_normalized.reshape(bsize, T_max, D)
    return tensor_norm


def split_dataframe(df, chunk_size=365):
    return [df.iloc[:, i : i + chunk_size] for i in range(0, df.shape[1], chunk_size)]
