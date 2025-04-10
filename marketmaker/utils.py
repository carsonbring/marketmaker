import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import torch


# euclidean distance
def distance(p1, p2):
    return torch.cdist(p1, p2)


# deterministic kmeans
def kmeans(data, k):
    """
    data: IntTensor of shape (sample_states, D)
    k: Number of centroids to find from the sample observed states
    Returns: Vector with the centroids
    """

    num_vectors = data.shape[0]

    distance_tensor = torch.full((k, num_vectors), float("inf"), dtype=torch.float32)
    centroid_index = torch.randint(0, num_vectors, (1,))
    centroid_vector = data[centroid_index]
    centroid_tensor = centroid_vector

    for _ in range(k - 1):
        distance_tensor = distance(centroid_tensor, data)
        _, max_index = distance_tensor.min(dim=0).values.max(dim=0)  # pyright: ignore
        print(f"DEBUG: centroid_tensor shape = {centroid_tensor}")
        print(f"DEBUG: new_centroid_to_add shape = {data[max_index].unsqueeze(0)}")
        centroid_tensor = torch.cat(
            (centroid_tensor, data[max_index].unsqueeze(0)), dim=0
        )
    print("Centroids", centroid_tensor)

    return centroid_tensor
