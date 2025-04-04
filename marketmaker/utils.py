import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import torch


# Needs to be euclidean distance since we have multiple dimensions
def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


def kmeans(data, k):
    """
    data: IntTensor of shape (sample_states, D)
    k: Number of centroids to find from the sample observed states
    Returns: Vector with the centroids
    """

    num_vectors, dimensions = data.shape
    # So the plan is to set the distances for each centroid to infinity and
    # then find the minimum distance along dim=0.
    # Then I will be able to find the new centroid based on finding the maximum
    # out of these minimums, which will be coallated into a new tensor
    # using gather
    # Finally I will then be able to to take the index of the maximum to find
    # the next centroid.
    # The centroids will be stored in a tensor with the shape (k, dimensions)

    distance_tensor = data.clone().detach()
    distance_tensor = distance_tensor.unsqueeze(0).reshape(k, -1, -1)
    torch.full((k, num_vectors, dimensions), -
               float("inf"), dtype=torch.float32)

    # nothing yet
    for i in range(k):
        centroid = torch.randint(0, num_vectors, (1,))
        # append each distance into a single tensor and then take the min of a vector?
        # We then take the original index of highest distance vector
