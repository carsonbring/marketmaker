import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys


def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


def kmeans(data, k):
    print("hello")
    # nothing yet
