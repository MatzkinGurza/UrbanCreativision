# IMPORTS

import numpy as np
import pandas as pd
import time
import os

# FUNCTIONS

def euclidean_distance(v1, v2):
    if v1.shape != v2.shape:
        raise ValueError("Vectors must have the same shape")
    
    return np.linalg.norm(v1 - v2)

def jaccard_similarity(s1, s2):

# CLASSES

# MAIN SCRIPT