import numpy as np
import math

def norm(v):
    return np.linalg.norm(v)

def normalize(v):
    return v / norm(v)

def atan2(v):
    a = math.atan2(v[1], v[0])
    return a if a >= 0 else a + 2 * math.pi
