import numpy as np
from typing import Generator
# import matplotlib.pyplot as plt

# hyperparameters
length = 200
size = round(length / 4)
noise_rate = 0.05

# functions
def nosify(ndarray: np.ndarray) -> np.ndarray:
    random = np.random.rand(ndarray.shape[0], ndarray.shape[1])
    return ndarray + random.astype(np.float32) * noise_rate

def shuffle(*ndarray: np.ndarray) -> Generator[np.ndarray, None, None]:
    indexes = np.arange(ndarray[0].shape[0])
    np.random.shuffle(indexes)
    for arr in ndarray:
        yield arr[indexes]

# set data
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# increase dataset length
x_repeat = x.repeat(size, axis=0)
y_repeat = y.repeat(size, axis=0)

# add noise and shuffle
x_noise = nosify(x_repeat)
x_ready, y_ready = shuffle(x_noise, y_repeat)