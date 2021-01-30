import numpy as np
import time


def naive_relu(x):
    assert len(x.shape) == 2  # x is a 2D Numpy tensor
    x = x.copy()  # avoid overwriting the input tensor
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    return x


def naive_add(x, y):
    assert len(x.shape) == 2  # x and y are 2D Numpy tensors
    assert x.shape == y.shape

    x = x.copy()  # avoid overwriting the input tensor
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[i, j]
    return x


np_array_1 = np.array([[1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [3, 4]])
np_array_2 = np.array([[1, 2], [1, 2], [1, 2], [1, 2], [3, 4], [2, 2]])

tempo_inicio = time.time()
z = np_array_1 + np_array_2
tempo_fim = time.time()
print(z.shape)
print('Tempo: ', tempo_fim - tempo_inicio)

tempo_inicio = time.time()
z = naive_add(np_array_1, np_array_2)
tempo_fim = time.time()
print(z.shape)
print('Tempo: ', tempo_fim - tempo_inicio)
