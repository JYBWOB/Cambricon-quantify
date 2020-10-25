import numpy as np

def get_mse(x, y):
    if x.shape != y.shape:
        print('The shape of the two data are different')
        print(x.shape, y.shape)
        return None
    x = x.flatten()
    y = y.flatten()
    sub = x - y
    sub = np.square(sub)
    return np.sum(sub) / sub.shape[0]


def get_rmse(x, y):
    if x.shape != y.shape:
        print('The shape of the two data are different')
        return None
    return get_mse(x, y) ** 0.5