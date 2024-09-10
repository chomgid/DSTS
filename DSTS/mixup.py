import numpy as np


def make_r(data) -> np.ndarray:
    r = np.ones_like(data[:,1:])
    col_num = data.shape[1]
    for col_index in range(0, col_num-1) : 
        r[:,col_index] = data[:,col_index+1]/data[:,0]
    return r


def make_alpha(data) -> np.ndarray:
    means = np.mean(data, axis=0)
    variances = np.std(data, axis=0)
    alpha = means**2 / variances
    return alpha


def make_rs_index(data, size, k):
    index_array = np.arange(size)
    arrays_list = []

    for i in range(size):
        x = data[i]
        xmat = np.delete(data, i, axis=0)
        prob = proba(x, xmat)
        del_arr = np.delete(index_array, i, axis=0)
        new_array = np.random.choice(del_arr, k, replace=True, p=prob)
        arrays_list.append(new_array)
    rs_index = np.stack(arrays_list)
    return rs_index


# calculate probabilities for SNN matching
def proba(x, xmat, temp=1):
    dist = np.linalg.norm(xmat-x, ord=2, axis=1)
    # use log sum exp trick
    min = np.min(dist)
    denom = np.log(sum(np.exp(-(dist-min)/temp)))-min/temp
    num = -dist/temp
    prob = np.exp(num-denom)
    return prob


# rstar matrix in a way that ensures the elements follow 12341234 ordering rather than 11223344 ordering.
def make_rstar(data, k, sort) -> np.ndarray:
    """
    make rstar matrix
    """
    size = data.shape[0]
    r = make_r(data)
    rs_index = make_rs_index(data, size, k)
    y1i = data[:,:1]
    rstar = []
    ystar = []
    for j in range(k):
        rs = r[rs_index[:,j]]
        y1s = y1i[rs_index[:,j]]
        lamb = np.random.beta(a=0.5, b=0.5, size=(size,1))
        rstar.append(lamb*r + (1-lamb)*rs)
        ystar.append(lamb*y1i + (1-lamb)*y1s)

    rstar_matrix = np.vstack(rstar)
    ystar_matrix = np.vstack(ystar).squeeze()
    if sort:
        index = np.argsort(ystar_matrix)
        rstar_matrix = rstar_matrix[index]
    return rstar_matrix


