import numpy as np


def num_Newton(lambda_, weights, aug_data, desired_means, n):
    """
    numerator of Newton's method (f(lambda))
    """
    weights_prev = weights
    weights_post = weights_prev * np.exp(-aug_data * lambda_ )
    weights_post = weights_post / np.sum(weights_post) * n
    
    return ((aug_data.T @ weights_post)/n - desired_means)


def denom_Newton(lambda_, weights, aug_data, desired_means, m):
    """
    denominator of Newton's method (f'(lambda))
    """
    A = aug_data.T @ (weights * np.exp(-aug_data * lambda_ ))
    B = np.sum(weights * np.exp(-aug_data * lambda_ ))
    dA = -(aug_data**2).T @ (weights * np.exp(-aug_data * lambda_ ))
    dB = -A

    return (dA*B-A*dB)/B**2


# optimized by Newton's method
def calibration(ori_data:np.ndarray, aug_data:np.ndarray, iter, tot_iter):
    """
    Calibrate synthesized data so that cross-sectional data structure is preserved.

    Parameters:
    ori_data (np.ndarray): Input data array of shape (size, length).
    aug_data (np.ndarray): Generated data array of shape (size, length)
    iter (int): how many times Newton-Rhapson update is performed for each timestamp
    tot_iter (float): how many times the whole time series is updated

    Returns:
    np.ndarray: Calibrated data array of shape (size * aug, length).

    """
    n=len(ori_data)
    m=len(aug_data)
    init_weights=np.ones(len(aug_data)) / m

    # benchmark information
    desired_means=np.mean(ori_data, axis=0)

    # lambda update usig Newton's method (iter*tot_iter times)
    init_lambda=np.zeros(len(desired_means))
    lamb=init_lambda
    weights=init_weights
    for _ in range(tot_iter):
        for i in range(0,len(aug_data.T)):
            for _ in range(iter):
                eps = num_Newton(lamb[i], weights, aug_data[:,i], desired_means[i], n)/denom_Newton(lamb[i], weights, aug_data[:,i], desired_means[i], m)
                lamb[i] = lamb[i] - eps
            weights_calib=weights*np.exp(-aug_data[:,i]*lamb[i])
            weights = weights_calib/np.sum(weights_calib)*n


                    
    # weights normalization
    weights_calib = weights / np.sum(weights)

    # probability-proportional-to-size without replacement sampling using normalized weights
    indices = np.random.choice(np.arange(len(aug_data)), size=len(aug_data), p=weights_calib, replace=True)
    calib_data = aug_data[indices]

    return calib_data

