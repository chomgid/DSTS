import numpy as np


def num_Newton2(lambda_, weights, aug_data, desired_means, n):
    """
    numerator of Newton's method (f(lambda))
    """
    weights_prev = weights
    weights_post = weights_prev * np.exp(-aug_data * lambda_ )
    weights_post = weights_post / np.sum(weights_post) * n
    return ((aug_data.T @ weights_post)/n - desired_means)


def num_Newton(lambda_, weights, aug_data, desired_means, n):
    """
    numerator of Newton's method (f(lambda))
    """
    weights = weights.squeeze()
    lambda_ = lambda_.squeeze()
    aug_data = aug_data.squeeze()
    desired_means = desired_means.squeeze()
    numerator = n*np.sum(weights * np.exp(-aug_data * lambda_ )*aug_data)
    denominator = np.sum(weights*np.exp(-aug_data * lambda_ ))*n
    return (numerator/denominator - desired_means)


def denom_Newton(lambda_, weights, aug_data, n):
    """
    denominator of Newton's method (f'(lambda))
    """
    weights = weights.squeeze()
    lambda_ = lambda_.squeeze()
    aug_data = aug_data.squeeze()

    A = aug_data.T @ (weights * np.exp(-aug_data * lambda_ ))
    B = np.sum(weights * np.exp(-aug_data * lambda_ ))
    dA = -(aug_data**2).T @ (weights * np.exp(-aug_data * lambda_ ))
    dB = -A
    return n*(dA*B-A*dB)/B**2


# optimized by Newton's method
def calibration(ori_data:np.ndarray, aug_data:np.ndarray, iter, tot_iter, aug):
    """
    Calibrate synthesized data so that cross-sectional data structure is preserved.

    Parameters:
    ori_data (np.ndarray): Input data array of shape (size, length).
    aug_data (np.ndarray): Generated data array of shape (size, length)
    iter (int): how many times Newton-Rhapson update is performed for each timestamp
    tot_iter (float): how many times the whole time series is updated
    aug (int): The multiplier for the size of the synthesized data relative to the original data.

    Returns:
    np.ndarray: Calibrated data array of shape (size * aug, length).

    """
    n=len(ori_data)
    m=len(aug_data)
    init_weights=np.ones(len(aug_data)) / m

    criterion = 1e-5

    # benchmark information
    desired_means=np.mean(ori_data, axis=0)

    # lambda update usig Newton's method (iter*tot_iter times)
    init_lambda=np.zeros(len(desired_means))
    lamb=init_lambda
    weights=init_weights
    for t in range(tot_iter):
        for i in range(0,len(aug_data.T)):
            for j in range(iter):
                num = num_Newton(lamb[i], weights, aug_data[:,i], desired_means[i], n)
                denom = denom_Newton(lamb[i], weights, aug_data[:,i], n)
                if num<criterion:
                    break
                eps = num/denom
                lamb[i] = lamb[i] - eps
            weights_calib=weights*np.exp(-aug_data[:,i]*lamb[i])
            weights = weights_calib/np.sum(weights_calib)*n

            if np.isnan(np.sum(weights)):
                print(f"tot_iter: {t}, iter: {j}, index: {i}")
                raise ValueError("nan found")


    # weights normalization
    weights_calib = weights / np.sum(weights)

    # probability-proportional-to-size without replacement sampling using normalized weights
    indices = np.random.choice(np.arange(len(aug_data)), size=aug*n, p=weights_calib, replace=False)
    calib_data = aug_data[indices]

    return calib_data

