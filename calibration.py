# input type is np.array
import numpy as np
# calculate weights_post to update weight
# n is lenght of original data
# m is length of aug_data

# diff_tot calculate total sum of column error
# diff_each calculate each column error
def diff_tot(lamb, weights, aug_data, desired_means, n):
    weights_prev = weights
    weights_post = weights_prev * np.exp(-aug_data @ lamb)
    weights_post = weights_post / np.sum(weights_post) * n

    return (np.sum(abs(aug_data.T @ weights_post /n - desired_means)))

def diff_each(lambda_, weights, aug_data, desired_means, n):
    weights_prev = weights
    weights_post = weights_prev * np.exp(-aug_data * lambda_ )
    weights_post = weights_post / np.sum(weights_post) * n
    
    return ((aug_data.T @ weights_post)/n - desired_means)

# index is order of lambda you want to update


def deriv_lamb_diff(lambda_, weights, aug_data, desired_means, m):
    return(-(aug_data*aug_data) @ np.exp(-lambda_*aug_data).T /m)


# method is newton-Rhapson algorithm
# iter is hyperparameter of how many times you want to update each lambda
def calibration(ori_data,aug_data, iter=15, lr=0.0001):
    n=len(ori_data)
    m=len(aug_data)
    init_weights=np.ones(len(aug_data)) / m * n

    # benchmark information
    desired_means=np.mean(ori_data, axis=0)

    # To find lambda
    # init_lambda
    init_lambda=np.zeros(len(desired_means))
    lamb=init_lambda
    weights=init_weights

    for i in range(0,len(aug_data.T)):
        for j in range(iter):
            eps = diff_each(lamb[i], weights, aug_data[:,i], desired_means[i], n)/deriv_lamb_diff(lamb[i], weights, aug_data[:,i], desired_means[i], m)
            lamb[i] = lamb[i] - lr*eps
            weights_calib=weights*np.exp(-aug_data[:,i]*lamb[i])
            weights = weights_calib/np.sum(weights_calib)*n
            print('each_iter_error')
            print(abs(diff_each(lamb[i], weights,aug_data[:,i], desired_means[i], n)))
        print('tot error')
        print(diff_tot(lamb, weights, aug_data, desired_means, n))

    # compare each row
    rst = np.array([aug_data[np.random.choice(np.arange(100), size=len(aug_data), p=weights_calib, replace=True)].mean(axis=0) for _ in range(100)])
    print("rst:" , rst)
                
    # tot_mean
    row_means_rst = rst.mean(axis=1)
    print("row_means_rst:", row_means_rst)
                    
    # sampling by weights
    weights_calib = weights / np.sum(weights)
    synthetic_data_indices = np.random.choice(np.arange(len(aug_data)), size=len(ori_data), p=weights_calib, replace=True)
    synthetic_data = aug_data[synthetic_data_indices]

    return (synthetic_data)
