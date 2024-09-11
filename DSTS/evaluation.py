import numpy as np
from Codes.metric_utils import calculate_scores
from Codes.discriminative_metric import discriminative_score_metrics 
from Codes.predictive_metric import predictive_score_metrics
from Codes.context_fid import Context_FID


def disc_score(ori_data, fake_data, iterations=5):
    ori_data = ori_data[:,:, np.newaxis]
    fake_data = fake_data[:,:,np.newaxis]
    # print(f'Total iterations: {iterations}')
    discriminative_score = []
 
    for i in range(iterations):
        temp_disc, _, _ = discriminative_score_metrics(ori_data[:], fake_data[:ori_data.shape[0]])
        discriminative_score.append(temp_disc)
        # print(f'Iter {i}: ', temp_disc, '\n')

    return calculate_scores(discriminative_score)


def pred_score(ori_data, fake_data, iterations=5):
    ori_data = ori_data[:,:, np.newaxis]
    fake_data = fake_data[:,:,np.newaxis]
    # print(f'Total iterations: {iterations}')
    perdictive_score = []

    for i in range(iterations):
        temp_pred = predictive_score_metrics(ori_data[:], fake_data[:ori_data.shape[0]])
        perdictive_score.append(temp_pred)
        # print(f'Iter {i}: ', temp_pred, '\n')

    return calculate_scores(perdictive_score)


def contextFID(ori_data, fake_data, iterations=5):
    ori_data = ori_data[:,:, np.newaxis]
    fake_data = fake_data[:,:,np.newaxis]
    # print(f'Total iterations: {iterations}')
    context_fid_score = []

    for i in range(iterations):
        context_fid = Context_FID(ori_data, fake_data)
        context_fid_score.append(context_fid)
        # print(f'Iter {i}: ', 'context-fid =', context_fid, '\n')
    
    return calculate_scores(context_fid_score)


def dupi_based_score(ori_data, fake_data):
    fake_data = fake_data[:ori_data.shape[0]]
    sum = 0
    for i in range(len(ori_data)):
        data = ori_data[i, :]
        
        ori_data_excl = np.delete(ori_data, i, axis=0) 
        combined = np.vstack((fake_data, ori_data_excl))
        
        # Calculate distances
        distances = np.sum(np.abs(combined - data), axis=1)

        if np.argmin(distances)<len(fake_data):
            sum += 1

    return sum/len(ori_data)


def quantile_score(ori_data, fake_data, quantile_list = [0.1,0.3,0.5,0.7,0.9]):
    fake_data = fake_data[:ori_data.shape[0]]
    ori_list = []
    for i in range(ori_data.shape[1]):
        ori = ori_data[:,i]
        ori_list.append(np.quantile(ori, quantile_list))
    ori_quantile=np.vstack(ori_list)

    fake_list = []
    for i in range(ori_data.shape[1]):
        fake = fake_data[:,i]
        fake_list.append(np.quantile(fake, quantile_list))
    quantiles = np.vstack(fake_list)
    quant_loss = np.average(np.abs(ori_quantile-quantiles), axis=0)

    return quant_loss, np.average(quant_loss)
