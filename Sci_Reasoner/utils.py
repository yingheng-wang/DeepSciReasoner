import numpy as np
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error



def comp_wd(x1, x2):
    bs = x1.shape[0]
    dim = x1.shape[1]
    vec_list = np.arange(dim)
    assert np.isfinite(x1).all()
    assert np.isfinite(x2).all()
    # Making Scipy Results
    result_1=0
    for i in range(bs):
        vec_dist_1 = stats.wasserstein_distance(vec_list, vec_list, x1[i], x2[i])
        result_1 += vec_dist_1
    return result_1*0.0625/bs


def metrics(label_gt, prediction):
    label_sum = np.sum(label_gt, axis=1, keepdims=True)
    pred_sum = np.sum(prediction, axis=1, keepdims=True)
    prediction = prediction / pred_sum
    prediction = prediction * label_sum
    mae, r2, mse, wd = eval_metrics(prediction, label_gt)
    return mae, r2, mse, wd


def eval_metrics(pred, label):
    mae = np.mean(np.abs((pred) - label))
    r2 = r2_score(label, pred, multioutput='variance_weighted')
    mse = mean_squared_error(pred, label)
    wd = comp_wd(pred, label)
    return mae, r2, mse, wd
