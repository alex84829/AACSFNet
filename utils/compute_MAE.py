import numpy as np
import os


# Mean error rate   Average Absolute Error (AAE)
def test_MAE(preds, labels):
    preds[preds < 0.0] = 0.0
    preds[preds > 15.0] = 15.0

    MAE = np.mean(np.abs(preds - labels))

    # print('MAE %.5f' % (MAE), end=' ')

    return MAE


def test_MSE(preds, labels):
    preds[preds < 0.0] = 0.0
    preds[preds > 15.0] = 15.0

    MSE = np.mean((preds - labels) ** 2)

    # print('MSE %.5f' % (MSE), end=' ')

    return MSE


def test_PCC(preds, labels):
    preds[preds < 0.0] = 0.0
    preds[preds > 15.0] = 15.0

    preds_hat = np.mean(preds)
    labels_hat = np.mean(labels)

    top = np.sum((preds - preds_hat) * (labels - labels_hat))

    down = np.sum((preds - preds_hat) ** 2) * np.sum((labels - labels_hat) ** 2)
    down = np.sqrt(down)

    if down == 0.0:
        down = 0.00001

    PCC = top / down

    return PCC
