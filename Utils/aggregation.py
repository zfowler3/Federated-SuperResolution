import numpy as np
from sklearn.metrics import jaccard_score


def aggregate(gt_labels, preds, psnr, mode='majority'):
    # total number of test imgs
    num_test_img = len(psnr[0]["val"])
    p = preds[0]["pred"]
    overall = np.zeros_like(p)

    for i in range(num_test_img):
        if mode == 'majority':
            final_pred = majority_vote(preds=preds, img_num=i)
        elif mode == 'avg':
            final_pred = average_vote(preds=preds, img_num=i)
        else:
            final_pred = weighted_vote(preds=preds, to_weight=psnr, img_num=i)

        overall[:, i, :] = final_pred

    # Get mIoU Scores for predictions
    miou_test = jaccard_score(gt_labels.flatten(), overall.flatten(), labels=list(range(6)), average='weighted')
    miou_test_class = jaccard_score(gt_labels.flatten(), overall.flatten(), labels=list(range(6)), average=None)

    return overall, miou_test, miou_test_class

def majority_vote(preds, img_num):
    s = preds[0]["pred"][:, img_num, :]
    x, y = s.shape[0], s.shape[1]
    combined = np.zeros(shape=(len(preds), x, y))
    for c in range(len(preds)):
        combined[:, c, :] = preds[c]["pred"][:, img_num, :]

    # For specific image, do a pixel-wise majority vote
    majority_pred = np.equal.outer(combined, np.arange(6)).sum(0).argmax(-1)
    return majority_pred

def average_vote(preds, img_num):
    s = preds[0]["pred"][:, img_num, :]
    x, y = s.shape[0], s.shape[1]
    combined = np.zeros(shape=(len(preds), x, y))
    for c in range(len(preds)):
        combined[:, c, :] = preds[c]["pred"][:, img_num, :]

    # For specific image, do pixel-wise average (and round to give integer vals)
    avg_pred = np.round(np.mean(combined, axis=0))
    return avg_pred

def weighted_vote(preds, to_weight, img_num):
    wts_for_img = []
    for c in range(len(preds)):
        wts_for_img.append(to_weight[c]["val"][img_num])
    weights = np.array(wts_for_img) / np.sum(wts_for_img)

    s = preds[0]["pred"][:, img_num, :]
    x, y = s.shape[0], s.shape[1]
    combined = np.zeros(shape=(len(preds), x, y))
    for c in range(len(preds)):
        combined[:, c, :] = preds[c]["pred"][:, img_num, :]
    # Do weighted average, where each image is weighted wrt its produced score
    weighted_pred = np.round(np.average(combined, axis=0, weights=weights))
    return weighted_pred