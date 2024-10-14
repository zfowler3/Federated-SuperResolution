import numpy as np

def aggregate(preds, psnr, mode='majority'):
    # total number of test imgs
    num_test_img = len(psnr[0]["psnr"])
    p = preds[0]["pred"]
    overall = np.zeros_like(p)

    for i in range(num_test_img):
        if mode == 'majority':
            final_pred = majority_vote(preds=preds, img_num=i)
        elif mode == 'avg':
            final_pred = average_vote(preds=preds, img_num=i)
        else:
            final_pred = weighted_vote(preds=preds, psnr=psnr, img_num=i)

        overall[:, i, :] = final_pred

    return overall

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
        wts_for_img.append(to_weight[c]["psnr"][img_num])
    weights = np.array(wts_for_img) / np.sum(wts_for_img)

    s = preds[0]["pred"][:, img_num, :]
    x, y = s.shape[0], s.shape[1]
    combined = np.zeros(shape=(len(preds), x, y))
    for c in range(len(preds)):
        combined[:, c, :] = preds[c]["pred"][:, img_num, :]
    # Do weighted average, where each image is weighted wrt its produced score
    weighted_pred = np.round(np.average(combined, axis=0, weights=weights))
    return weighted_pred