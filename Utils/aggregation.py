import numpy as np

def aggregate(preds, psnr, mode='majority'):
    # total number of test imgs
    num_test_img = len(psnr[0]["psnr"])

    for i in range(num_test_img):
        if mode == 'majority':
            final_pred = majority_vote(preds=preds, img_num=i)
    return

def majority_vote(preds, img_num):
    s = preds[0]["pred"][:, img_num, :].T
    x, y = s.shape[0], s.shape[1]
    combined = np.zeros(shape=(len(preds), x, y))
    for c in range(len(preds)):
        combined[:, c, :] = preds[c]["pred"][:, img_num, :].T

    # For specific image, do a pixel-wise majority vote
    majority_pred = np.equal.outer(combined, np.arange(6)).sum(0).argmax(-1)
    return majority_pred