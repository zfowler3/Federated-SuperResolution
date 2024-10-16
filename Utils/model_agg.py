import copy
import torch

def AvgWeights(weights):
    """ All clients are weighted equally """
    # For this implementation nk = 1
    w_avg = copy.deepcopy(weights[0])
    for k in w_avg.keys():
        for i in range(1, len(weights)):
            w_avg[k] += weights[i][k]
        w_avg[k] = torch.div(w_avg[k], len(weights)) # len(weights) is N ( total number of used clients )
    return w_avg


def Sample_Weighted(weights, client_data):
    """ Aggregation approach utilized in FedAvg
    Each client is weighted by their sample amounts wrt the overall size of all clients
     For iid: Sample_Weighted == AvgWeights """
    # For this implementation nk = number of samples in particular client
    w_sample_weighted = copy.deepcopy(weights[0])
    for k in w_sample_weighted.keys():
        w_sample_weighted[k] = w_sample_weighted[k] * (client_data[0]/sum(client_data))

    for k in w_sample_weighted.keys():
        for i in range(1, len(weights)):
            div_term = client_data[i]/sum(client_data)
            w_sample_weighted[k] += (weights[i][k] * div_term)
        #w_sample_weighted[k] = torch.div(w_sample_weighted[k], len(weights))
    return w_sample_weighted