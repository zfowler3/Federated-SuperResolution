import numpy as np

def get_dataset_seismic(args):

    train_data = np.load('/home/zoe/data/train/train_seismic.npy')
    test_seismic = np.load('/home/zoe/data/test_once/test2_seismic.npy')
    test2 = np.load('/home/zoe/data/test_once/test1_seismic.npy')
    # Normalize seismic
    train_data = (train_data - train_data.min()) / (train_data.max() - train_data.min())
    test_seismic = (test_seismic - test_seismic.min()) / (test_seismic.max() - test_seismic.min())
    test2_s = (test2 - test2.min()) / (test2.max() - test2.min())
    # Create client idxs
    user_idxs = create_clients(train_data, args.num_users)

    return train_data, test_seismic, user_idxs, test2

def create_clients(data, num_clients):
    # Clients will be divided in seismic based on crosslines
    num_crosslines = data.shape[1]
    num_crosslines_per_client = int(num_crosslines/num_clients)
    client_idxs = {}
    start = 0
    end = num_crosslines_per_client
    for c in range(num_clients):
        current_idxs = np.arange(start, end)
        start = end
        end += num_crosslines_per_client
        client_idxs[c] = current_idxs

    return client_idxs