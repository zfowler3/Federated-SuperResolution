import numpy as np

from Data.dataloader import InlineLoader


def get_dataset_seismic(args):

    train_data = np.load(args.data_path + 'train/train_seismic.npy')
    train_labels = np.load(args.data_path + 'train/train_labels.npy')
    test_seismic = np.load(args.data_path + 'test_once/test2_seismic.npy')
    test2 = np.load(args.data_path + 'test_once/test1_seismic.npy')
    # Normalize seismic
    train_data = (train_data - train_data.min()) / (train_data.max() - train_data.min())
    test_seismic = (test_seismic - test_seismic.min()) / (test_seismic.max() - test_seismic.min())
    test2_s = (test2 - test2.min()) / (test2.max() - test2.min())
    # Create client idxs
    #user_idxs = create_clients(train_data, args.num_users)
    user_idxs = overall_partition(train_data, num_clients=args.num_clients, labels=train_labels)

    return train_data, test_seismic, user_idxs, test2_s

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

def create_clients_rand(data, num_clients):
    client_idxs = {}
    choice_tracker = []
    percentage_missing = [0.2, 0.4, 0.6, 0.8]
    choices = ['inline', 'crossline']
    choices=['crossline']
    for c in range(num_clients):
        p = np.random.choice(percentage_missing, size=1)[0]
        choose = 1 - p
        choice = np.random.choice(choices, size=1)[0]
        if choice == 'inline':
            arr = data.shape[0]
        else:
            arr = data.shape[1]

        cur_idxs = np.random.choice(np.arange(arr), size=int(arr*choose), replace=False)
        client_idxs[c] = cur_idxs
        choice_tracker.append(choice)

    return client_idxs, choice_tracker

def create_local_test(idxs, amount=0.2, save_folder='/home/zoe/GhassanGT Dropbox/Zoe Fowler/Zoe/InSync/BIGandDATA/Seismic/saved_idxs/'):
    n_clients = len(idxs)
    net_dataidx_test = {i: np.array([], dtype="int64") for i in range(n_clients)}
    for i in range(n_clients):
        current_client_idxs = idxs[i]
        test_idxs = np.random.choice(current_client_idxs, size=int(amount*len(current_client_idxs)), replace=False)
        net_dataidx_test[i] = test_idxs.astype(int)
        new_train = np.where(np.isin(current_client_idxs, test_idxs, invert=True))[0]
        idxs[i] = current_client_idxs[new_train].astype(int)

    np.save(save_folder + 'test_idxs.npy', net_dataidx_test, allow_pickle=True)
    np.save(save_folder + 'train_idxs.npy', idxs, allow_pickle=True)
    return net_dataidx_test, idxs

def overall_partition(data, num_clients, labels):
    local_loaders = {
        i: {"datasize": 0, "train": None, "test": None, "test_size": 0} for i in range(num_clients)
    }
    client_idxs, _ = create_clients_rand(data, num_clients)
    test, train = create_local_test(idxs=client_idxs)
    for client_idx, dataidxs in train.items():
        local_loaders[client_idx]["datasize"] = len(dataidxs)
        local_loaders[client_idx]["train"] = InlineLoader(seismic_cube=data, label_cube=labels, inline_inds=dataidxs)
    for client_idx, dataidxs in test.items():
        local_loaders[client_idx]["test"] = InlineLoader(seismic_cube=data, label_cube=labels, inline_inds=dataidxs)
        local_loaders[client_idx]["test_size"] = len(dataidxs)
    print('DataLoaders Complete')
    print(local_loaders)
    return local_loaders