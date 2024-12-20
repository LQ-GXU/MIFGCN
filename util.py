import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
import gongju
from torch.utils.data import TensorDataset, DataLoader
# import matplotlib.pyplot as plt


def bactch_asym_adj(adj):
    num = 0
    adj = adj.cpu()
    for i in adj:
        i = i.detach().numpy()
        i = sp.coo_matrix(i)
        rowsum = np.array(i.sum(1)).flatten()
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        nadj = d_mat.dot(i).astype(np.float32).todense()
        nadj = torch.tensor(nadj)
        if num == 0:
            dadj = nadj
            num = num + 1
        else:
            dadj = torch.cat((dadj, nadj), 0)
    dadj = dadj.reshape(adj.shape[0], adj.shape[1],
                        adj.shape[1]).to(device="cuda:0")
    return dadj


class DataLoader(object):
    def __init__(self,  xs3, ys, batch_size, pad_with_last_sample=True):
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            # num_padding1 = (batch_size - (len(xs1) % batch_size)) % batch_size
            # num_padding2 = (batch_size - (len(xs2) % batch_size)) % batch_size
            num_padding3 = (batch_size - (len(xs3) % batch_size)) % batch_size
            # x_padding1 = np.repeat(xs1[-1:], num_padding1, axis=0)
            # x_padding2 = np.repeat(xs2[-1:], num_padding2, axis=0)
            x_padding3 = np.repeat(xs3[-1:], num_padding3, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding3, axis=0)
            # xs1 = np.concatenate([xs1, x_padding1], axis=0)
            # xs2 = np.concatenate([xs2, x_padding2], axis=0)
            xs3 = np.concatenate([xs3, x_padding3], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs3)
        self.num_batch = int(self.size // self.batch_size)
        # self.xs1 = xs1
        # self.xs2 = xs2
        self.xs3 = xs3
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs3, ys = elf.xs3[permutation], self.ys[permutation]
        # self.xs1 = xs1
        # self.xs2 = xs2
        self.xs3 = xs3
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size *
                              (self.current_ind + 1))
                # x_i1 = self.xs1[start_ind: end_ind, ...]
                # x_i2 = self.xs2[start_ind: end_ind, ...]
                x_i3 = self.xs3[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i3, y_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def sym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj):
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(
        adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_adj(pkl_filename, adjtype):
    adj_mx = load_pickle(pkl_filename)
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(
            adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return adj


# def load_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None):
#
#     data = {}  # 包含x_train,y_train,x_val...的字典
#     for category in ['train', 'val', 'test']:
#         cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
#         data['x_' + category] = cat_data['x']
#         data['y_' + category] = cat_data['y']
#     scaler = StandardScaler(data['x_train'][..., 0].mean(), data['x_train'][..., 0].std())
#
#
#     for category in ['train', 'val', 'test']:
#         data['x_' + category][...,
#                           0] = scaler.transform(data['x_' + category][..., 0])  # 依然是（10700,12,170,1）
#
#     # 对顺序出现的数据全局随机打乱
#     print("Perform shuffle on the dataset")
#
#     random_train = torch.arange(int(data['x_train'].shape[0]))  # (10700,)
#     random_train = torch.randperm(random_train.size(0))
#     data['x_train'] =  data['x_train'][random_train,...]
#     data['y_train'] =  data['y_train'][random_train,...]
#
#     random_val = torch.arange(int(data['x_val'].shape[0]))  # (3566,)
#     random_val = torch.randperm(random_val.size(0))
#     data['x_val'] =  data['x_val'][random_val,...]
#     data['y_val'] =  data['y_val'][random_val,...]
#
#     random_test = torch.arange(int(data['x_test'].shape[0]))
#     random_test = torch.randperm(random_test.size(0))
#     data['x_test'] =  data['x_test'][random_test,...]
#     data['y_test'] =  data['y_test'][random_test,...]
#
#     data['train_loader'] = DataLoader(
#         data['x_train'], data['y_train'], batch_size)
#     data['val_loader'] = DataLoader(
#         data['x_val'], data['y_val'], valid_batch_size)
#     data['test_loader'] = DataLoader(
#         data['x_test'], data['y_test'], test_batch_size)
#     data['scaler'] = scaler

def load_dataset(all_data,all_weather, batch_size, valid_batch_size, test_batch_size,num_week, num_day, num_hour, points_per_hour):
    data = {}

    train_x = all_data['train']['recent']
    val_x = all_data['val']['recent']
    test_x = all_data['test']['recent']
    stats, train_x, val_x, test_x = gongju.normalization(train_x, val_x, test_x)

    train_x1 = all_weather['train']['recent']
    val_x1 = all_weather['val']['recent']
    test_x1 = all_weather['test']['recent']
    list = gongju.normalize_weather([train_x1, val_x1, test_x1])
    train_x1, val_x1, test_x1 = list[0], list[1], list[2]
    # train_dataset = TensorDataset(torch.from_numpy(train_x.transpose(0,3,2,1)).type(torch.FloatTensor).to('cuda:0'), torch.from_numpy(all_data['train']['target'].transpose(0,3,2,1)).type(torch.FloatTensor).to('cuda:0'))
    # val_dataset = TensorDataset(torch.from_numpy(val_x.transpose(0,3,2,1)).type(torch.FloatTensor).to('cuda:0'), torch.from_numpy(all_data['val']['target'].transpose(0,3,2,1)).type(torch.FloatTensor).to('cuda:0'))
    # test_dataset = TensorDataset(torch.from_numpy(val_x.transpose(0,3,2,1)).type(torch.FloatTensor).to('cuda:0'), torch.from_numpy(all_data['test']['target'].transpose(0,3,2,1)).type(torch.FloatTensor).to('cuda:0'))
    # a = all_data['test']['target'][288:575,0,0,0]
    # plt.plot(a)
    # plt.show()
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)




    # all_data['train']['week'] = np.concatenate((train_x[:, :num_week * points_per_hour, :, :], train_x1[:, :num_week * points_per_hour, :, :]), axis=-1)
    # all_data['val']['week'] = np.concatenate((val_x[:, :num_week * points_per_hour, :, :], val_x1[:, :num_week * points_per_hour, :, :]),axis=-1)
    # all_data['test']['week'] = np.concatenate((test_x[:, :num_week * points_per_hour, :, :], test_x1[:, :num_week * points_per_hour, :, :]), axis=-1)
    #
    # all_data['train']['day'] = np.concatenate((train_x[:, num_week * points_per_hour:(num_week * points_per_hour+num_day*points_per_hour), :, :],train_x1[:, num_week * points_per_hour:(num_week * points_per_hour+num_day*points_per_hour), :, :]),axis=-1)
    # all_data['val']['day'] = np.concatenate((val_x[:, num_week * points_per_hour:(num_week * points_per_hour+num_day*points_per_hour), :, :],val_x1[:, num_week * points_per_hour:(num_week * points_per_hour+num_day*points_per_hour), :, :]),axis=-1)
    # all_data['test']['day'] = np.concatenate((test_x[:, num_week * points_per_hour:(num_week * points_per_hour+num_day*points_per_hour), :, :],test_x1[:, num_week * points_per_hour:(num_week * points_per_hour+num_day*points_per_hour), :, :]),axis=-1)

    all_data['train']['recent'] = np.concatenate((train_x,train_x1),axis=-1)
    all_data['val']['recent'] = np.concatenate((val_x,val_x1),axis=-1)
    all_data['test']['recent'] = np.concatenate((test_x,test_x1),axis=-1)



    print("Perform shuffle on the dataset")
    random_train = torch.arange(int(all_data['train']['recent'].shape[0]))
    random_train = torch.randperm(random_train.size(0))
    # all_data['train']['week'] = all_data['train']['week'][random_train, ...]
    # all_data['train']['day'] = all_data['train']['day'][random_train, ...]
    all_data['train']['recent'] = all_data['train']['recent'][random_train, ...]
    all_data['train']['target'] = all_data['train']['target'][random_train, ...]

    random_val = torch.arange(int(all_data['val']['recent'].shape[0]))
    random_val = torch.randperm(random_val.size(0))
    # all_data['val']['week'] = all_data['val']['week'][random_val, ...]
    # all_data['val']['day'] = all_data['val']['day'][random_val, ...]
    all_data['val']['recent'] = all_data['val']['recent'][random_val, ...]
    all_data['val']['target'] = all_data['val']['target'][random_val, ...]

    random_test = torch.arange(int(all_data['test']['recent'].shape[0]))
    random_test = torch.randperm(random_test.size(0))
    # all_data['test']['week'] = all_data['test']['week'][random_test, ...]
    # all_data['test']['day'] = all_data['test']['day'][random_test, ...]
    all_data['test']['recent'] = all_data['test']['recent'][random_test, ...]
    all_data['test']['target'] = all_data['test']['target'][random_test, ...]

    indice = torch.argsort(random_test)


    data['train_loader'] = DataLoader(

        all_data['train']['recent'], all_data['train']['target'], batch_size)
    data['val_loader'] = DataLoader(

        all_data['val']['recent'], all_data['val']['target'], batch_size)
    data['test_loader'] = DataLoader(

        all_data['test']['recent'], all_data['test']['target'], batch_size)

    

    data['y_test'] = all_data['test']['target']
    data['mean'], data['std'] = stats['mean'][0][0][0][0], stats['std'][0][0][0][0]

    return data, indice


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    #rmse = torch.square(torch.sub(preds, labels)).float()
    #rmse = torch.nan_to_num(rmse*mask)
    #return torch.mean(rmse)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)



# def masked_rmse(preds,lables,null_val=np.nan):
#     return torch.sqrt(torch.mean((preds-lables)**2)).to(torch.float64)
#
# def masked_mae(preds,lables,null_val=np.nan):
#     return torch.mean(torch.abs(preds-lables)).to(torch.float64)



def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    #mae = torch.abs(torch.sub(preds, labels)).float()
    #mae = torch.nan_to_num(mae*mask)
    #return torch.mean(mae)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


#def masked_mape(preds, labels, null_val=np.nan):
    #if np.isnan(null_val):
        #mask = ~torch.isnan(labels)
    #else:
      #  mask = (labels != null_val)
    #mask = mask.float()
    #mask /= torch.mean((mask))
    #mape = torch.abs(torch.divide(torch.sub(preds, labels).float(), labels))
    #mape = torch.nan_to_num(mape*mask)
    #return torch.mean(mape)

     #mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
     #loss = torch.abs(preds - labels) / labels
     #loss = loss * mask
     #loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
     #return torch.mean(loss)

def masked_mape(preds, labels, null_val=np.nan):
    mape = (torch.abs(preds-labels)/torch.abs(labels)+1e-5).to(torch.float64)
    mape = torch.where(mape > 5, 0, mape)
    return torch.mean(mape)

def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse
