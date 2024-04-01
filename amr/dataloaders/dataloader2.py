import torch.utils.data as data
import torch
import pickle
import numpy as np
import h5py
import os
from torch.utils.data import DataLoader
from .transform import *
import random

__all__ = ['AMRDataLoader']

class PreFetcher:
    r""" Data pre-fetcher to accelerate the data loading
    """

    def __init__(self, loader):
        self.ori_loader = loader
        self.len = len(loader)
        self.stream = torch.cuda.Stream()
        self.next_input = None

    def preload(self):
        try:
            self.next_input = next(self.loader)
        except StopIteration:
            self.next_input = None
            return

        with torch.cuda.stream(self.stream):
            for idx, tensor in enumerate(self.next_input):
                self.next_input[idx] = tensor.cuda(non_blocking=True)

    def __len__(self):
        return self.len

    def __iter__(self):
        self.loader = iter(self.ori_loader)
        self.preload()
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        if input is None:
            raise StopIteration
        for tensor in input:
            tensor.record_stream(torch.cuda.current_stream())
        self.preload()
        return input


class AMRDataset(data.Dataset):
    def __init__(self, X, Y, Z, method):
        super(AMRDataset, self).__init__()
        self.X = torch.as_tensor(X)
        self.Y = torch.as_tensor(Y)
        self.Z = torch.as_tensor(Z)

    def __getitem__(self, index):
        return self.X[index], self.Y[index], self.Z[index]

    def __len__(self):
        return self.X.shape[0]


"""
重构思想：
1. AMRDataLoader用于从文件中读取数据并分为训练集：测试集：验证集=2:1:1。 可以通过mod_type选取需要的调试方式，可以全选，也可以挑选子集。
涉及到星座图生成可能比较慢，在这里还要选择是信息流还是星座图，如果是星座图的话，检查是否生成过星座图（方便下次载入），否则执行生成星座图的过程。
RML2016和RML2018的读取可以分开为两个，也可以合并为一个AMRDataLoader，合并为一个DataLoader用参数控制。
2. 读取到的数据在载入AMRDataset的时候，通过载入模式选择不同的操作，包括选取的数据是IQ还是AP，是否归一化，是否zero_mask等等定制化操作。
Xmode json结构定制:

3. 将返回的AMRDataset格式载入DataLoader形成不同的训练、测试、验证集
"""



class AMRDataLoader(object):
    """
    dataset: {RML2016,RML2018}
    Xmode: 定制化数据载入风格
        dict{
            type:{'IQ','AP','IQ_and_AP','star'} 载入数据类型；IQ->AP使用原始IQ信号
            options:{
                IQ_norm: bool IQ数据是否归一化到[0,1]
                zero_mask: bool 是否进行掩码操作以进行数据增强
            }
            options具有可扩展性
        }
    batch_size:
    num_workers:
    pin_memory:
    mod_type:挑选调试方式子集，若全选则为全集
    """
    def __init__(self, dataset, Xmode, batch_size, num_workers, pin_memory, ddp, random_mix, mod_type=[]):
        self.Random_Matrix = None
        # 预处理星座图
        if Xmode["type"] == 'star':
            # 生成星座图
            if not os.path.exists('dataset/star_'+dataset):
                generate_star(dataset, Xmode["options"])
            # 读取星座图数据
            if dataset == 'RML2016':
                pass
            elif dataset == 'RML2018':
                pass

        # 信号流处理
        else:
            if dataset == 'RML2016':
                Xd = pickle.load(open('./dataset/RML2016.10a_dict.pkl', 'rb'), encoding='latin')
                snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
                self.snrs = snrs
                self.mods = mod_type
                mods = mod_type
                print(mods)
                X = []
                lbl = []
                for mod in mod_type:
                    for snr in snrs:
                        X.append(Xd[(mod, snr)])
                        for i in range(Xd[(mod, snr)].shape[0]):  lbl.append((mod, snr))

                X = np.vstack(X)

                if Xmode.type == 'AP':
                    X = get_amp_phase(X)
                    print("AP = True")
                if Xmode.options.IQ_norm == True:
                    X = get_iq_framed(X)
                    print("IQ_norm = True")

                '''
                for i in range(len(mods)):
                    plt.figure()
                    plt.plot(X[1000*20*(i+1)-1,0,:])
                    plt.plot(X[1000*20*(i+1)-1,1,:])
                    plt.savefig("2016/" + mods[i] + ".png")
                '''
                    
                if random_mix:
                    N_random_sample = 50
                    self.Random_Matrix = np.zeros([len(mod_type)*len(snrs),2,128*N_random_sample])
                    count = 0
                    for i in range(len(mod_type)):
                        for snr_idx in range(len(snrs)):
                            choice = np.random.choice(range(1000), size=N_random_sample, replace=False)
                            random_sample = X[choice + i*len(snrs)*1000 + snr_idx*1000]
                            random_sample = random_sample.swapaxes(0,1)
                            random_sample = np.reshape(random_sample, [2,128*N_random_sample])
                            self.Random_Matrix[count] = random_sample
                            count = count + 1

                n_examples = X.shape[0]
                n_train = int(0.6 * n_examples)
                n_valid = int(0.2 * n_examples)

                allnum = list(range(0, n_examples))
                random.shuffle(allnum)
                #allnum = np.loadtxt('./dataset/shuffle_2016.txt', dtype='int', delimiter = ' ')
                
                train_idx = allnum[0:n_train]
                valid_idx = allnum[n_train:n_train + n_valid]
                test_idx = allnum[n_train + n_valid:]

                X_train = X[train_idx]
                Y_train = list(map(lambda x: mods.index(lbl[x][0]), train_idx))
                Z_train = list(map(lambda x: lbl[x][1], train_idx))

                X_valid = X[valid_idx]
                Y_valid = list(map(lambda x: mods.index(lbl[x][0]), valid_idx))
                Z_valid = list(map(lambda x: lbl[x][1], valid_idx))

                X_test = X[test_idx]
                Y_test = list(map(lambda x: mods.index(lbl[x][0]), test_idx))
                Z_test = list(map(lambda x: lbl[x][1], test_idx))

                print(max(Z_test))
            elif dataset == 'RML2018':
                mods = ['32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK', 'OQPSK', '8ASK', 'BPSK', '8PSK',
                        'AM-SSB-SC',
                        '4ASK', '16PSK', '64APSK', '128QAM', '128APSK', 'AM-DSB-SC', 'AM-SSB-WC', '64QAM', 'QPSK',
                        '256QAM',
                        'AM-DSB-WC', 'OOK', '16QAM']
                with h5py.File('./dataset/RML2018.01.hdf5', 'r+') as h5file:
                    '''
                    allX = np.asarray(h5file['X'][:])
                    allY = np.asarray([mods[np.argmax(i)] for i in h5file['Y'][:]])
                    allZ = np.asarray(h5file['Z'][:])
                    '''
                    X = np.asarray(h5file['X'][:])
                    Y = np.asarray([np.argmax(i) for i in h5file['Y'][:]])
                    Z = np.asarray(h5file['Z'][:])

                '''
                X = []
                Y = []
                Z = []
                for idx in range(allX.shape[0]):
                    if allY[idx] in mod_type:
                        X.append(allX[idx])
                        Y.append(mod_type.index(allY[idx]))
                        Z.append(allZ[idx])
                del allX
                del allY
                del allZ
                '''
                X = np.asarray(X)
                X = np.moveaxis(X, 1, 2)
                Y = np.asarray(Y)
                Z = np.asarray(Z)
                Z = np.squeeze(Z)

                if Xmode.type == 'AP':
                    X[0:int(X.shape[0]/2),:,:] = get_amp_phase(X[0:int(X.shape[0]/2),:,:])
                    X[int(X.shape[0]/2):,:,:] = get_amp_phase(X[int(X.shape[0]/2):,:,:])
                    print("APX!")
                if Xmode.options.IQ_norm == True:
                    X = normalize_IQ(X)
                    print("IQ_norm = True")

                '''
                for i in range(len(mods)):
                    plt.figure()
                    plt.plot(X[4096*26*(i+1)-1,0,:])
                    plt.plot(X[4096*26*(i+1)-1,1,:])
                    plt.savefig("2018/" + mods[i] + ".png")
                '''
                self.snrs = np.unique(Z).tolist()
                self.mods = mod_type

                if random_mix:
                    N_random_sample = 50
                    self.Random_Matrix = np.zeros([len(mod_type)*len(self.snrs),2,X.shape[2]*N_random_sample])
                    count = 0
                    for i in range(len(mod_type)):
                        for snr_idx in range(len(self.snrs)):
                            choice = np.random.choice(range(4096), size=N_random_sample, replace=False)
                            random_sample = X[choice + i*len(self.snrs)*4096 + snr_idx*4096]
                            random_sample = random_sample.swapaxes(0,1)
                            random_sample = np.reshape(random_sample, [2,X.shape[2]*N_random_sample])
                            self.Random_Matrix[count] = random_sample
                            count = count + 1

                n_examples = X.shape[0]
                n_train = int(0.6 * n_examples)
                n_valid = int(0.2 * n_examples)
                
                allnum = list(range(0, n_examples))
                random.shuffle(allnum)

                train_idx = allnum[0:n_train]
                valid_idx = allnum[n_train:n_train + n_valid]
                test_idx = allnum[n_train + n_valid:]
                X_train = X[train_idx]
                Y_train = Y[train_idx]
                Z_train = Z[train_idx]
                X_valid = X[valid_idx]
                Y_valid = Y[valid_idx]
                Z_valid = Z[valid_idx]
                X_test = X[test_idx]
                Y_test = Y[test_idx]
                Z_test = Z[test_idx]
                print(max(Z_test))
                del X
                del Y
                del Z

            elif dataset == 'cfo':
                mods = ["BPSK", "QPSK", "8PSK", "PAM4", "QAM16", "QAM64", "GFSK", "WBFM", "AM-DSB", "AM-SSB"]
                with h5py.File('./dataset/cfo.hdf5', 'r+') as h5file:
                    allX = np.asarray(h5file['X'][:])
                    allY = np.asarray([mods[i] for i in h5file['mod'][:]])
                    allZ = np.asarray(h5file['snr'][:])

                X = []
                Y = []
                Z = []
                for idx in range(allX.shape[0]):
                    if allY[idx] in mod_type:
                        X.append(allX[idx])
                        Y.append(mod_type.index(allY[idx]))
                        Z.append(allZ[idx])
                del allX
                del allY
                del allZ
                X = np.asarray(X)
                X = np.moveaxis(X, 1, 2)
                Y = np.asarray(Y)
                Z = np.asarray(Z)

                self.snrs = np.unique(Z).tolist()
                self.mods = mod_type

                n_examples = X.shape[0]
                n_train = int(0.6 * n_examples)
                n_valid = int(0.2 * n_examples)
                
                allnum = list(range(0, n_examples))
                random.shuffle(allnum)

                train_idx = allnum[0:n_train]
                valid_idx = allnum[n_train:n_train + n_valid]
                test_idx = allnum[n_train + n_valid:]
                X_train = X[train_idx]
                Y_train = Y[train_idx]
                Z_train = Z[train_idx]
                X_valid = X[valid_idx]
                Y_valid = Y[valid_idx]
                Z_valid = Z[valid_idx]
                X_test = X[test_idx]
                Y_test = Y[test_idx]
                Z_test = Z[test_idx]
                print(max(Z_test))
                del X
                del Y
                del Z
                if Xmode == 'AP':
                    X_train = get_amp_phase(X_train)
                    X_valid = get_amp_phase(X_valid)
                    X_test = get_amp_phase(X_test)
                    print("APX!")
                if Xmode.options.IQ_norm == True:
                    X_train = normalize_IQ(X_train)
                    X_valid = normalize_IQ(X_valid)
                    X_test = normalize_IQ(X_test)
                    print("IQ_norm = True")


        train_dataset = AMRDataset(X_train, Y_train, Z_train, Xmode)
        valid_dataset = AMRDataset(X_valid, Y_valid, Z_valid, Xmode)
        test_dataset = AMRDataset(X_test, Y_test, Z_test, Xmode)

        if ddp == True:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,shuffle=True,)
            valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset,shuffle=False,)
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset,shuffle=False,)
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=pin_memory, sampler=train_sampler,)
            self.valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=pin_memory, sampler=valid_sampler,)
            self.test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,
                                    pin_memory=pin_memory, sampler=test_sampler,)
        else:
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=pin_memory, shuffle=True)
            self.valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=pin_memory, shuffle=False)
            self.test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,
                                    pin_memory=pin_memory, shuffle=False)

        if pin_memory is True:
            self.train_loader = PreFetcher(self.train_loader)
            self.valid_loader = PreFetcher(self.valid_loader)
            self.test_loader = PreFetcher(self.test_loader)

    def __call__(self):
        return self.train_loader, self.valid_loader, self.test_loader, self.snrs, self.mods, self.Random_Matrix


if __name__ == '__main__':
    train_loader, valid_loader, test_loader = AMRDataLoader("RML2016.10a_dict.pkl","ResNet",batch_size=100,num_workers=4, device='cpu', pin_memory=False)()
    print(train_loader)
    print(valid_loader)
    print(test_loader)
