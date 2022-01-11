from os.path import isfile, join, splitext
import numpy as np
import torch
import random
seed = 213
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

class DataLoader_prolong():
    def __init__(self,fold_idx,train_ratio=0.9,test_ration=0.1,valid_ration=0.1):
        self.fold_idx = fold_idx
        temp = train_ratio + test_ration
        self.train_ratio = train_ratio / temp
        self.test_ration = test_ration / temp
        self.valid_ration = valid_ration

        self.X_1epoch = None
        self.y_1epoch = None
        self.idx_1epoch = None

    def read_dataset(self,path='./data'):
        npz = np.load(join(path, 'util_data' + '.npz'))
        self.X_1epoch = npz['X_1epoch']
        self.y_1epoch = npz['y_1epoch']
        self.idx_1epoch = npz['idx_1epoch']

    def split_fintune(self):
        idx_choosen = self.idx_1epoch[self.fold_idx,]
        X_1epoch = self.X_1epoch[idx_choosen]
        y_1epoch = self.y_1epoch[idx_choosen]

        sample_num = len(self.X_1epoch)
        train_num = int(self.train_ratio*sample_num)
        test_num = sample_num - train_num
        valid_num = int(train_num*self.valid_ration)
        train_num = train_num - valid_num

        X_1epoch_train = X_1epoch[0:train_num, ]
        y_1epoch_train = y_1epoch[0:train_num, ]

        X_1epoch_test = X_1epoch[train_num:train_num+test_num, ]
        y_1epoch_test = y_1epoch[train_num:train_num+test_num, ]


        X_1epoch_valid = X_1epoch[train_num+test_num:sample_num, ]
        y_1epoch_valid = y_1epoch[train_num+test_num:sample_num, ]

        # train
        X_1epoch_train = torch.from_numpy(X_1epoch_train).type(torch.FloatTensor)
        X_1epoch_train = X_1epoch_train.permute(0, 2, 1)
        y_1epoch_train = y_1epoch_train.reshape(train_num)
        y_1epoch_train = torch.from_numpy(y_1epoch_train).type(torch.long)

        # test
        X_1epoch_test = torch.from_numpy(X_1epoch_test).type(torch.FloatTensor)
        X_1epoch_test = X_1epoch_test.permute(0, 2, 1)
        y_1epoch_test = y_1epoch_test.reshape(test_num)
        y_1epoch_test = torch.from_numpy(y_1epoch_test).type(torch.long)

        # valid
        X_1epoch_valid = torch.from_numpy(X_1epoch_valid).type(torch.FloatTensor)
        X_1epoch_valid = X_1epoch_valid.permute(0, 2, 1)
        y_1epoch_valid = y_1epoch_valid.reshape(valid_num)
        y_1epoch_valid = torch.from_numpy(y_1epoch_valid).type(torch.long)


        return X_1epoch_train,y_1epoch_train,X_1epoch_test, y_1epoch_test,X_1epoch_valid, y_1epoch_valid





if __name__ == "__main__":
    d = DataLoader_prolong(fold_idx=0)
    print('Loading prolong data\n')
    d.read_dataset()
    print('start to split\n')
    X_1epoch_train, y_1epoch_train, X_1epoch_test, y_1epoch_test, X_1epoch_valid, y_1epoch_valid = d.split_fintune()
    print('X_1epoch_train:', X_1epoch_train.shape)
    print('y_1epoch_train:', y_1epoch_train.shape)
    print('X_1epoch_test:', X_1epoch_test.shape)
    print('y_1epoch_test:', y_1epoch_test.shape)
    print('X_1epoch_valid:', X_1epoch_valid.shape)
    print('y_1epoch_valid:', y_1epoch_valid.shape)

    '''
    X_1epoch_train: torch.Size([157547, 1, 6000])
    y_1epoch_train: torch.Size([157547])
    X_1epoch_test: torch.Size([19451, 1, 6000])
    y_1epoch_test: torch.Size([19451])
    X_1epoch_valid: torch.Size([17505, 1, 6000])
    y_1epoch_valid: torch.Size([17505])
    '''
