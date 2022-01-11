import os
from os import listdir
from os.path import isfile, join, splitext
import numpy as np
import random

seed = 0

np.random.seed(seed)

class Dataset_create():
    def __init__(self,fold=20,seq_len=20):
        self.X_person = []
        self.y_person = []
        self.X_1epoch = []
        self.y_1epoch = []
        self.X_seq = []
        self.y_seq = []

        self.idx_1epoch = []
        self.idx_seq = []

        self.seq_len = seq_len
        self.fold = fold

    def read_all_data_per_person(self, mypath="E:/EEG/deepsleepnet-master3/data/eeg_fpz_cz"):
        file_list = [f for f in listdir(mypath) if isfile(join(mypath, f))]

        data_X, data_y = [], []
        for i in range(len(file_list)):  # len(file_list)
            with np.load(join(mypath, file_list[i])) as npz:
                data_X.append(npz['x'])
                data_y.append(npz['y'])
        self.X_person = data_X
        self.y_person = data_y

    def make_dataset(self):
        lost_epoch = 0

        for i in range(len(self.X_person)):  # i对应病人序号len(data_X)
            xtemp = np.array(self.X_person[i])
            ytemp = np.array(self.y_person[i]).reshape(len(self.y_person[i]),1)
            # 先把全散装的做出来
            if i==0:
                self.X_1epoch = xtemp
                self.y_1epoch = ytemp
            else:
                self.X_1epoch = np.concatenate([self.X_1epoch, xtemp], axis=0)
                self.y_1epoch = np.concatenate([self.y_1epoch, ytemp], axis=0)

            #再把seq_len的做出来
            for j in range(0,len(xtemp), self.seq_len):
                if j + self.seq_len <= len(xtemp):
                    self.X_seq.append(xtemp[j:j + self.seq_len])
                    self.y_seq.append(ytemp[j:j + self.seq_len])
            xtemp, ytemp = [], []

        self.X_1epoch = np.array(self.X_1epoch)
        self.y_1epoch = np.array(self.y_1epoch)
        self.X_seq = np.array(self.X_seq)
        self.y_seq = np.array(self.y_seq)

    def rotate_data(self,sample_num):
        # 输入数据集第一维的长度，进行20-fold的循环
        # 输出打乱后的20fold序列
        idx = [i for i in range(sample_num)]
        np.random.shuffle(idx)
        idx_rand = []
        for i in range(self.fold):
            move = int(1.0/self.fold*sample_num*i)
            l = idx[-move:]+idx[:-move]
            idx_rand.append(l)

        return idx_rand

    def make_fold_idx(self):
        self.idx_1epoch = self.rotate_data(len(self.X_1epoch))
        self.idx_seq = self.rotate_data(len(self.X_seq))

    def printer(self):
        print('X_1epoch:', np.shape(self.X_1epoch))
        print('y_1epoch:', np.shape(self.y_1epoch))
        print('X_seq:', np.shape(self.X_seq))
        print('y_seq:', np.shape(self.y_seq))
        print('idx_1epoch:', np.shape(self.idx_1epoch))
        print('idx_seq:', np.shape(self.idx_seq))

    def create_datasets(self,path='./data'):
        print('collecting data from seprate patinets')
        self.read_all_data_per_person()
        print('\nfinished collecting\n\nmaking datasets')
        self.make_dataset()
        print('\ndatasets have been made\n\nmaking fold idx orders')
        self.make_fold_idx()
        print('\nidx orders have been made\n\nstart to save data')

        if os.path.exists(path) is False:
            os.mkdir(path)

        with open(join(path, 'util_data' + '.npz'), 'wb') as f:
            np.savez(
                f,
                X_1epoch=self.X_1epoch,
                y_1epoch=self.y_1epoch,
                X_seq=self.X_seq,
                y_seq=self.y_seq,
                idx_1epoch=self.idx_1epoch,
                idx_seq=self.idx_seq
            )
        print('\nall data has been saved')


if __name__ == "__main__":
    d = Dataset_create(fold=20,seq_len=20)
    d.create_datasets()
    d.printer()

# X_1epoch: (194655, 3000, 1)
# y_1epoch: (194655, 1)
# X_seq: (9669, 20, 3000, 1)
# y_seq: (9669, 20, 1)
# idx_1epoch: (20, 194655)
# idx_seq: (20, 9669)
