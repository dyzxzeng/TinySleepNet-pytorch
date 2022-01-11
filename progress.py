import torch
import numpy as np
from dataloader import DataLoader_prolong
from networks import ResnetBlock,BasicBlock
import torch.utils.data as Data
from torch.optim import Adam, SGD
import torch.nn as nn
import time
import sys
import random

seed = 231
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)



def training( resnet,
             train_set,
             valid_set,
             max_epoch=200,
             learning_rate=1e-4,
             batch_size=300,
             l2_weight=1e-4,
             decay_step = 100,
             filename='resnet_'):


    train_loader = Data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    optimizer = Adam(resnet.parameters(), lr=learning_rate)

    class_weights = np.array([1, 1.5, 1, 1.2, 1]).reshape(5, 1)
    class_weights = torch.from_numpy(class_weights).type(torch.FloatTensor)
    class_weights = class_weights.to(device)
    CE_loss = torch.nn.CrossEntropyLoss(weight=class_weights)

    resnet = resnet.to(device)
    CE_loss = CE_loss.to(device)

    for p in resnet.parameters():
        p.requires_grad = True

    step_total = len(train_loader)
    for itr in range(max_epoch):
        Training_Loss = 0.0
        Training_Loss_class = 0.0
        Training_Loss_reg = 0.0
        correct_cnt = 0.0
        start = time.time()
        preds, truth = [], []
        sample_num = 0

        if ((itr+1)%decay_step)==0:
            for param_group in optimizer.param_groups:
                ii = (itr+1)//decay_step
                param_group['lr'] = learning_rate * 0.1**ii

        for step,(x,y) in enumerate(train_loader):
            resnet.train()

            curent_sample_size = len(x)
            x = x.to(device)
            y = y.to(device)

            class_out = resnet.forward(x)

            batch_class_loss = CE_loss(class_out, y)
            # 正则loss
            cnn_weights = [parm for name, parm in resnet.named_parameters() if 'conv' in name]
            reg_loss = 0
            for p in cnn_weights:
                reg_loss += torch.sum(p ** 2) / 2
            reg_loss = reg_loss * l2_weight

            batch_loss = batch_class_loss + reg_loss
            # batch_loss = batch_class_loss
            optimizer.zero_grad()
            nn.utils.clip_grad_norm_(resnet.parameters(), max_norm=5.0, norm_type=2)
            batch_loss.backward()
            optimizer.step()

            tmp_preds = np.reshape(np.argmax(class_out.cpu().detach().numpy(), axis=1), (curent_sample_size))
            tmp_truth = np.reshape(y.cpu().detach().numpy(), (curent_sample_size))
            correct_cnt += sum(tmp_truth == tmp_preds)
            batch_acc = sum(tmp_truth == tmp_preds) / curent_sample_size
            sample_num += curent_sample_size

            sys.stdout.write(
                '\r epoch: %d, [step: %d / all %d], batch_loss: %f [class: %f, reg: %f], batch_acc: %f' \
                % (itr + 1, step + 1, step_total, batch_loss.data.cpu().numpy(), batch_class_loss.data.cpu().numpy(),
                   reg_loss.data.cpu().numpy(), batch_acc))
            sys.stdout.flush()

            Training_Loss += float(batch_loss) * curent_sample_size
            Training_Loss_class += float(batch_class_loss) * curent_sample_size
            Training_Loss_reg += float(reg_loss) * curent_sample_size
            del x, y, class_out, cnn_weights, batch_class_loss

        end = time.time()
        usetime = end - start

        train_acc = correct_cnt / sample_num
        Training_Loss /= sample_num
        Training_Loss_reg /= sample_num
        Training_Loss_class /= sample_num

        valid_loader = Data.DataLoader(
            dataset=valid_set,
            batch_size=100,
            shuffle=False
        )

        valid_loss_class, valid_acc_class, pre_label_mtr = \
            test(resnet, valid_loader)

        print(
            '\nEpoch {}/{} use {:.4} sec -------- Train Loss:{:.4} [Class:{:.4}, Reg:{:.4}], Train ACC:{:.4}, Valid Loss:{:.4}, Valid ACC:{:.4}\n'
            .format(itr + 1, max_epoch, usetime, Training_Loss, Training_Loss_class, Training_Loss_reg, train_acc,
                    valid_loss_class, valid_acc_class))
        print(pre_label_mtr, '\n')

        if (itr+1)%10==0:
            torch.save(resnet, filename + 'resnet' + str((itr+1)//10) + '.pkl')
    return resnet


def test(resnet,
         test_loader):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    resnet = resnet.to(device)

    class_weights = np.array([1, 1.5, 1, 1.2, 1]).reshape(5, 1)
    class_weights = torch.from_numpy(class_weights).type(torch.FloatTensor)
    CE_class_loss = torch.nn.CrossEntropyLoss(weight=class_weights)

    class_weights = class_weights.to(device)
    CE_class_loss = CE_class_loss.to(device)

    resnet.eval()

    correct_class = 0
    testing_loss = 0.0
    sample_num = 0
    pre_label_mtr = torch.zeros((5,5),dtype=torch.int32)
    preds, truth = [], []

    for step, (x, y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)
        current_batch_num = len(x)

        class_out = resnet.forward(x)
        batch_loss_class = CE_class_loss(class_out, y)

        batch_loss = batch_loss_class.detach()
        sample_num += current_batch_num
        testing_loss += float(batch_loss)*current_batch_num

        tmp_preds = np.reshape(np.argmax(class_out.cpu().detach().numpy(),axis=1),(current_batch_num))
        tmp_truth = np.reshape(y.cpu().detach().numpy(),(current_batch_num))
        correct_class += sum(tmp_truth == tmp_preds)

        for i in range(current_batch_num):
            pre_label_mtr[tmp_preds[i]][tmp_truth[i]] += 1
        del x, y

    testing_loss /= sample_num
    testing_acc_class = correct_class / sample_num
    return testing_loss, testing_acc_class, pre_label_mtr



def evaluate(resnet,
             testset):
    test_loader = Data.DataLoader(
        dataset=testset,
        batch_size=256
    )
    test_loss, test_acc, test_pred_label_metrics = test(resnet, test_loader)
    #打印validation
    print()
    print('-------------------- Test Result --------------------')
    print('          Test Loss:{:.4} \t Test Acc:{:.4}'.format(test_loss,test_acc))
    print('\tDetails')
    print(test_pred_label_metrics)

if __name__ == '__main__':
    resnet = ResnetBlock(block=BasicBlock, block_stride=[1,2,2,2,2,2,2,2], in_channel=1, out_channel=32)

    print('-------------------- Reading data --------------------')
    d = DataLoader_prolong(fold_idx=0,train_ratio=0.8, test_ration=0.2,valid_ration=0.1)
    d.read_dataset()
    print('\n-------------------- Data split --------------------')
    X_train, y_train, X_test, y_test, X_valid, y_valid = d.split_fintune()

    del d

    print('X_train:', X_train.shape)
    print('y_train:', y_train.shape)
    print('X_test:', X_test.shape)
    print('y_test:', y_test.shape)
    print('X_valid:', X_valid.shape)
    print('y_valid:', y_valid.shape)


    train_set = Data.TensorDataset(X_train, y_train)
    valid_set = Data.TensorDataset(X_valid, y_valid)
    test_set = Data.TensorDataset(X_test, y_test)

    print('\n-------------------- Ready for Together --------------------')


    resnet = training(resnet=resnet, train_set=train_set, valid_set=valid_set, max_epoch=30, learning_rate=1e-3,
                      batch_size=256, l2_weight=1e-4, decay_step=6, filename='resnet_')


    for i in range(6):
        resnet = torch.load('together_resnet' + str(i+1) + '.pkl')
        test_loader = Data.DataLoader(
            dataset=test_set,
            batch_size=256
        )
        evaluate(resnet, test_set)


