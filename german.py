import numpy as np
import pandas as pd
from datasets import *
from netcost import *
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader
import torch.optim as optim
import pickle
from costcla.metrics import cost_loss
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

accuracyv = []
topaccuracy = []
lossv = []
trainloss = []

save_fold = './german/BD/'
traindf = pd.read_hdf('./german/train_h5.h5', 'df')
testdf = pd.read_hdf('./german/test_h5.h5', 'df')
pd = traindf.append(testdf).reset_index()
m = pd.amount.max()
pd.amount = [x/m for x in list(pd.amount)]
mc10, mc01 = pd.c10.max(), pd.c01.max()


num = 0
for num in range(0, 5):
    msk = np.random.rand(len(pd)) < 0.8

    traindf = pd[msk].reset_index()
    testdf = pd[~msk].reset_index()

    c10train, c10test = list(traindf['c10']), list(testdf['c10'])
    c01train, c01test = list(traindf['c01']), list(testdf['c01'])

    traincost_matrix = [np.matrix([[0, x/mc10], [y/mc01, 0]]) for x, y in zip(c10train, c01train)]
    testcost_matrix = [np.matrix([[0, x/mc10], [y/mc01, 0]]) for x, y in zip(c10test, c01test)]
    traincost_matrix2 = [np.matrix([[0, 0.2], [1, 0]]) for x in c10train]
    testcost_matrix2 = [np.matrix([[0, 0.2], [1, 0]]) for x in c10test]

    train_dataset = German_cred(traindf['x'], traindf['label'], traincost_matrix2, list(traindf['amount']), traincost_matrix, bregman=True)
    test_dataset = German_cred(testdf['x'], testdf['label'], testcost_matrix2, list(testdf['amount']), testcost_matrix, bregman=True)
    train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=True)

    srt= np.sort(np.asarray(list(testdf['amount'])))
    topten = srt[int(np.ceil(srt.shape[0]*0.9))]
    toptendf = testdf.loc[testdf['amount'] > topten]

    lossv = []
    topaccuracy = []
    accuracyv = []
    trainloss = []
    toptenfeats = []

    net = Mlp(list(traindf['x'])[0].shape[0])
    #cost = cross_entropy()
    #cost = instance_class_weight_ce()
    cost = R_norm(2.2)
    #cost2 = instance_class_weight_ce()
    optimizer = optim.SGD(net.parameters(), lr=0.00001)
    for epoch in range(0, 200):
        epochloss = 0
        for i, sample in enumerate(train_dataloader):
            x_ = Variable(sample['x'].type(torch.FloatTensor))
            y_ = Variable(sample['y'].type(torch.LongTensor))
            c = Variable(sample['c'].type(torch.FloatTensor))
            c_mat = Variable(sample['c_mat'].type(torch.FloatTensor))
            outputs = net(x_, training=True)
            #loss = torch.mean(cost2(outputs[0], y_, c_mat))
            #loss = cost(outputs[0], y_)
            loss = cost(outputs[1], y_, c, c_mat)
            epochloss += loss.data[0]
            loss.backward()
            optimizer.step()

        vloss = 0
        toptenfeat2 = 0
        total = 0
        correct = 0
        predicted_top, y_val_top = [], []
        for k, vsample in enumerate(test_dataloader):
            x_val = Variable(vsample['x'].type(torch.FloatTensor))
            y_val = vsample['y']
            total += y_val.size(0)
            c_val = Variable(vsample['c'].type(torch.FloatTensor))
            c_mat_val = Variable(vsample['c_mat'].type(torch.FloatTensor))
            c_mat_val_e = Variable(vsample['c_mat_e'].type(torch.FloatTensor))
            w_val = vsample['am']
            outputs = net(x_val, training=False)
            temp = outputs[1]
            _, predicted = torch.max(temp.data, 1)
            for j in range(0, y_val.size(0)):
                if w_val[j] > topten:
                    y_val_top.append(y_val[j])
                    predicted_top.append(predicted[j])
                if y_val[j] != predicted[j]:
                    toptenfeat2 += c_mat_val_e[j][predicted[j], y_val[j]]
                else:
                    correct += 1
        tscore = accuracy_score(y_val_top, predicted_top)
        score = correct/total
        accuracyv.append(score)
        toptenfeats.append(toptenfeat2)
        topaccuracy.append(tscore)
        trainloss.append(epochloss)
    print('[%d, accuracy: %.3f topaccuracy: %.3f testcost: %.3f' %
          (epoch + 1, score, tscore, toptenfeat2))
    np.save(save_fold+'{}toptenfeats.npy'.format(num), toptenfeats)
    np.save(save_fold+'{}topaccuracy.npy'.format(num), topaccuracy)
    np.save(save_fold+'{}accuracy.npy'.format(num), accuracyv)
    np.save(save_fold+'{}trainloss.npy'.format(num), trainloss)
    torch.save(net.state_dict(),save_fold+ '{}net.pt'.format(num))
    num += 1
    del net
