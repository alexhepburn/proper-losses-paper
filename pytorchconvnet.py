import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from chromagram import *
from tqdm import tqdm
import pickle
import random
from netcost import *

for p in range(0, 3):
    C_matrix = np.load('C_matrix.npy')
    with open('class_dict.pickle', 'rb') as f:
        class_dict = pickle.load(f)
    # Load in data
    traindf = pd.read_hdf('train{}.h5'.format(p), 'df')
    testdf = pd.read_hdf('test{}.h5'.format(p), 'df')
    m = max([traindf['loglistens'].max(), testdf['loglistens'].max()])
    traindf['loglistens'] = [((x)/m) for x in list(traindf['loglistens'])]
    testdf['loglistens'] = [((x)/m) for x in list(testdf['loglistens'])]
    testloglistens = np.sort(np.array(testdf['loglistens']))

    testlabels= ([class_dict[l] for l in list(testdf['genre'])])
    topten = testloglistens[int(np.ceil(testloglistens.shape[0]*0.90))]
    toptendf = testdf.loc[testdf['loglistens'] > topten]
    ind = np.diag_indices(13)
    C = np.ones((13, 13))
    C[ind] = 0
    trainlabels= ([class_dict[l] for l in list(traindf['genre'])])
    chroma_dataset = Chromagram(traindf['chroma'], trainlabels, traindf['loglistens'],
        C_matrix, transform=None, bregman=True)
    dataloader = DataLoader(chroma_dataset, batch_size=128,
                            shuffle=True)
    valid_dataset = Chromagram(testdf['chroma'], testlabels, testdf['loglistens'],
        C_matrix, transform=None, bregman=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=128,
                            shuffle=True)
    C_matrix = Variable(torch.from_numpy(C_matrix)).cuda()
    C_vector = torch.sum(C_matrix, dim=0)
    convnet = Convnet()
    convnet.cuda()

    #crit2 = instance_class_weight_ce()
    #crit2 = cross_entropy()
    save_fold = './r4/'

    optimizer = optim.Adamax(convnet.parameters(), lr=0.00001, weight_decay=5e-4)
    accuracyv = []
    topaccuracy = []
    lossv = []
    trainlosses = []
    toptenfeats = []
    r=4
    for epoch in range(0, 20):
        trainloss = 0
        crit2 = R_norm(r=r)
        r+=0.0
        for i, sample in enumerate(dataloader):
            x = Variable(sample['x'].cuda())
            y = Variable(sample['y'].type(torch.cuda.LongTensor))
            w = Variable(sample['w'].type(torch.cuda.FloatTensor))
            c = Variable(sample['c'].type(torch.cuda.FloatTensor))
            c_mat = Variable(sample['c_mat'].type(torch.cuda.FloatTensor))
            c_mat_e = Variable(sample['c_mat_e'].type(torch.cuda.FloatTensor))
            optimizer.zero_grad()
            outputs = convnet(x)
            loss = crit2(outputs[1], y, c, c_mat, w)
            #loss = crit2(outputs[0], y)
            #loss = crit2(outputs[0], y, c_mat, w)
            _, predicted = torch.max(outputs[1], 1)
            for j in range(0, y.data.shape[0]):
                trainloss +=c_mat_e[j][predicted.data[j], y.data[j]]
            loss.backward()
            optimizer.step()

        total = 0
        correct = 0
        vloss = 0
        topcorrect = 0
        toptotal = 0
        toptenfeat = 0
        for k , vsample in enumerate(valid_dataloader):
            x_val = Variable(vsample['x'].cuda())
            y_val = vsample['y']
            w_val = Variable(vsample['w'].type(torch.cuda.FloatTensor))
            c_val = Variable(vsample['c'].type(torch.cuda.FloatTensor))
            c_vec = Variable(vsample['c_vec'].type(torch.cuda.FloatTensor))
            c_mat_val = Variable(vsample['c_mat'].type(torch.cuda.FloatTensor))
            c_mat_e_val = vsample['c_mat_e']
            outputs = convnet(x_val)
            temp = outputs[1]
            _, predicted = torch.max(temp.data, 1)
            total += y_val.size(0)
            for j in range(0, y_val.size(0)):
                if w_val.data[j] > topten:
                    toptotal += 1
                if y_val[j] == predicted[j]:
                    correct += 1
                    if w_val.data[j] > topten:
                        topcorrect += 1
                toptenfeat += c_mat_e_val[j][predicted[j], y_val[j]]
        score = correct / total * 100
        tscore = topcorrect / toptotal * 100
        print('[%d,     accuracy: %.3f topaccuracy: %.3f testcost: %.3f traincost: %.3f' %
              (epoch + 1, score, tscore, toptenfeat, trainloss))
        accuracyv.append(score)
        toptenfeats.append(toptenfeat)
        topaccuracy.append(tscore)
        trainlosses.append(trainloss)
        torch.save(convnet.state_dict(),save_fold+'{}{}current.pt'.format(p, epoch))
    np.save(save_fold+'{}topaccuracy.npy'.format(p), topaccuracy)
    np.save(save_fold+'{}accuracy.npy'.format(p), accuracyv)
    np.save(save_fold+'{}trainloss.npy'.format(p), trainloss)
    np.save(save_fold+'{}toptenfeats.npy'.format(p), toptenfeats)

    del convnet
    del optimizer
    del crit2
