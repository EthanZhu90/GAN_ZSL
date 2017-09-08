import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.autograd as autograd
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data_utils

import __init__
import pickle
from sklearn import svm
from easydict import EasyDict as edict
import scipy.io as sio
import numpy as np
import argparse
from time import gmtime, strftime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
from tensorflow.examples.tutorials.mnist import input_data

from roi_data_layer.layer import FeatDataLayer
from datasets.factory import get_imdb
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.train_utils import get_testing_text_feature
from fast_rcnn.train_utils import get_training_text_feature
from fast_rcnn.train_utils import get_training_roidb_classify

#from ac_wgangp_text_Z0PureLinear import _netG, weights_init
from ac_wgangp_text import _netG, _netG2, weights_init


z_dim = 100
nSample = 500 # number of fake feature for each class
X_dim = 512
h_dim = 512
y_dim = 50
parser = argparse.ArgumentParser()
parser.add_argument('--gpu' , default='0', type=str, help='index of GPU to use')
opt = parser.parse_args()
opt.netG = 1
print(opt)

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

class _classifier(nn.Module):
    def __init__(self):
        super(_classifier, self).__init__()
        self.main = nn.Sequential(nn.Linear(X_dim, h_dim),
                                  nn.ReLU(True),
                                  nn.Linear(h_dim, y_dim))

    def forward(self, feat):
        output = self.main(feat)
        return output

def train_SVM():

    toTrain = 1
    if toTrain:
        if opt.netG == 1:  # _netG
            # 2  _netG2,  reduce the dim of text first
            netG = _netG().cuda()
            opt.resume = 'out/zsl_wgan_gp_text/D5_98000.tar'
        elif opt.netG == 2:
            netG = _netG2().cuda()
            opt.resume = 'out/zsl_wgan_gp_text_netG2/D5_48000.tar'
        else:
            print("Wrong netG option")


        # opt.resume = 'out/zsl_wgan_gp_text_Z0/D5_46000.tar'
        # opt.resume = 'out/zsl_wgan_gp_text_Z0PureLinear/D5_58000.tar'
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            netG.load_state_dict(checkpoint['state_dict_G'])
            print(checkpoint['log'])
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

            # exp_info = '{}_50way_classifier_Z0'.format(nSample)
        exp_info = '{}_50way_classifier'.format(nSample)
        out_dir = 'out/{:s}'.format(exp_info)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        # if 0: #opt.netG != '':
        #     netG.load_state_dict(torch.load(opt.netG))
        print(netG)
        test_text_feature = get_testing_text_feature()
        print test_text_feature.shape

        gen_feat = np.zeros([0, X_dim])

        for i in range(50):
            text_feat = np.tile(test_text_feature[i].astype('float32'), (nSample, 1))
            # print text_feat.shape
            text_feat = Variable(torch.from_numpy(text_feat)).cuda()
            z = Variable(torch.randn(nSample, z_dim)).cuda()
            G_sample = netG(z, text_feat)
            gen_feat = np.vstack((gen_feat, G_sample.data.cpu().numpy()))

        print gen_feat
        print gen_feat.shape

        targets = np.zeros(gen_feat.shape[0])
        for i in range(50):
            targets[i * nSample: (i + 1) * nSample] = i
        features = gen_feat.astype('float32')
        targets  = targets.astype('int')

        ## start training the 50-way classifer.
        # train = data_utils.TensorDataset(features, targets)
        # train_loader = data_utils.DataLoader(train, batch_size=300, shuffle=True)
        #
        print("Training the SVM")
        clf = svm.SVC(decision_function_shape='ovr')  ### ovo
        clf.fit(features, targets)
        print("Training the SVM Done")
        SVM_name =  'trained_SVM_{}Sample_netG{}.pkl'.format(nSample, opt.netG)
        with open(SVM_name, 'wb') as output:
            pickle.dump(clf, output, pickle.HIGHEST_PROTOCOL)

    else:  # load the trained SVM
        SVM_name = 'trained_SVM_{}Sample_netG{}.pkl'.format(nSample, opt.netG)
        with open(SVM_name, 'rb') as output:
            clf = pickle.load(output)

    ################  Test
    pfc_feat_file_train = 'output/ori_data_classify_train/512feat_46000/deep_fc8.mat'
    pfc_feat_data_train = sio.loadmat(pfc_feat_file_train)
    # pfc_feat_data = pfc_feat_data['deep_feature_list']
    pfc_feat_data_train = pfc_feat_data_train['fc8_feature_list']

    pfc_feat_file = 'output/ori_data_classify_test/512feat_46000/deep_fc8.mat'
    pfc_feat_data = sio.loadmat(pfc_feat_file)
    # pfc_feat_data = pfc_feat_data['deep_feature_list']
    pfc_feat_data = pfc_feat_data['fc8_feature_list']

    # ## Normalize feat_data with mean and var of training data
    mean = pfc_feat_data_train.mean()
    var = pfc_feat_data_train.var()
    pfc_feat_data = (pfc_feat_data - mean) / var
    print("mean: " + str(mean))
    print("var: " + str(var))



    imdb_train = 'CUBird_train_zero_shot'
    imdb_test = 'CUBird_test_zero_shot'
    imdb_train = get_imdb(imdb_train)
    imdb_test = get_imdb(imdb_test)
    roidb_train, roidb_test = get_training_roidb_classify(imdb_train, imdb_test, cfg)
    labels = [sample['label'] for sample in roidb_test]
    preds = []
    preds = clf.predict(pfc_feat_data)
    print preds.shape
    # for i in range(pfc_feat_data.shape[0]):
    #     outputs = classifier.forward(Variable(torch.from_numpy(pfc_feat_data[i])))
    #     pred = np.argmax(outputs.data.numpy())
    #     preds.append(pred)

    acc = 100 * (np.asarray(labels) == preds).sum() / float(len(labels))
    print("Accuracy is {:.4}%".format(acc))

def train():
    netG = _netG().cuda()

    # opt.resume = 'out/zsl_wgan_gp_text/D5_98000.tar'
    # opt.resume = 'out/zsl_wgan_gp_text_Z0/D5_46000.tar'
    opt.resume = 'out/zsl_wgan_gp_text_Z0PureLinear/D5_58000.tar'
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        netG.load_state_dict(checkpoint['state_dict_G'])
        print(checkpoint['log'])
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))

    classifier = _classifier().cuda()
    classifier.apply(weights_init)

    # exp_info = '{}_50way_classifier_Z0'.format(nSample)
    exp_info = '{}_50way_classifier_Z0PureLinear'.format(nSample)
    out_dir = 'out/{:s}'.format(exp_info)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # if 0: #opt.netG != '':
    #     netG.load_state_dict(torch.load(opt.netG))
    print(netG)
    print(classifier)
    test_text_feature = get_testing_text_feature()
    print test_text_feature.shape


    gen_feat = np.zeros([0,X_dim])

    for i in range(50):
        text_feat = np.tile(test_text_feature[i].astype('float32'), (nSample, 1))
        # print text_feat.shape
        text_feat = Variable(torch.from_numpy(text_feat)).cuda()
        # z = Variable(torch.randn(nSample, z_dim)).cuda()
        G_sample = netG(text_feat)
        gen_feat = np.vstack((gen_feat,G_sample.data.cpu().numpy()))

    print gen_feat
    print gen_feat.shape

    # gen_feat = gen_feat - 0.5 #mean

    # check if the random features are trainable.
    # gen_feat = np.random.rand(5000, X_dim)
    targets = np.zeros(gen_feat.shape[0])
    for i in range(50):
        targets[i*nSample: (i+1)*nSample] = i
    features = torch.from_numpy(gen_feat.astype('float32'))
    targets  = torch.from_numpy(targets.astype('int'))

    ## start training the 50-way classifer.
    train = data_utils.TensorDataset(features, targets)
    train_loader = data_utils.DataLoader(train, batch_size=300, shuffle=True)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-4)

    for epoch in range(1000):
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            optimizer.zero_grad()
            outputs = classifier.forward(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
        acc_real = 100 * (np.argmax(outputs.data.cpu().numpy(), axis=1) == labels.data.cpu().numpy()).sum() / float(labels.data.size()[0])
        log_text = 'epoch {}  Acc {:.4}'.format(epoch, acc_real)
        print(log_text)

        if epoch % 100 == 0 and epoch != 0:
            if not os.path.exists('out/'):
                os.makedirs('out/')

            torch.save({
                'epoch': epoch + 1,
                'state_dict_G': classifier.state_dict(),
                'log': log_text,
            }, out_dir + '/epoch_{:d}.tar'.format(epoch))


def test_Perceptron():
    print("Using 2 layer perceptron as classifier")

    classifier = _classifier()
    opt.resume = 'out/500_50way_classifier_Z0PureLinear/epoch_100.tar'
    # opt.resume = 'out/500_50way_classifier_Z10/epoch_1000.tar'
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        classifier.load_state_dict(checkpoint['state_dict_G'])
        print(checkpoint['log'])
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))

    pfc_feat_file = 'output/ori_data_classify_test/512feat_46000/deep_fc8.mat'
    pfc_feat_data = sio.loadmat(pfc_feat_file)
    # pfc_feat_data = pfc_feat_data['deep_feature_list']
    pfc_feat_data = pfc_feat_data['fc8_feature_list']

    pfc_feat_file_train = 'output/ori_data_classify_train/512feat_46000/deep_fc8.mat'
    pfc_feat_data_train = sio.loadmat(pfc_feat_file_train)
    # pfc_feat_data = pfc_feat_data['deep_feature_list']
    pfc_feat_data_train = pfc_feat_data_train['fc8_feature_list']

    # ## Normalize feat_data with mean and var of training data
    # mean = pfc_feat_data_train.mean()
    # var  = pfc_feat_data_train.var()
    # pfc_feat_data = (pfc_feat_data - mean)/var
    # print("mean: " + str(mean))
    # print("var: " + str(var))
    # print(pfc_feat_data)

    imdb_train = 'CUBird_train_zero_shot'
    imdb_test = 'CUBird_test_zero_shot'
    imdb_train = get_imdb(imdb_train)
    imdb_test = get_imdb(imdb_test)
    roidb_train, roidb_test = get_training_roidb_classify(imdb_train, imdb_test, cfg)
    labels = [sample['label'] for sample in roidb_test]
    preds = []
    for i in range(pfc_feat_data.shape[0]):
        outputs = classifier.forward(Variable(torch.from_numpy(pfc_feat_data[i])))
        pred = np.argmax(outputs.data.numpy())
        preds.append(pred)

    acc = 100 * (np.array(labels) == np.array(preds)).sum() / float(len(labels))
    print("Accuracy is {:.4}%".format(acc))




def test_NN():
    print("Using Nearest Neighbor as classifier")
    nSample = 1
    netG = _netG().cuda()

    # opt.resume = 'out/zsl_wgan_gp_text/D5_98000.tar'
    opt.resume = 'out/zsl_wgan_gp_text_Z0PureLinear/D5_58000.tar'
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        netG.load_state_dict(checkpoint['state_dict_G'])
        print(checkpoint['log'])
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))

    exp_info = '{}_50way_classifier_Z0'.format(nSample)
    out_dir = 'out/{:s}'.format(exp_info)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # if 0: #opt.netG != '':
    #     netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    test_text_feature = get_testing_text_feature()
    print test_text_feature.shape
    gen_feat = np.zeros([0,X_dim])

    for i in range(50):
        text_feat = np.tile(test_text_feature[i].astype('float32'), (nSample, 1))
        # print text_feat.shape
        text_feat = Variable(torch.from_numpy(text_feat)).cuda()
        # z = Variable(torch.randn(nSample, z_dim)).cuda()
        G_sample = netG(text_feat)
        gen_feat = np.vstack((gen_feat,G_sample.data.cpu().numpy()))

    print gen_feat
    print gen_feat.shape


    pfc_feat_file = 'output/ori_data_classify_test/512feat_46000/deep_fc8.mat'
    pfc_feat_data = sio.loadmat(pfc_feat_file)
    # pfc_feat_data = pfc_feat_data['deep_feature_list']
    pfc_feat_data = pfc_feat_data['fc8_feature_list']

    pfc_feat_file_train = 'output/ori_data_classify_train/512feat_46000/deep_fc8.mat'
    pfc_feat_data_train = sio.loadmat(pfc_feat_file_train)
    # pfc_feat_data = pfc_feat_data['deep_feature_list']
    pfc_feat_data_train = pfc_feat_data_train['fc8_feature_list']

    # ## Normalize feat_data with mean and var of training data
    mean = pfc_feat_data_train.mean()
    var  = pfc_feat_data_train.var()
    pfc_feat_data = (pfc_feat_data - mean)/var
    print("mean: " + str(mean))
    print("var: " + str(var))
    print(pfc_feat_data)

    ## To get groundtruth label
    imdb_train = 'CUBird_train_zero_shot'
    imdb_test = 'CUBird_test_zero_shot'
    imdb_train = get_imdb(imdb_train)
    imdb_test = get_imdb(imdb_test)
    roidb_train, roidb_test = get_training_roidb_classify(imdb_train, imdb_test, cfg)
    labels = [sample['label'] for sample in roidb_test]
    ## To get groundtruth label End

    preds = []
    for i in range(pfc_feat_data.shape[0]):
        dists = [np.linalg.norm(feat - pfc_feat_data[i]) for feat in gen_feat]
        pred = np.argmax(dists)
        preds.append(pred)

    acc = 100 * (np.array(labels) == np.array(preds)).sum() / float(len(labels))
    print("Accuracy is {:.4}%".format(acc))

if __name__ == "__main__":
    if 1:
        # train()
        train_SVM()
    else:
        #test_Perceptron()
        test_NN()