import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.autograd as autograd
import torch.optim as optim
import torch.nn.init as init

import __init__

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


parser = argparse.ArgumentParser()
parser.add_argument('--gpu' , default='0', type=str, help='index of GPU to use')
opt = parser.parse_args()
print(opt)

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

cfg_file = 'experiments/cfgs/faster_rcnn_ZSL.yml'
# load config
cfg_from_file(cfg_file)
cfg = edict(cfg)

#mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
#mb_size = 32
LAMBDA = 10 # Gradient penalty lambda hyperparameter
rdc_text_dim = 1000
z_dim = 100
X_dim = 512
text_dim = 11083
y_dim = 150  # label
h_dim = 512
cnt = 0
eps = 1e-8
# opt.clamp_lower = -0.01
# opt.clamp_upper = 0.01
opt.resume = None
opt.netG = 2  # _netG
         # 2  _netG2,  reduce the dim of text first
class _netG(nn.Module):
    def __init__(self):
        super(_netG, self).__init__()
        self.main = nn.Sequential(nn.Linear(z_dim + text_dim, h_dim),
                                  nn.LeakyReLU(),
                                  nn.Linear(h_dim, X_dim),
                                  nn.Sigmoid())

    def forward(self, z, c):
        input = torch.cat([z, c], 1)
        output = self.main(input)
        return output

## reduce to dim of text first
class _netG2(nn.Module):
    def __init__(self):
        super(_netG2, self).__init__()
        self.rdc_text = nn.Linear(text_dim, rdc_text_dim)
        self.main = nn.Sequential(nn.Linear(z_dim + rdc_text_dim, h_dim),
                                  nn.LeakyReLU(),
                                  nn.Linear(h_dim, X_dim),
                                  nn.Sigmoid())

    def forward(self, z, c):
        rdc_text = self.rdc_text(c)
        input = torch.cat([z, rdc_text], 1)
        output = self.main(input)
        return output

# class _netD(nn.Module):
#     def __init__(self):
#         super(_netD, self).__init__()
#         # Discriminator net layer one
#         self.D_shared = nn.Sequential(torch.nn.Linear(X_dim + text_dim, h_dim),
#                                       torch.nn.PReLU())
#         # Discriminator net branch one: For Gan_loss
#         self.D_gan = nn.Sequential(torch.nn.Linear(h_dim, 1),
#                                    torch.nn.Sigmoid())
#         # Discriminator net branch two: For aux cls loss
#         self.D_aux = nn.Linear(h_dim, y_dim)
#
#     def forward(self, x, c):
#         input = torch.cat([x, c], 1)
#         h = self.D_shared(input)
#         return self.D_gan(h), self.D_aux(h)
class _WnetD(nn.Module):
    def __init__(self):
        super(_WnetD, self).__init__()
        # Discriminator net layer one
        self.D_shared = nn.Sequential(nn.Linear(X_dim, h_dim),
                                      nn.ReLU())
        # Discriminator net branch one: For Gan_loss
        self.D_gan = nn.Linear(h_dim, 1)
        # Discriminator net branch two: For aux cls loss
        self.D_aux = nn.Linear(h_dim, y_dim)


    def forward(self, input):
        h = self.D_shared(input)
        return self.D_gan(h), self.D_aux(h)
#
#         # Generator function
# def G(z, c):
#     inputs = torch.cat([z, c], 1)
#     return G_(inputs)
#
# # Discriminator net layer one
# D_shared = torch.nn.Sequential(
#     torch.nn.Linear(X_dim, h_dim),
#     torch.nn.PReLU()
# )
#
# # Discriminator net branch one: For Gan_loss
# D_gan = torch.nn.Sequential(
#     torch.nn.Linear(h_dim, 1),
#     torch.nn.Sigmoid()
# )
#
# # Discriminator net branch two: For aux cls loss
# D_aux = torch.nn.Sequential(
#     torch.nn.Linear(h_dim, y_dim)
# )
#
# # discriminator function
# def D(X):
#     h = D_shared(X)
#     return D_gan(h), D_aux(h)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Linear') != -1:
        print('Initialize Linear moduel')
        init.xavier_normal(m.weight.data)
        init.constant(m.bias, 0.0)

def reset_grad(nets):
    for net in nets:
        net.zero_grad()

def label2mat(labels, y_dim):
    c = np.zeros([labels.shape[0], y_dim])
    for idx, d in enumerate(labels):
        c[idx, d] = 1
    return c



def calc_gradient_penalty(netD, real_data, fake_data):
    #print real_data.size()
    alpha = torch.rand(cfg.TRAIN.IMS_PER_BATCH, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates, _= netD(interpolates)

    # TODO: Make ConvBackward diffentiable
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty
    return gradient_penalty


def train():
    pfc_feat_file = 'output/ori_data_classify_train/512feat_46000/deep_fc8.mat'
    pfc_feat_data = sio.loadmat(pfc_feat_file)
    #pfc_feat_data = pfc_feat_data['deep_feature_list']
    pfc_feat_data = pfc_feat_data['fc8_feature_list']
    # ## Normalize feat_data
    mean = pfc_feat_data.mean()
    var  = pfc_feat_data.var()
    pfc_feat_data = (pfc_feat_data - mean)/var
    # print(pfc_feat_data)

    imdb_train = 'CUBird_train_zero_shot'
    imdb_test = 'CUBird_test_zero_shot'

    imdb_train = get_imdb(imdb_train)
    imdb_test = get_imdb(imdb_test)
    roidb_train, roidb_test = get_training_roidb_classify(imdb_train,imdb_test, cfg)
    data_layer = FeatDataLayer(roidb_train, pfc_feat_data, cfg)

    train_text_feature = get_training_text_feature()
    print train_text_feature.shape

    if opt.netG ==1:   # _netG
    # 2  _netG2,  reduce the dim of text first
        netG = _netG().cuda()
    elif opt.netG ==2 :
        netG = _netG2().cuda()
    else:
        print("Wrong netG option")

    netG.apply(weights_init)


    # if 0: #opt.netG != '':
    #     netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    netD = _WnetD().cuda()
    netD.apply(weights_init)
    # if 0: #opt.netG != '':
    #     netD.load_state_dict(torch.load(opt.netG))
    print(netD)


    exp_info = 'zsl_wgan_gp_text_netG{}'.format(opt.netG)
    out_dir  = 'out/{:s}'.format(exp_info)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    log_dir  = out_dir + '/log_{:s}.txt'.format(exp_info)
    with open(log_dir, 'a') as f:
        f.write('Training Start:')
        f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')

    start_step = 0

    # opt.resume = 'out/zsl_gan_G10_10000.tar'
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            netG.load_state_dict(checkpoint['state_dict_G'])
            netD.load_state_dict(checkpoint['state_dict_D'])
            start_step = checkpoint['it']
            print(checkpoint['log'])
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    nets = [netG,netD]


    optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))


    # ones_label = Variable(torch.ones([cfg.TRAIN.IMS_PER_BATCH,1])).cuda()
    # zeros_label = Variable(torch.zeros([cfg.TRAIN.IMS_PER_BATCH,1])).cuda()
    interp_alpha1 = Variable(torch.FloatTensor(cfg.TRAIN.IMS_PER_BATCH, 1).cuda())
    interp_alpha2 = Variable(torch.FloatTensor(cfg.TRAIN.IMS_PER_BATCH, 1).cuda())

    for it in range(start_step, 50000):
        """ Discriminator """
        for _ in range(5):
            # Sample data
            #print('step %d'%it)
            blobs = data_layer.forward()
            feat_data = blobs['data']  # image data
            labels = blobs['labels'].astype(int)  # class labels
            text_feat =  np.array([ train_text_feature[i,:] for i in labels])
            text_feat = Variable(torch.from_numpy(text_feat.astype('float32'))).cuda()
            #X, y = mnist.train.next_batch(mb_size)
            X = Variable(torch.from_numpy(feat_data)).cuda()

            # c is one-hot
            # c = label2mat(labels, y_dim)
            # c = Variable(torch.from_numpy(c.astype('float32'))).cuda()
            # y_true is not one-hot (requirement from nn.cross_entropy)
            y_true = Variable(torch.from_numpy(labels.astype('int'))).cuda()
            z = Variable(torch.randn(cfg.TRAIN.IMS_PER_BATCH, z_dim)).cuda()

            #D_real is for Gan loss, C_real is for label loss.
            D_real, C_real = netD(X)
            # D_loss = torch.mean(torch.log(D_real + eps) + torch.log(1 - D_fake + eps))
            # D_loss_real = F.binary_cross_entropy(D_real, ones_label)
            D_loss_real = torch.mean(D_real)
            C_loss_real = F.cross_entropy(C_real, y_true) # cls loss
            DC_loss = -D_loss_real + C_loss_real
            DC_loss.backward()

            # GAN's D loss
            G_sample = netG(z, text_feat).detach()
            D_fake, C_fake = netD(G_sample)
            # D_loss_fake = F.binary_cross_entropy(D_fake, zeros_label)
            D_loss_fake = torch.mean(D_fake)
            C_loss_fake = F.cross_entropy(C_fake, y_true)
            DC_loss = D_loss_fake + C_loss_fake
            DC_loss.backward()

            # train with gradient penalty
            grad_penalty = calc_gradient_penalty(netD, X.data, G_sample.data)
            grad_penalty.backward()

            D_cost = D_loss_fake - D_loss_real + grad_penalty
            Wasserstein_D = D_loss_real - D_loss_fake
            optimizerD.step()

            # # clamp parameters to a cube
            # for p in netD.parameters():
            #     p.data.clamp_(opt.clamp_lower, opt.clamp_upper)
            reset_grad(nets)
            # Cross entropy aux loss
            # C_loss = -F.cross_entropy(C_real, y_true) - F.cross_entropy(C_fake, y_true)


        """ Generator """
        i = 0
        # while i <=10 or G_loss.data[0] >= 2.0 :
        # for i in range(10):
        for _ in range(1):
            blobs = data_layer.forward()
            feat_data = blobs['data']  # image data
            labels = blobs['labels'].astype(int)  # class labels
            text_feat = np.array([train_text_feature[i, :] for i in labels])
            text_feat = Variable(torch.from_numpy(text_feat.astype('float32'))).cuda()

            X = Variable(torch.from_numpy(feat_data)).cuda()
            # c is one-hot
            # c = label2mat(labels, y_dim)
            # c = Variable(torch.from_numpy(c.astype('float32'))).cuda()

            # y_true is not one-hot (requirement from nn.cross_entropy)
            y_true = Variable(torch.from_numpy(labels.astype('int'))).cuda()
            z = Variable(torch.randn(cfg.TRAIN.IMS_PER_BATCH, z_dim)).cuda()

            G_sample = netG(z, text_feat)
            D_fake, C_fake = netD(G_sample)
            _,      C_real = netD(X)

            # GAN's G loss
            G_loss = torch.mean(D_fake)
            # G_loss = F.binary_cross_entropy(D_fake, ones_label)
            #G_loss = torch.mean(torch.log(D_fake + eps))
            # Cross entropy aux loss
            C_loss = (F.cross_entropy(C_real, y_true) + F.cross_entropy(C_fake, y_true))/2

            # Maximize
            GC_loss = -G_loss + C_loss
            #GC_loss = -(G_loss)
            GC_loss.backward()
            optimizerG.step()
            reset_grad(nets)
            # i += 1
            # if i >= 100:
            #     print('Note: # of iter for G_net >= 100.')
            #     break

            #print(G_loss)
        # Print and plot every now and then
        if it % 50 == 0: #& it != 0:
            # # idx = np.random.randint(0, 10)
            # c = np.zeros([16, y_dim])
            # c[range(16), idx] = 1
            # c = Variable(torch.from_numpy(c.astype('float32'))).cuda()
            #
            # z = Variable(torch.randn(16, z_dim)).cuda()
            #
            # samples = netG(z, c).data.cpu().numpy()
            acc_real = (np.argmax(C_real.data.cpu().numpy(), axis=1) == y_true.data.cpu().numpy()).sum() / float(y_true.data.size()[0])
            acc_fake = (np.argmax(C_fake.data.cpu().numpy(), axis=1) == y_true.data.cpu().numpy()).sum() / float(y_true.data.size()[0])
            # acc_D_r = ((D_real.data.cpu().numpy() > 0.5).sum() + (D_fake.data.cpu().numpy() < 0.5).sum()) / (float(y_true.data.size()[0])*2)
            # acc_D_r = (D_real.data.cpu().numpy() > 0.5).sum() / (float(y_true.data.size()[0]))
            # acc_D_f = (D_fake.data.cpu().numpy() < 0.5).sum() / (float(y_true.data.size()[0]))
            # D_loss_data = (D_loss_fake.data[0] + D_loss_real.data[0])/2

            log_text = 'Iter-{}; D_loss: {:.4}; Was_D: {:.4}; G_loss: {:.4}; D_loss_real: {:.4}; D_loss_fake: {:.4}; C_loss: {:.4};  acc_real: {:.4}; acc_fake: {:.4}'\
                        .format(it, D_cost.data[0], Wasserstein_D.data[0], G_loss.data[0], D_loss_real.data[0], D_loss_fake.data[0], C_loss.data[0],  acc_real * 100, acc_fake * 100)
            print(log_text)
            with open(log_dir, 'a') as f:
                f.write(log_text+'\n')

        if it % 2000 == 0 and it !=0:
            if not os.path.exists('out/'):
                os.makedirs('out/')

            torch.save({
                    'it': it + 1,
                    'state_dict_G': netG.state_dict(),
                    'state_dict_D': netD.state_dict(),
                    'log': log_text,
                },  out_dir + '/D{:d}_{:d}.tar'.format(5, it))


if __name__ == "__main__":
    train()
