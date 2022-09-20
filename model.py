import torch
import torch.nn as nn

import numpy as np
from numpy.core.defchararray import encode
import torch
import torch.nn.functional as F
import os
import sklearn.neighbors as n
from skimage.transform import resize
from skimage import color

class DnCNN(nn.Module):
    # def __init__(self, channels, num_of_layers=17):
    #     super(DnCNN, self).__init__()
    #     kernel_size = 3
    #     padding = 1
    #     features = 64
    #     layers = []
    #     layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
    #     layers.append(nn.ReLU(inplace=True))
    #     for _ in range(num_of_layers-2):
    #         layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
    #         layers.append(nn.BatchNorm2d(features))
    #         layers.append(nn.ReLU(inplace=True))
    #     layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
    #     self.dncnn = nn.Sequential(*layers)
    # def forward(self, x):
    #     out = self.dncnn(x)
    #     return out

    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        self.input = nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=channels, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer()
        self.layer2 = self.make_layer()
        self.layer3 = self.make_layer()
        self.layer4 = self.make_layer()
        self.layer5 = self.make_layer()
        self.layer6 = self.make_layer()
        self.layer7 = self.make_layer()
        self.layer8 = self.make_layer()
        self.layer9 = self.make_layer()
        self.layer10 = self.make_layer()
        # self.layer11 = self.make_layer()
        # self.layer12 = self.make_layer()
        # self.layer13 = self.make_layer()
        # self.layer14 = self.make_layer()
        # self.layer15 = self.make_layer()
        # self.layer16 = self.make_layer()
        # self.layer17 = self.make_layer()

    def forward(self, x):
        x = self.relu(self.input(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        # x = self.layer11(x)
        # x = self.layer12(x)
        # x = self.layer13(x)
        # x = self.layer14(x)
        # x = self.layer15(x)
        # x = self.layer16(x)
        # x = self.layer17(x)
        out = self.output(x)

        return out

    def make_layer(self):

        layers = []
        layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1,bias=False))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)
def weights_init(model):
    if type(model) in [nn.Conv2d, nn.Linear]:
        nn.init.xavier_normal_(model.weight.data)
        nn.init.constant_(model.bias.data, 0.1)    
class Color_model(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(Color_model, self).__init__()
        
        model1=[nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),]
        model1+=[nn.ReLU(True),]
        model1+=[nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),]
        model1+=[nn.ReLU(True),]
        model1+=[norm_layer(64),]

        model2=[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),]
        model2+=[nn.ReLU(True),]
        model2+=[nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),]
        model2+=[nn.ReLU(True),]
        model2+=[norm_layer(128),]

        model3=[nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[norm_layer(256),]

        model4=[nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[norm_layer(512),]

        model5=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[norm_layer(512),]

        model6=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[norm_layer(512),]

        model7=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[norm_layer(512),]

        model8=[nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]

        model8+=[nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True),]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8 = nn.Sequential(*model8)
        self.apply(weights_init)

    def forward(self, gray_image):
        conv1_2 = self.model1(gray_image)  # [bs, 64, 112, 112]
        conv2_2 = self.model2(conv1_2)  # [bs, 128, 56, 56]
        conv3_3 = self.model3(conv2_2)  # [bs, 256, 28, 28]
        conv4_3 = self.model4(conv3_3)  # [bs, 512, 28, 28]
        conv5_3 = self.model5(conv4_3)  # [bs, 512, 28, 28]
        conv6_3 = self.model6(conv5_3)  # [bs, 512, 28, 28]
        conv7_3 = self.model7(conv6_3)  # [bs, 512, 28, 28]
        features = self.model8(conv7_3)  # [bs, 313, 56, 56]
        features = features/0.38  # 0.38 is Softmax temperature T. Paper Eq.(5)
        return features

class NNEncLayer(object):
  ''' 
  使用soft-encoding scheme，将ab空间量化为Q=313个量化级，对应paper中的H^(-1)映射
  --Inputs
      x: [bs, 2, H, W]
  OUTPUTS
      encode
      max_encode
  '''

  def __init__(self):
      self.NN = 10
      self.sigma = 5.
      self.ENC_DIR = 'models/resources/'
      self.nnenc = NNEncode(self.NN, self.sigma, km_filepath=os.path.join(self.ENC_DIR, 'pts_in_hull.npy'))

      self.X = 224
      self.Y = 224
      self.Q = self.nnenc.K  # 313

  def forward(self, x):
      #return np.argmax(self.nnenc.encode_points_mtx_nd(x), axis=1).astype(np.int32)
      encode = self.nnenc.encode_points_mtx_nd(x)  # [bs, 313, 56, 56]，对应paper Eq.(2)中的 Z_{h,w,q}
      max_encode = np.argmax(encode, axis=1).astype(np.int32)  # [bs, 56, 56]，用于Eq.(2)的loss计算
      return encode, max_encode

  def reshape(self, bottom, top):
      top[0].reshape(self.N, self.Q, self.X, self.Y)


class PriorBoostLayer(object):
  '''
  根据在ImageNet上统计得到的ab通道先验概率进行加权
  Layer boosts ab values based on their rarity
  INPUTS    Z_{h,w,q}, shape:[bs, Q, H, W] 即 encode[bs, 313, 56, 56]
  OUTPUTS   v(Z_{h,w}), shape:[bs, 1, H, W]
  '''
  def __init__(self):
      self.ENC_DIR = 'models/resources/'
      self.gamma = .5  # lambda in paper Eq.(4)
      self.alpha = 1.
      self.pc = PriorFactor(self.alpha, gamma=self.gamma, priorFile=os.path.join(self.ENC_DIR, 'prior_probs.npy'))

      self.X = 224
      self.Y = 224

  def forward(self, bottom):
      return self.pc.forward(bottom, axis=1)


class NonGrayMaskLayer(object):
  ''' Layer outputs a mask based on if the image is grayscale or not
  INPUTS
      bottom[0]       Nx2xXxY     ab values
  OUTPUTS
      top[0].data     Nx1xXxY     1 if image is NOT grayscale
                                  0 if image is grayscale
  '''

  def setup(self, bottom, top):
      if len(bottom) == 0:
          raise Exception("Layer should have inputs")

      self.thresh = 5  # threshold on ab value
      self.N = bottom.data.shape[0]
      self.X = bottom.data.shape[2]
      self.Y = bottom.data.shape[3]

  def forward(self, bottom):
      bottom=bottom.numpy()
      # if an image has any (a,b) value which exceeds threshold, output 1
      # ab通道数值和小于5的空间位置不计算loss
      return (np.sum(np.sum(np.sum((np.abs(bottom) > 5).astype('float'), axis=1), axis=1), axis=1) > 0)[:,
                          na(), na(), na()].astype('float')


class PriorFactor():
  ''' Class handles prior factor '''

  def __init__(self, alpha, gamma=0, verbose=True, priorFile=''):
      # INPUTS
      #   alpha           integer     prior correction factor, 0 to ignore prior, 1 to divide by prior, alpha to divide by prior**alpha
      #   gamma           integer     percentage to mix in uniform prior with empirical prior
      #   priorFile       file        file which contains prior probabilities across classes

      # settings
      self.alpha = alpha  # 1
      self.gamma = gamma  # 0.5
      self.verbose = verbose  # True

      # empirical prior probability
      self.prior_probs = np.load(priorFile)  # shape [313]

      # define uniform probability  对应paper Eq.(4)相加的均匀分布
      self.uni_probs = np.zeros_like(self.prior_probs)
      self.uni_probs[self.prior_probs != 0] = 1.
      self.uni_probs = self.uni_probs / np.sum(self.uni_probs)  # 1/Q

      # convex combination of empirical prior and uniform distribution
      self.prior_mix = (1 - self.gamma) * self.prior_probs + self.gamma * self.uni_probs
      # set prior factor
      self.prior_factor = self.prior_mix ** -self.alpha  # ((1-\lambda)*p+\lambda/Q)^-1, paper Eq.(4)的前半部分
      self.prior_factor = self.prior_factor / np.sum(self.prior_probs * self.prior_factor)  # re-normalize  paper Eq.(4)的后半部分

      # implied empirical prior
      self.implied_prior = self.prior_probs * self.prior_factor
      self.implied_prior = self.implied_prior / np.sum(self.implied_prior)  # re-normalize

      #if (self.verbose):
      #    self.print_correction_stats()

  def print_correction_stats(self):
      print('Prior factor correction:')
      print('  (alpha,gamma) = (%.2f, %.2f)' % (self.alpha, self.gamma))
      print('  (min,max,mean,med,exp) = (%.2f, %.2f, %.2f, %.2f, %.2f)' % (
          np.min(self.prior_factor), np.max(self.prior_factor), np.mean(self.prior_factor),
          np.median(self.prior_factor),
          np.sum(self.prior_factor * self.prior_probs)))

  def forward(self, data_ab_quant, axis=1):
      data_ab_maxind = np.argmax(data_ab_quant, axis=axis)
      corr_factor = self.prior_factor[data_ab_maxind]
      if (axis == 0):
          return corr_factor[na(), :]
      elif (axis == 1):
          return corr_factor[:, na(), :]  # [bs, 1, 56, 56]
      elif (axis == 2):
          return corr_factor[:, :, na(), :]
      elif (axis == 3):
          return corr_factor[:, :, :, na()]


class NNEncode():
  ''' 使用NearestNeighbors搜索和高斯核对ab空间point进行编码 '''
  def __init__(self, NN, sigma, km_filepath='', cc=-1):
      if (check_value(cc, -1)):
          self.cc = np.load(km_filepath)  # 
      else:
          self.cc = cc
      self.K = self.cc.shape[0]  # Q=313
      self.NN = int(NN)  # 10
      self.sigma = sigma  # 5
      self.nbrs = n.NearestNeighbors(n_neighbors=NN, algorithm='ball_tree').fit(self.cc)
      self.alreadyUsed = False

  def encode_points_mtx_nd(self, pts_nd, axis=1, returnSparse=False, sameBlock=True):
      pts_flt = flatten_nd_array(pts_nd, axis=axis)  # reshape tensor [bs, axis, H, W]-->[bs*H*W, axis]
      P = pts_flt.shape[0]  # bs*H*W
      if (sameBlock and self.alreadyUsed):
          self.pts_enc_flt[...] = 0  # already pre-allocated
      else:
          self.alreadyUsed = True
          self.pts_enc_flt = np.zeros((P, self.K))  # [bs*H*W, 313]
          self.p_inds = np.arange(0, P, dtype='int')[:, na()]  # 0,...,bs*H*W-1 [bs*H*W, 1]
          
      (dists, inds) = self.nbrs.kneighbors(pts_flt)  # dist: [bs*H*W, 10], inds: [bs*H*W, 10]
      #  distances and indices to the neighbors of each point.
      wts = np.exp(-dists ** 2 / (2 * self.sigma ** 2))  # [bs*H*W, 10]
      wts = wts / np.sum(wts, axis=1)[:, na()]  # [bs*H*W, 10]
      # 上面计算得到 wts，对应paper中为每个Y的ab找5个最近邻，根据dist加权得到weights
      self.pts_enc_flt[self.p_inds, inds] = wts
      
      pts_enc_nd = unflatten_2d_array(self.pts_enc_flt, pts_nd, axis=axis)  # [bs, 313, 56, 56]
      return pts_enc_nd
  
# *****************************
# ***** Utility functions *****
# *****************************
def check_value(inds, val):
  ''' Check to see if an array is a single element equaling a particular value
  for pre-processing inputs in a function '''
  if (np.array(inds).size == 1):
      if (inds == val):
          return True
  return False


def na():  
  # shorthand for new axis
  return np.newaxis


def flatten_nd_array(pts_nd, axis=1):
  ''' Flatten an nd array into a 2d array with a certain axis
  INPUTS
      pts_nd       N0xN1x...xNd array
      axis         integer
  OUTPUTS
      pts_flt     prod(N \ N_axis) x N_axis array     '''
  NDIM = pts_nd.dim()  # 4
  SHP = np.array(pts_nd.shape)  # array([40, 2, 56, 56])
  nax = np.setdiff1d(np.arange(0, NDIM), np.array((axis)))  # array([0, 2, 3]). 找两个数组的差异，此处表示 non axis indices
  NPTS = np.prod(SHP[nax])  # 给定轴上的数组元素的乘积，40*56*56=125440
  axorder = tuple(np.concatenate((nax, np.array(axis).flatten()), axis=0).tolist())  # (0, 2, 3, 1)
  pts_flt = pts_nd.permute(axorder)  # pytorch, [bs, 56, 56, 2]
  pts_flt = pts_flt.contiguous().view(NPTS.item(), SHP[axis].item())  # pytorch, [bs*56*56, 2]
  return pts_flt


def unflatten_2d_array(pts_flt, pts_nd, axis=1, squeeze=False):
  ''' Unflatten a 2d array with a certain axis
  INPUTS
      pts_flt     prod(N \ N_axis) x M array
      pts_nd      N0xN1x...xNd array
      axis        integer
      squeeze     bool     if true, M=1, squeeze it out
  OUTPUTS
      pts_out     N0xN1x...xNd array        '''
  NDIM = pts_nd.dim()
  SHP = np.array(pts_nd.shape)
  nax = np.setdiff1d(np.arange(0, NDIM), np.array((axis)))  # non axis indices

  if (squeeze):
      axorder = nax
      axorder_rev = np.argsort(axorder)
      NEW_SHP = SHP[nax].tolist()

      pts_out = pts_flt.reshape(NEW_SHP)
      pts_out = pts_out.transpose(axorder_rev)
  else:
      axorder = np.concatenate((nax, np.array(axis).flatten()), axis=0)
      axorder_rev = tuple(np.argsort(axorder).tolist())
      M = pts_flt.shape[1]
      NEW_SHP = SHP[nax].tolist()
      NEW_SHP.append(M)
      pts_out = pts_flt.reshape(NEW_SHP)
      pts_out = pts_out.transpose(axorder_rev)
  return pts_out


def decode(data_l, conv8_313, rebalance=1):
  #print('data_l',type(data_l))
  #print('shape',data_l.shape)
  #np.save('data_l.npy',data_l)
  data_l=data_l[0]+50
  data_l=data_l.cpu().data.numpy().transpose((1,2,0))
  conv8_313 = conv8_313[0]
  enc_dir = './resources'
  conv8_313_rh = conv8_313 * rebalance
  #print('conv8',conv8_313_rh.size())
  class8_313_rh = F.softmax(conv8_313_rh,dim=0).cpu().data.numpy().transpose((1,2,0))
  #np.save('class8_313.npy',class8_313_rh)
  class8=np.argmax(class8_313_rh,axis=-1)
  #print('class8',class8.shape)
  cc = np.load(os.path.join(enc_dir, 'pts_in_hull.npy'))
  #data_ab = np.dot(class8_313_rh, cc)
  data_ab=cc[class8[:][:]]
  #data_ab=np.transpose(data_ab,axes=(1,2,0))
  #data_l=np.transpose(data_l,axes=(1,2,0))
  #data_ab = resize(data_ab, (224, 224,2))
  data_ab=data_ab.repeat(4, axis=0).repeat(4, axis=1)
  
  img_lab = np.concatenate((data_l, data_ab), axis=-1)
  img_rgb = color.lab2rgb(img_lab)

  return img_rgb
