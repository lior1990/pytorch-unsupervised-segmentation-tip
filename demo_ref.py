#from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import sys
import os
import numpy as np
import torch.nn.init
import random
import glob
import datetime
import tqdm

use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')
parser.add_argument('--nChannel', metavar='N', default=100, type=int, 
                    help='number of channels')
parser.add_argument('--maxIter', metavar='T', default=1, type=int, 
                    help='number of maximum iterations')
parser.add_argument('--maxUpdate', metavar='T', default=1000, type=int, 
                    help='number of maximum update count')
parser.add_argument('--minLabels', metavar='minL', default=3, type=int, 
                    help='minimum number of labels')
parser.add_argument('--batch_size', metavar='bsz', default=1, type=int, 
                    help='number of batch_size')
parser.add_argument('--lr', metavar='LR', default=0.1, type=float, 
                    help='learning rate')
parser.add_argument('--nConv', metavar='M', default=2, type=int, 
                    help='number of convolutional layers')
parser.add_argument('--visualize', metavar='1 or 0', default=1, type=int, 
                    help='visualization flag')
parser.add_argument('--input', metavar='FOLDERNAME',
                    help='input image folder name', required=True)
parser.add_argument('--stepsize_sim', metavar='SIM', default=1, type=float,
                    help='step size for similarity loss', required=False)
parser.add_argument('--stepsize_con', metavar='CON', default=5, type=float, 
                    help='step size for continuity loss')
parser.add_argument('--labels_start_index', default=0, type=int)

args = parser.parse_args()

# CNN model
class MyNet(nn.Module):
    def __init__(self,input_dim):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, args.nChannel, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(args.nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(args.nConv-1):
            self.conv2.append( nn.Conv2d(args.nChannel, args.nChannel, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(args.nChannel) )
        self.conv3 = nn.Conv2d(args.nChannel, args.nChannel, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(args.nChannel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(args.nConv-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x

img_list = sorted(glob.glob(args.input+'/ref/*'))
im = cv2.imread(img_list[0])

# train
model = MyNet( im.shape[2] )
if use_cuda:
    model.cuda()
model.train()

# similarity loss definition
loss_fn = torch.nn.CrossEntropyLoss()

# continuity loss definition
loss_hpy = torch.nn.L1Loss(size_average = True)
loss_hpz = torch.nn.L1Loss(size_average = True)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

for batch_idx in range(args.maxIter):
    print('Training started. '+str(datetime.datetime.now())+'   '+str(batch_idx+1)+' / '+str(args.maxIter))
    for im_file in range(int(len(img_list)/args.batch_size)):
        for loop in tqdm.tqdm(range(args.maxUpdate)):
            im = []
            for batch_count in range(args.batch_size):
                # load image
                resized_im = cv2.imread(img_list[args.batch_size*im_file + batch_count])
                resized_im = cv2.resize(resized_im, dsize=(256, 256))
                resized_im = resized_im.transpose( (2, 0, 1) ).astype('float32')/255.
                im.append(resized_im)

            data = torch.from_numpy( np.array(im) )
            if use_cuda:
                data = data.cuda()
            data = Variable(data)
    
            HPy_target = torch.zeros(data.shape[0], resized_im.shape[1]-1, resized_im.shape[2], args.nChannel)
            HPz_target = torch.zeros(data.shape[0], resized_im.shape[1], resized_im.shape[2]-1, args.nChannel)
            if use_cuda:
                HPy_target = HPy_target.cuda()
                HPz_target = HPz_target.cuda()

            # forwarding
            optimizer.zero_grad()
            output = model( data )
            output = output.permute( 0, 2, 3, 1 ).contiguous().view( data.shape[0], -1, args.nChannel )

            outputHP = output.reshape( (data.shape[0], resized_im.shape[1], resized_im.shape[2], args.nChannel) )
    
            HPy = outputHP[:, 1:, :, :] - outputHP[:, 0:-1, :, :]
            HPz = outputHP[:, :, 1:, :] - outputHP[:, :, 0:-1, :]    
            lhpy = loss_hpy(HPy,HPy_target)
            lhpz = loss_hpz(HPz,HPz_target)

            output = output.reshape( output.shape[0] * output.shape[1], -1 )
            ignore, target = torch.max( output, 1 )

            im_target = target.data.cpu().numpy()
            nLabels = len(np.unique(im_target))

            loss = args.stepsize_sim * loss_fn(output, target) + args.stepsize_con * (lhpy + lhpz)
            loss.backward()
            optimizer.step()

            print(' label num :', nLabels, ' | loss :', loss.item())

            if nLabels <= args.minLabels:
                print("nLabels", nLabels, "reached minLabels", args.minLabels, ".")
                break

    torch.save(model.state_dict(), os.path.join(args.input, 'b'+str(args.batch_size)+'_itr'+str(args.maxIter)+'_layer'+str(args.nConv+1)+'.pth'))

test_img_list = sorted(glob.glob(args.input+'/test/*'))

os.makedirs(os.path.join(args.input, 'result/'), exist_ok=True)
os.makedirs(os.path.join(args.input, 'resized/'), exist_ok=True)


def replace_indices(arr: "np.array") -> "np.array":
    d = {}

    new_arr = np.zeros_like(arr)
    values = np.arange(args.nChannel) + 1 + args.labels_start_index
    free_index = 0

    for i, val in enumerate(arr):
        if val not in d:
            d[val] = values[free_index]
            free_index += 1

        new_arr[i] = d[val]

    return new_arr


print('Testing '+str(len(test_img_list))+' images.')

global_flatten_inds = None

for img_file in tqdm.tqdm(test_img_list):
    im = cv2.imread(img_file)
    im = cv2.resize(im, dsize=(256, 256))

    data = torch.from_numpy( np.array([im.transpose( (2, 0, 1) ).astype('float32')/255.]) )
    if use_cuda:
        data = data.cuda()
    data = Variable(data)
    output = model( data )[ 0 ]
    output = output.permute( 1, 2, 0 ).contiguous().view( -1, args.nChannel )
    ignore, target = torch.max( output, 1 )
    flatten_inds = target.data.cpu().numpy()

    if global_flatten_inds is None:
        unique_labels = np.unique(flatten_inds)
        global_flatten_inds = unique_labels
    else:
        unique_labels = np.unique(flatten_inds)
        assert (global_flatten_inds == unique_labels), f"{global_flatten_inds}, {unique_labels}"

    inds = replace_indices(flatten_inds).reshape( (im.shape[0], im.shape[1]) ).astype( np.uint8 )
    print(f"labels: {unique_labels}")
    cv2.imwrite( os.path.join(args.input, 'result/') + os.path.basename(img_file), inds )
    cv2.imwrite(os.path.join(args.input, 'resized/') + os.path.basename(img_file), im )
