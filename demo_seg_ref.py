# from __future__ import print_function
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import cv2
import numpy as np
import torch.nn.init

from utils import replace_indices

use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')
parser.add_argument('--maxIter', metavar='T', default=1000, type=int,
                    help='number of maximum iterations')
parser.add_argument('--lr', metavar='LR', default=0.1, type=float,
                    help='learning rate')
parser.add_argument('--nConv', metavar='M', default=2, type=int,
                    help='number of convolutional layers')
parser.add_argument('--visualize', metavar='1 or 0', default=1, type=int,
                    help='visualization flag')
parser.add_argument('--input', metavar='FILENAME',
                    help='input image file name', required=True)
parser.add_argument('--ref_input', metavar='REF_FILENAME',
                    help='reference input image file name', required=True)
parser.add_argument('--stepsize_sim', metavar='SIM', default=1, type=float,
                    help='step size for similarity loss', required=False)
parser.add_argument('--stepsize_con', metavar='CON', default=1, type=float,
                    help='step size for continuity loss')
parser.add_argument('--stepsize_scr', metavar='SCR', default=0.5, type=float,
                    help='step size for scribble loss')
args = parser.parse_args()


# CNN model
class MyNet(nn.Module):
    def __init__(self, input_dim, n_channel):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, n_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(n_channel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(args.nConv - 1):
            self.conv2.append(nn.Conv2d(n_channel, n_channel, kernel_size=3, stride=1, padding=1))
            self.bn2.append(nn.BatchNorm2d(n_channel))
        self.conv3 = nn.Conv2d(n_channel, n_channel, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(n_channel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        for i in range(args.nConv - 1):
            x = self.conv2[i](x)
            x = F.relu(x)
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x


# load image
im = cv2.imread(args.input)
ref_im = cv2.imread(args.ref_input)
ref_label = cv2.imread(args.ref_input.replace(".jpg", ".png"), cv2.IMREAD_GRAYSCALE)

data = torch.from_numpy(np.array([im.transpose((2, 0, 1)).astype('float32') / 255.]))
ref_data = torch.from_numpy(np.array([ref_im.transpose((2, 0, 1)).astype('float32') / 255.]))
ref_label = ref_label
n_channel = len(np.unique(ref_label))
ref_label = torch.from_numpy(replace_indices(ref_label.flatten(), n_channel)).long()

if use_cuda:
    data = data.cuda()
    ref_data = ref_data.cuda()
    ref_label = ref_label.cuda()

data = Variable(data)
ref_data = Variable(ref_data)
ref_label = Variable(ref_label.squeeze(0))

# train
model = MyNet(data.size(1), n_channel)
if use_cuda:
    model.cuda()
model.train()

# similarity loss definition
loss_fn = torch.nn.CrossEntropyLoss()

# continuity loss definition
loss_hpy = torch.nn.L1Loss(size_average=True)
loss_hpz = torch.nn.L1Loss(size_average=True)

HPy_target = torch.zeros(im.shape[0] - 1, im.shape[1], n_channel)
HPz_target = torch.zeros(im.shape[0], im.shape[1] - 1, n_channel)
if use_cuda:
    HPy_target = HPy_target.cuda()
    HPz_target = HPz_target.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
label_colours = np.random.randint(255, size=(100, 3))

ref_loss = torch.tensor(1)
batch_idx = 0

from tqdm import tqdm
progress_bar = tqdm(range(1))

while not np.isclose(ref_loss.item(), 0):
    # forwarding
    optimizer.zero_grad()
    output, ref_output = model(torch.cat([data, ref_data]))
    output = output.permute(1, 2, 0).contiguous().view(-1, n_channel)
    ref_output = ref_output.permute(1,2,0).contiguous().view(-1, n_channel)

    outputHP = output.reshape((im.shape[0], im.shape[1], n_channel))
    HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
    HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
    lhpy = loss_hpy(HPy, HPy_target)
    lhpz = loss_hpz(HPz, HPz_target)

    ignore, target = torch.max(output, 1)
    im_target = target.data.cpu().numpy()
    nLabels = len(np.unique(im_target))

    ref_loss = loss_fn(ref_output, ref_label)

    # loss
    loss = args.stepsize_sim * loss_fn(output, target) + args.stepsize_con * (lhpy + lhpz) + ref_loss

    loss.backward()
    optimizer.step()

    progress_bar.set_description(f"{batch_idx} '|' ' label num : {nLabels}  | loss :', {loss.item()} , ' | ref loss :', {ref_loss.item()}")
    batch_idx += 1


label_colours = np.random.randint(255,size=(n_channel,3))

# save output image
if not args.visualize:
    output, _ = model(torch.cat([data, ref_data]))
    output = output.permute(1, 2, 0).contiguous().view(-1, n_channel)
    ignore, target = torch.max(output, 1)
    im_target = target.data.cpu().numpy()
    im_target = im_target.reshape(im.shape[0], im.shape[1]).astype(np.uint8)
    inds_rgb = np.array([label_colours[ c % n_channel ] for c in im_target])
    inds_rgb = inds_rgb.reshape( im.shape ).astype( np.uint8 )
cv2.imwrite("output.png", im_target)
cv2.imwrite("output_rgb.png", inds_rgb)
