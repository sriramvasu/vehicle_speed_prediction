import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
from PIL import Image
# from networks.submodules import *
from util import conv, predict_flow, deconv, crop_like, flow2rgb
from torch.utils.data import DataLoader, Dataset
import cv2
from flow_transforms import *

class FlowNetS(nn.Module):
    expansion = 1
    def __init__(self, batchNorm=False):
        super(FlowNetS,self).__init__()

        self.batchNorm = batchNorm
        self.conv1   = conv(self.batchNorm,   6,   64, kernel_size=7, stride=2)
        self.conv2   = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2)
        self.conv3   = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256,  256)
        self.conv4   = conv(self.batchNorm, 256,  512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512,  512)
        self.conv5   = conv(self.batchNorm, 512,  512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512,  512)
        self.conv6   = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm,1024, 1024)

        self.deconv5 = deconv(1024,512)
        self.deconv4 = deconv(1026,256)
        self.deconv3 = deconv(770,128)
        self.deconv2 = deconv(386,64)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6       = self.predict_flow6(out_conv6)
        flow6_up    = crop_like(self.upsampled_flow6_to_5(flow6), out_conv5)
        out_deconv5 = crop_like(self.deconv5(out_conv6), out_conv5)

        concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
        flow5       = self.predict_flow5(concat5)
        flow5_up    = crop_like(self.upsampled_flow5_to_4(flow5), out_conv4)
        out_deconv4 = crop_like(self.deconv4(concat5), out_conv4)

        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
        flow4       = self.predict_flow4(concat4)
        flow4_up    = crop_like(self.upsampled_flow4_to_3(flow4), out_conv3)
        out_deconv3 = crop_like(self.deconv3(concat4), out_conv3)

        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        flow3       = self.predict_flow3(concat3)
        flow3_up    = crop_like(self.upsampled_flow3_to_2(flow3), out_conv2)
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2)

        concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
        flow2 = self.predict_flow2(concat2)

        if self.training:
            return flow2,flow3,flow4,flow5,flow6
        else:
            return flow2

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

class speed_net(nn.Module):
    def __init__(self, batchNorm=False):
        super(speed_net, self).__init__()
        self.batchNorm = batchNorm
        self.conv1   = conv(self.batchNorm,   2,   64, kernel_size=7, stride=2)
        self.conv2   = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2)
        self.conv3   = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2)


model_weights=torch.load('/home/sriram/Downloads/flownets_EPE1.951.pth.tar', map_location='cpu') 
# model_weights=torch.load('/home/sriram/Downloads/flownets_bn_EPE2.459.pth.tar', map_location='cpu') 
print(model_weights.keys())
print(model_weights['arch'])
for key in model_weights['state_dict']: 
    print(model_weights['state_dict'][key].shape) 

model=FlowNetS()
for param in model.parameters():
	print(param.shape)
# model.load_state_dict(model_weights)

dset=video_loader('/home/sriram/codes/speed_challenge_2017/data/train.mp4', '/home/sriram/codes/speed_challenge_2017/data/train.txt')
split=int(0.25*len(dset))
train_sampler = SubsetRandomSampler(list(range(len(dset)))[split:])
valid_sampler = SubsetRandomSampler(list(range(len(dset)))[:split])
train_videoloader=DataLoader(dset, batch_size=1, num_workers=1, sampler=train_sampler, shuffle=1)
valid_videoloader=DataLoader(dset, batch_size=1, num_workers=1, sampler=valid_sampler, shuffle=0)
model.eval()
max_frames=10
outs=[]
# for i,(frames, speeds) in enumerate(videoloader):
#     if(i>=max_frames):
#         break
#     print(frames.shape)
#     print(speeds)
#     flow=model(frames)
#     # print(frames.size()[3]+[-1:])
#     output = F.interpolate(flow, size=frames.size()[-2:], mode='bilinear', align_corners=False)
#     rgb_flow = flow2rgb(20 * output.squeeze(0), max_value=None)
#     outs.append(rgb_flow.transpose([1,2,0]))
#     print(rgb_flow.transpose([1,2,0]).shape)
#     print('\n')

# print((outs[0].shape[:-1]))
# cap = cv2.VideoWriter('flowout.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 1, (outs[0].shape[:-1]))
# for i in range(len(outs)):
#         # writing to a image array
#         Image.fromarray((255.0*outs[i]).astype('uint8')).show()
#         cap.write((255.0*outs[i]).astype('uint8'))
# cap.release()


import imageio
gif = imageio.get_reader('/home/sriram/codes/speed_challenge_2017/data/input_1.gif')
tx=torchvision.transforms.Compose([ArrayToTensor(), torchvision.transforms.Normalize(mean=[0.0,0.0,0.0],std=[255.0,255.0,255.0]), torchvision.transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])])
# Here's the number you're looking for
number_of_frames = len(gif)
inputs=[]
for i,frame in enumerate(gif):
    inputs.append(tx(frame[:,:,:-1]))
    cv2.imwrite('data/img_%d.png'%(i), frame[:,:,:-1])
  # each frame is a numpy matrix
print(torch.cat(inputs).shape)
o1=model(torch.unsqueeze(torch.cat(inputs[:-1]),0))
o2=F.interpolate(o1, size=inputs[0].size()[-2:], mode='bilinear', align_corners=False)
o3=flow2rgb(20 * o2.squeeze(0), max_value=None)
Image.fromarray(o3.transpose([1,2,0]).astype('uint8')).show()
