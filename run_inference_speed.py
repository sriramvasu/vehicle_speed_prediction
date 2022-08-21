import argparse
from path import Path
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.sampler import SubsetRandomSampler
from models.util import *
from torch.utils.tensorboard import SummaryWriter
import cv2
import torchvision
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import models
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import flow_transforms
from imageio import imread, imwrite
import numpy as np
from util import flow2rgb
import re
import h5py
import random
import os
import datetime
import albumentations as A
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))


parser = argparse.ArgumentParser(description='PyTorch FlowNet inference on a folder of img pairs',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('data', metavar='DIR',
#                     help='path to images folder, image names must match \'[name]0.[ext]\' and \'[name]1.[ext]\'')
parser.add_argument('pretrained', metavar='PTH', help='path to pre-trained model')
parser.add_argument('--gpu', default=0, help='GPU ID')
parser.add_argument('--output', '-o', metavar='DIR', default=None,
                    help='path to output folder. If not set, will be created in data folder')
parser.add_argument('--output-value', '-v', choices=['raw', 'vis', 'both'], default='both',
                    help='which value to output, between raw input (as a npy file) and color vizualisation (as an image file).'
                    ' If not set, will output both')
parser.add_argument('--div-flow', default=20, type=float,
                    help='value by which flow will be divided. overwritten if stored in pretrained file')
parser.add_argument("--img-exts", metavar='EXT', default=['png', 'jpg', 'bmp', 'ppm'], nargs='*', type=str,
                    help="images extensions to glob")
parser.add_argument('--max_flow', default=None, type=float,
                    help='max flow value. Flow map color is saturated above this value. If not set, will use flow map\'s max value')
parser.add_argument('--upsampling', '-u', choices=['nearest', 'bilinear'], default=None, help='if not set, will output FlowNet raw input,'
                    'which is 4 times downsampled. If set, will output full resolution flow map, with selected upsampling')
parser.add_argument('--bidirectional', action='store_true', help='if set, will output invert flow (from 1 to 0) along with regular flow')


# def video2h5(path):
#     vid_name=path.split('.')[0]
#     file=open('%s.%s'%(vid_name, 'txt'),'r')
#     extn=path.split('.')[1]
#     cap=cv2.VideoCapture(path)
#     frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    

def break_video(path, n_parts):
    vid_name=path.split('.')[0]
    file=open('%s.%s'%(vid_name, 'txt'),'r')
    extn=path.split('.')[1]
    cap=cv2.VideoCapture(path)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
    for part in range(n_parts-1):
        # buf=np.empty([int(frameCount/n_parts), frameHeight, frameWidth])
        fc, ret=0, True
        cap_write = cv2.VideoWriter('%s_%d.%s'%(vid_name, part, extn),cv2.VideoWriter_fourcc(*'DIVX'), 30, (frameWidth, frameHeight))
        outfile=open('%s_%d.%s'%(vid_name, part, 'txt'), 'w')
        for frame_no in range(int(frameCount/n_parts)):
        # while(fc>=part*int(frameCount/n_parts) and fc<(part+1)*int(frameCount/n_parts)):
            ret, frame=cap.read()
            cap_write.write(frame)
            outfile.write(file.readline())
        cap_write.release()
        outfile.close()
    # buf=np.empty([int(frameCount%n_parts), frameHeight, frameWidth])
    outfile=open('%s_%d.%s'%(vid_name, n_parts-1, 'txt'), 'w')
    cap_write = cv2.VideoWriter('%s_%d.%s'%(vid_name, n_parts-1, extn),cv2.VideoWriter_fourcc(*'DIVX'), 30, (frameWidth, frameHeight))
    while(ret):
        ret, frame=cap.read()
        cap_write.write(frame)
        outfile.write(file.readline())
    cap_write.release()
    outfile.close()
    cap.release()
      

# class load_video_h5(Dataset):
#     def __init__(self, h5_path, num_frames=5):
#         self.h5_path=h5_path
#         self.num_frames=num_frames

#     def __len__(self):




class video_loader(Dataset):
    def load_video_and_speeds(self, path):
        cap = cv2.VideoCapture(path)
        vid_name=path.split('.')[0]
        label_file=open('%s.%s'%(vid_name, 'txt'), 'r')
        # print("Video open status:%s"%(str(cap.isOpened())))
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # print(frameCount, frameWidth, frameHeight)
        buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
        fc, ret, speeds = 0, True, []
        while (fc < frameCount and ret):
            ret, buf[fc] = cap.read()
            speeds.append(float(label_file.readline().strip()))
            fc+=1
        return buf, np.array(speeds)

    def __init__(self, data_path='/home/sriram/Documents/speed_challenge_2017/data', read_re_format='train_\\d{1,2}.mp4', num_frames=5):
        super(video_loader, self).__init__()
        self.speed_mean=12.6739
        self.speed_std=8.30573
        self.imgchannel_mean=[0.485, 0.456, 0.406]
        self.imgchannel_std=[0.229, 0.224, 0.225]
        self.num_frames=num_frames
        matcher=re.compile(read_re_format)
        self.reqd_files=[os.path.join(data_path, x) for x in os.listdir(data_path)if matcher.match(x)]
        self.frame_counts=[cv2.VideoCapture(x).get(cv2.CAP_PROP_FRAME_COUNT) for x in self.reqd_files]
        self.total=0
        # for i,x in enumerate(self.frame_counts):
        #     self.total+=(x-num_frames)

        # tx=ApplyTransforms([SelectFrames(21), SplitFramePairs()])
        # self.video_buffer=SplitFramePairs()(self.video_buffer)
    def __len__(self):
        return len(self.reqd_files)

    def __getitem__(self, idx):
        video_buffer, speeds=self.load_video_and_speeds(self.reqd_files[idx])
        # video_buffer=self.frame_transform(video_buffer)
        # collector=[]
        # for ij in range(video_buffer.shape[0]):
        #     out1=self.size_transforms(image=video_buffer[ij,:])['image']
        #     collector.append(self.pixel_transforms(image=out1)['image'])
            # collector.append(out1)
        # video_buffer=self.frame_transform(np.stack(collector))
        speeds=(speeds-self.speed_mean)/self.speed_std
        return video_buffer, speeds
        # Image.fromarray(ans[0,0,:,:,::-1].astype('uint8')).show()
        # Image.fromarray(ans[0,1,:,:,::-1].astype('uint8')).show()

class transform_img2video():
    def __init__(self, transform):
        self.transform=transform
    def __call__(self, video):
        return np.stack([self.transform(image=video[ij,:])['image'] for ij in range(video.shape[0])])

class DatasetTransformer(Dataset):
    def __init__(self, dataset, transform=lambda x:x):
        super(DatasetTransformer, self).__init__()
        self.dataset=dataset
        self.transform=transform
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        video_buf, speeds=self.dataset[idx]
        return self.transform(video_buf), speeds

class flownet_postprocess(torch.nn.Module):
    def __init__(self, batchNorm=False):
        super(flownet_postprocess, self).__init__()
        self.batchNorm=batchNorm
        self.conv1=conv(self.batchNorm,   2,   256, kernel_size=3, stride=1)
        self.pool1=torch.nn.MaxPool2d(3)
        self.conv2=conv(self.batchNorm, 256, 512, kernel_size=3, stride=1)
        self.pool2=torch.nn.MaxPool2d(3)
    def forward(self, x):
        out1=self.pool1(self.conv1(x))
        return self.pool2(self.conv2(out1))

class optical_flow_extractor(torch.nn.Module):
    def __init__(self, batchNorm=False):
        super(optical_flow_extractor, self).__init__()
        self.batchNorm = batchNorm
        network_data=torch.load(args.pretrained)
        self.flownet_base=models.__dict__[network_data['arch']](network_data).eval()
        self.flownet_post=flownet_postprocess()
    def forward(self, x):
        return self.flownet_base(x)


class convlstmcell(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super(convlstmcell, self).__init__()
        self.hidden_dim=hidden_dim
        self.convlayer=nn.Conv2d(in_channels=input_dim+hidden_dim, out_channels=4*hidden_dim, kernel_size=kernel_size, padding=[kernel_size//2, kernel_size//2]).cuda()
        if(batch_normalization and style==1):
            self.bn=nn.BatchNorm2d(4*hidden_dim)
        if(batch_normalization and style==2):
            self.bn_i=nn.BatchNorm2d(input_dim)
            self.bn_h=nn.BatchNorm2d(hidden_dim)
            self.bn_c=nn.BatchNorm2d(hidden_dim)

    def forward(self, input_tensor, curr_state):
        h_cur, c_cur = curr_state
        if(batch_normalization and style==1):
            combined = torch.cat([input_tensor, h_cur], dim=1)
            combined_conv = self.bn(self.convlayer(combined))
        elif(batch_normalization and style==2):
            combined = torch.cat([self.bn_i(input_tensor), self.bn_h(h_cur)], dim=1)
            combined_conv = self.convlayer(combined)
        else:
            combined = torch.cat([input_tensor, h_cur], dim=1)
            combined_conv = self.convlayer(combined)           
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        if(batch_normalization and style==2):
            c_next = f * self.bn_c(c_cur) + i * g
        else:
            c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class convlstmnet(torch.nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, num_layers, batch_norm=False, return_layer_outputs=False):
        super(convlstmnet, self).__init__()
        self.input_dim=input_dim
        self.return_layer_outputs=return_layer_outputs
        self.input_size=input_size
        self.num_layers=num_layers
        self.hidden_dim=hidden_dim
        self.lstmcells=[convlstmcell(input_dim, hidden_dim)]+[convlstmcell(hidden_dim, hidden_dim) for i in range(num_layers-1)]
        # self.dropouts=[torch.nn.Dropout2d(dropout_prob) for i in range(num_layers-1)]
        self.lstmcells=nn.ModuleList(self.lstmcells)
    def forward(self, input_tensor):
        seq_len=input_tensor.shape[1]
        batch_size=input_tensor.shape[0]
        cell_states=[(torch.zeros([batch_size, self.hidden_dim, self.input_size[0], self.input_size[1]], requires_grad=True).to(device), 
                     torch.zeros([batch_size, self.hidden_dim, self.input_size[0], self.input_size[1]], requires_grad=True).to(device)) for i in range(self.num_layers)]
        outputs=[]
        for i in range(seq_len):
            for j in range(self.num_layers):
                if(j==0):
                    cell_states[j]=self.lstmcells[j](input_tensor[:, i, :], cell_states[j])
                    # cell_states[j]=(self.dropouts[j](cell_states[j][0]), cell_states[j][1])
                elif(j<self.num_layers-1 and j>0):
                    cell_states[j]=self.lstmcells[j](cell_states[j-1][0], cell_states[j])
                    # cell_states[j]=(self.dropouts[j](cell_states[j][0]), cell_states[j][1])
                else:
                    cell_states[j]=self.lstmcells[j](cell_states[j-1][0], cell_states[j])
            if(self.return_layer_outputs):
                outputs.append(cell_states[-1][0])
        if(self.return_layer_outputs):
            return torch.stack(outputs, dim=1)
        else:
            return cell_states[-1][0]

class classification_net(torch.nn.Module):
    def __init__(self, batchNorm=False):
        super(classification_net, self).__init__()
        self.batchNorm = batchNorm
        self.conv1   = conv(self.batchNorm,   hidden_channels,   256, kernel_size=3, stride=1)
        # self.pool1=torch.nn.MaxPool2d(3)
        self.conv2=conv(self.batchNorm, 256, 128, kernel_size=3, stride=1)
        # self.pool2=torch.nn.MaxPool2d(3)
        self.conv3=conv(False, 8192, 500, kernel_size=1, stride=1)
        self.out=nn.Linear(500, 1)

    def forward(self, x):
        batch_size=x.shape[0]
        out1=self.conv2(self.conv1(x))
        out2=self.out(self.conv3(out1.view(batch_size, -1, 1,1)).view(batch_size, -1))
        return out2

class inception_feature_extractor(torch.nn.Module):
    def __init__(self):
        super(inception_feature_extractor, self).__init__()
        inception = torchvision.models.inception_v3(pretrained=True)
        self.part1=torch.nn.Sequential(*list(inception.children())[:15])
        self.out1=list(inception.children())[15]
        self.part2=torch.nn.Sequential(*list(inception.children())[16:19])
        self.out2=list(inception.children())[19]
    def forward(self, x):
        return self.part2(self.part1(x))

def train(flownet_model, inception, lstmnet, speednet, train_videoloader, valid_videoloader, optimizer, writer):
    global_step=0
    print(optimizer.lr, lambda_regul, start_epoch)
    for epoch in range(start_epoch, 300):
        if(epoch%5==0 and epoch>=1):
            print("saving model")
            torch.save({'epoch':epoch,
                        'train_globalstep': global_step,
                        'logdir':writer.log_dir,
                        'flownetpost_statedict':flownet_model.flownet_post.state_dict(), 
                        'lstmnet_statedict':lstmnet.state_dict(),
                        'speednet_statedict':speednet.state_dict(),
                        'optimizer_statedict':optimizer.state_dict()}, 
                        'speed_models/flownet_convlstm_%s_%d'%(prefix, epoch))
        if(epoch%2==0 and epoch>=1):
            validate(flownet_model, inception, lstmnet, speednet, valid_videoloader, epoch, writer)
        for i, (video_buf, speeds) in enumerate(train_videoloader):
            # for ij in range(video_buf.shape[1]):
                # show_img=(video_buf[0, ij,:].numpy().transpose([1,2,0])*255.0*np.array(train_videoloader.dataset.imgchannel_std)+255.0*np.array(train_videoloader.dataset.imgchannel_mean)).astype('uint8')
                # cv2.imshow('window1', show_img)
                # cv2.waitKey(0)
            flownet_model.flownet_base.eval();inception.eval(); 
            flownet_model.flownet_post.train(); lstmnet.train();speednet.train()
            video_buf=video_buf.view(-1,video_buf.shape[-3], video_buf.shape[-2], video_buf.shape[-1])
            video_buf=torch.cat([video_buf[:video_buf.shape[0]-1, :], video_buf[1:, :]], axis=1)
            speeds=speeds.view(-1)[1:]
            flow_out=torch.zeros([video_buf.shape[0], 2, 75, 75])
            inception_features=torch.zeros([video_buf.shape[0], 2048, 8, 8])

            # print(video_buf.shape)
            for j in range(int(video_buf.shape[0]/batch_size)):
                flow_out[j*batch_size:(j+1)*batch_size]=flownet_model(video_buf[j*batch_size:(j+1)*batch_size, :].to(device)).detach().cpu()
                inception_features[j*batch_size:(j+1)*batch_size]=inception(video_buf[j*batch_size:(j+1)*batch_size, 3:,:].to(device)).detach().cpu()
            flow_out[(j+1)*batch_size:, :]=flownet_model(video_buf[(j+1)*batch_size:, :].to(device)).detach().cpu()
            inception_features[(j+1)*batch_size:, :]=inception(video_buf[(j+1)*batch_size:, 3:,:].to(device)).detach().cpu()
            # inception_features=torch.cat([flow_out, inception_features], dim=1)
            # print(j, video_buf.shape, flow_out.shape)
            tot_loss=0.0
            tot_regulloss=0.0
            tot_weightnorm=0.0
            # print(torch.stack([torch.sum(flow_out[:flow_out.shape[0]-num_frames+1, :], dim=(1,2,3)).view(-1), speeds[num_frames-1:]],  dim=0))
            sol=0
            for j in range(0, inception_features.shape[0], batch_size):
                if(j+batch_size+num_frames-1<flow_out.shape[0]):
                    flow_clip=torch.stack([flownet_model.flownet_post(flow_out[ij:ij+num_frames,:].to(device)) for ij in range(j, j+batch_size)], axis=0)
                    img_clip=torch.stack([inception_features[ij:ij+num_frames,:] for ij in range(j, j+batch_size)], axis=0).to(device)
                    label_speeds=torch.stack([speeds[ij+num_frames-1] for ij in range(j, j+batch_size)], axis=0).to(device)
                else:
                    sol=1
                    flow_clip=torch.stack([flownet_model.flownet_post(flow_out[ij:ij+num_frames,:].to(device)) for ij in range(j, flow_out.shape[0]-num_frames+1)], axis=0)
                    img_clip=torch.stack([inception_features[ij:ij+num_frames,:] for ij in range(j, inception_features.shape[0]-num_frames+1)], axis=0).to(device)
                    label_speeds=torch.stack([speeds[ij+num_frames-1] for ij in range(j, flow_out.shape[0]-num_frames+1)], axis=0).to(device)

                lstmnet.zero_grad()
                speednet.zero_grad()
                outer=lstmnet(torch.cat([flow_clip, img_clip], dim=2).to(device))
                pred=speednet(outer).view(-1)
                loss=torch.sum((label_speeds-pred)**2)
                weight_norm=0.0
                for param in list(lstmnet.parameters())+list(speednet.parameters())+list(flownet_model.flownet_post.parameters()):
                    weight_norm+=torch.sum(param**2)
                tot_loss+=loss.detach()
                loss+=lambda_regul*weight_norm
                tot_regulloss+=lambda_regul*weight_norm.detach()
                tot_weightnorm+=weight_norm.detach()
                loss.backward()
                optimizer.step()
                writer.add_scalar("train/speed_loss", loss, global_step)
                writer.add_scalar("train/weight_norm_loss", 0.0025*weight_norm, global_step)
                global_step+=1
                if(sol==1):
                    break

            print("Train epoch: %d, video num: %d, loss: %s, weight_norm: %s, regul_loss: %s"%(epoch, i+1, str(tot_loss), str(tot_weightnorm), str(tot_regulloss)))
            writer.add_scalar("train/video_speed_loss", tot_loss, epoch*0.75*len(train_videoloader.dataset)+i)
            writer.add_scalar("train/video_regul_loss", tot_regulloss, epoch*0.75*len(train_videoloader.dataset)+i)
            writer.add_scalar("train/total_weight_norm", tot_weightnorm, epoch*0.75*len(train_videoloader.dataset)+i)
            
    #     # output = F.interpolate(flow, size=frames.size()[-2:], mode='bilinear', align_corners=False)
    #     # rgb_flow = flow2rgb(20 * output.squeeze(0), max_value=None)
    #     # outs.append((255.0*rgb_flow).transpose([1,2,0]).astype('uint8'))
    #     # print(rgb_flow.transpose([1,2,0]).shape)
    #     # print('\n')

def validate(flownet_model, inception, lstmnet, speednet, valid_videoloader, epoch, writer):
    # print(len(train_videoloader.dataset))
    flownet_model.flownet_base.eval();inception.eval(); 
    flownet_model.flownet_post.eval(); lstmnet.train();speednet.eval()
    valid_global_step=0
    for i, (video_buf, speeds) in enumerate(valid_videoloader):
        global_step=0
        video_buf=video_buf.view(-1,video_buf.shape[-3], video_buf.shape[-2], video_buf.shape[-1])
        video_buf=torch.cat([video_buf[:video_buf.shape[0]-1, :], video_buf[1:, :]], axis=1)
        speeds=speeds.view(-1)[1:]
        flow_out=torch.zeros([video_buf.shape[0], 2, 75, 75])
        inception_features=torch.zeros([video_buf.shape[0], 2048, 8, 8])
        for j in range(int(video_buf.shape[0]/batch_size)):
                flow_out[j*batch_size:(j+1)*batch_size]=flownet_model(video_buf[j*batch_size:(j+1)*batch_size, :].to(device)).detach().cpu()
                inception_features[j*batch_size:(j+1)*batch_size]=inception(video_buf[j*batch_size:(j+1)*batch_size, 3:,:].to(device)).detach().cpu()
        flow_out[(j+1)*batch_size:, :]=flownet_model(video_buf[(j+1)*batch_size:, :].to(device)).detach().cpu()
        inception_features[(j+1)*batch_size:, :]=inception(video_buf[(j+1)*batch_size:, 3:,:].to(device)).detach().cpu()
        tot_loss=0.0

        sol=0
        for j in range(0, inception_features.shape[0], batch_size):
            if(j+batch_size+num_frames-1<flow_out.shape[0]):
                flow_clip=torch.stack([flownet_model.flownet_post(flow_out[ij:ij+num_frames,:].to(device)) for ij in range(j, j+batch_size)], axis=0)
                img_clip=torch.stack([inception_features[ij:ij+num_frames,:] for ij in range(j, j+batch_size)], axis=0).to(device)
                label_speeds=torch.stack([speeds[ij+num_frames-1] for ij in range(j, j+batch_size)], axis=0).to(device)
            else:
                sol=1
                flow_clip=torch.stack([flownet_model.flownet_post(flow_out[ij:ij+num_frames,:].to(device)) for ij in range(j, flow_out.shape[0]-num_frames+1)], axis=0)
                img_clip=torch.stack([inception_features[ij:ij+num_frames,:] for ij in range(j, inception_features.shape[0]-num_frames+1)], axis=0).to(device)
                label_speeds=torch.stack([speeds[ij+num_frames-1] for ij in range(j, flow_out.shape[0]-num_frames+1)], axis=0).to(device)

            outer=lstmnet(torch.cat([flow_clip, img_clip], dim=2).to(device))
            pred=speednet(outer).view(-1)
            loss=torch.sum((label_speeds-pred)**2)
            tot_loss+=loss.detach()           
            writer.add_scalar("valid/speed_loss", loss, global_step)
            global_step+=1
            if(sol==1):
                break
        valid_global_step=global_step    
        print("valid epoch: %d, video num: %d, loss: %s"%(epoch, i+1, str(tot_loss)))
        writer.add_scalar("valid/video_speed_loss", tot_loss, epoch*0.25*len(valid_videoloader.dataset)+i)
        # print(len(predicted_speeds), speeds.view(-1)[num_frames:].shape[0], tot_loss)
    # print(predicted_speeds, speeds.view(-1)[num_frames:])



def main():
    global args, save_path
    args = parser.parse_args()
    global device
    if(gpu_id==''):
        device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
    else:
        device = torch.device("cuda:%d"%(gpu_id) if torch.cuda.is_available() else torch.device("cpu"))
    
    print(prefix, device)
    # if(device==torch.device('cpu')):
    #     network_data = torch.load(args.pretrained, map_location='cpu')
    # elif(device==torch.device("cuda:%s"%(gpu_id))):
    # network_data = torch.load(args.pretrained)

    # print("=> using pre-trained model '{}'".format(network_data['arch']))
    # flownet_base = models.__dict__[network_data['arch']](network_data).to(device)
    inception=inception_feature_extractor().to(device)

    flownet_model=optical_flow_extractor().to(device)
    lstmnet=convlstmnet([8,8], 2560, hidden_channels, num_layers).to(device)
    speednet=classification_net(batchNorm=batch_normalization).to(device)

    model_params=list(lstmnet.parameters())+list(speednet.parameters())+list(flownet_model.flownet_post.parameters())
    optimizer=torch.optim.Adam(model_params, lr=init_lr)
    
    
    if(trained_modelfile is not None):
        checkpoint = torch.load('speed_models/%s'%(trained_modelfile))
        start_epoch=checkpoint['epoch']
        global_step=checkpoint['train_globalstep']
        writer=SummaryWriter(log_dir=checkpoint['logdir'])
        flownet_model.flownet_post.load_state_dict(checkpoint['flownetpost_statedict'])
        lstmnet.load_state_dict(checkpoint['lstmnet_statedict'])
        speednet.load_state_dict(checkpoint['speednet_statedict'])
        optimizer.load_state_dict(checkpoint['optimizer_statedict'])
    else:
        global_step=0
        start_epoch=0
        writer=SummaryWriter(log_dir='runs/%s_%s'%(prefix, time))

    optimizer.lr=init_lr/10
    print(init_lr)
    dset=video_loader()
    frame_transform=torchvision.transforms.Compose([flow_transforms.normalize(mean=[0.0,0.0,0.0],std=[255.0,255.0,255.0]),
                                                        # flow_transforms.normalize(mean=[0.411,0.432,0.45], std=[1,1,1]), 
                                                        flow_transforms.normalize(mean=dset.imgchannel_mean, std=dset.imgchannel_std), 
                                                        # flow_transforms.Resize((299,299)), 
                                                        flow_transforms.ArrayToTensor()])
    size_transforms=A.Compose([
                            A.Crop(0,180,640,360),
                            A.Resize(299,299)
    ])
    pixel_transforms=A.Compose([
                            A.RandomContrast(limit=(0,0.5), p=0.5),
                            A.RandomBrightness(limit=(0.0,0.3), p=0.5),
                            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=15, p=0.5),
                            A.GaussNoise(var_limit=(10,50), p=0.5)
                        #   A.RandomGamma(p=1.0),
                        #   A.GaussNoise(p=1.0),
                        #   A.CLAHE()
    ])

    
    pj=random.Random(5).sample(list(range(len(dset))), len(dset))
    split=int(0.25*len(dset))
    train_dset=Subset(dset, pj[split:])
    valid_dset=Subset(dset, pj[:split])
    train_dset=DatasetTransformer(train_dset, torchvision.transforms.Compose([transform_img2video(A.Compose([pixel_transforms, size_transforms])) , frame_transform]))
    valid_dset=DatasetTransformer(valid_dset, torchvision.transforms.Compose([transform_img2video(size_transforms), frame_transform]))
    # print(len(dset))
    
    # train_sampler = SubsetRandomSampler()
    # valid_sampler = SubsetRandomSampler(pj[:split])

    train_videoloader=DataLoader(train_dset, batch_size=1, num_workers=1, shuffle=True)
    valid_videoloader=DataLoader(valid_dset, batch_size=1, num_workers=1, shuffle=True)
    

    # hidden-channels, num_layers: expt1:(4,3); expt2: (2,3)
    
    # writer=SummaryWriter(log_dir='runs/%s_%s'%(prefix, datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")))
    # for i in range(100):
    #     print(dset.__getitem__(i)[0].shape)
    # train(flownet_model, inception, lstmnet, speednet, train_videoloader, valid_videoloader, optimizer, writer)
    validate(flownet_model, inception, lstmnet, speednet, valid_videoloader, start_epoch, writer)
 
def speed_meanvar(train_videoloader):
    s, s2, num_b=0,0,0
    for i,(frames, speeds) in enumerate(train_videoloader):
        if(i%100==0):
            print(i)
        s+=torch.sum(speeds)
        s2+=torch.sum(speeds**2)
        num_b+=speeds.shape[0]
    mean=s*1.0/num_b
    std=s2*1.0/num_b-mean**2
    print(mean, std)
   
    #     if args.bidirectional:
    #         # feed inverted pair along with normal pair
    #         inverted_input_var = torch.cat([img2, img1]).unsqueeze(0)
    #         input_var = torch.cat([input_var, inverted_input_var])


if __name__ == '__main__':
    valid_global_step=0
    # global global_step
    global_step=0
    gpu_id=''
    batch_normalization=True
    style=2
    start_epoch=0
    # fnppnetparams=[128,256]
    # convlstmparams=[2048+256]
    batch_size=6
    num_frames=25
    hidden_channels=256
    num_layers=4
    lambda_regul=0.00025
    init_lr=5e-5
    time=datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    # trained_modelfile='flownet_convlstm_inceptionpretrained_flownetS_dataaug(crop+morepixel)_numframes_25_hiddenchannels_256_numlayers_4_lambdaregul_0.000250_initlr_0.000050_batchnorm_True_style_2_40'
    trained_modelfile=None
    prefix='inceptionpretrained_flownetS_dataaug(crop+morepixel)_numframes_%d_hiddenchannels_%d_numlayers_%d_lambdaregul_%f_initlr_%f_batchnorm_%s_style_%d_time_%s'%(num_frames, hidden_channels, num_layers, lambda_regul, init_lr, str(batch_normalization), style, time)
    # init_lr=2.5e-6
    # lambda_regul=0.001
    #other global variables
    main()

