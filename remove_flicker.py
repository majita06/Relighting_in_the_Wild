import os
import torch
import torch.nn as nn
import numpy as np
from glob import glob
import model as net
net.train_baseline = False
import argparse
from vgg import VGG19
from tqdm import tqdm
import cv2
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--max_epoch",'-e', default=30, type=int, help="The max number of epochs for training")
parser.add_argument("--in_dir",'-i',default='./data/test_video/sample_frames', type=str, help="dir of input video")
parser.add_argument("--processed",'-p', default='./result/relighting_video/2nd/sample_frames+cluster88', type=str, help="dir of frames with flickering")
parser.add_argument("--light_dir",'-l', default='./result/relighting_video/1st/sample_frames+cluster88', type=str, help="dir of light for relighting")
parser.add_argument("--out_dir",'-o', default='./result/relighting_video/flicker_reduction', type=str, help="dir of output video")
parser.add_argument('--gpu', '-g', default="0", type=str, help='GPU ID (negative value means CPU)')
args = parser.parse_args()

indir = args.in_dir
gpu = args.gpu
if gpu>-1:
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu

processed_dir = args.processed
video_name =  os.path.basename(processed_dir)
human_name = video_name.split('+')[0]
light_name = video_name.split('+')[1]
light_dir = args.light_dir
outdir = args.out_dir + '/' + video_name
maxepoch = args.max_epoch + 1



###### define loss function ######
def normalize_batch(batch):
    # Normalize batch using ImageNet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
    return (batch - mean) / std
def compute_error(real,fake):
    return torch.mean(torch.abs(fake-real))
def Lp_loss(x, y):
    x = (x+1.)/2.
    y = (y+1.)/2.
    vgg_real = VGG_19(normalize_batch(x))
    vgg_fake = VGG_19(normalize_batch(y))
    p0 = compute_error(normalize_batch(x), normalize_batch(y))
    content_loss_list = []
    content_loss_list.append(p0)
    feat_layers = {'conv1_2' : 1./2.6, 'conv2_2' : 1./4.8, 'conv3_2': 1./3.7, 'conv4_2':1./5.6, 'conv5_2':10./1.5}
    for layer, w in feat_layers.items():
        pi = compute_error(vgg_real[layer], vgg_fake[layer])
        content_loss_list.append(w * pi)
    content_loss = torch.sum(torch.stack(content_loss_list))
    return content_loss
##################################


###### Define model ###### 
net = net.CNNAE2ResNet(in_channels=30,albedo_decoder_channels=3)
opt = torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.5, 0.999))
VGG_19 = VGG19(requires_grad=False)
##########################
if gpu>-1:
    net.to('cuda')
    VGG_19.to('cuda')


def prepare_paired_input(id, input_paths, input_light_paths, processed_paths):
    net_img_in = cv2.imread(input_paths[id],cv2.IMREAD_COLOR).astype(np.float32)/255.
    net_img_in = utils.square(net_img_in)
    net_img_in = cv2.resize(net_img_in,(1024,1024))
    mask = cv2.imread(input_paths[id][:-len('.png')] + '_mask.png',cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.
    mask = utils.square(mask)
    mask = cv2.resize(mask,(1024,1024))
    mask3 = np.stack([mask,mask,mask],2)
    net_img_in = net_img_in * mask3
    net_img_in = 2. * net_img_in - 1.

    net_img_in = torch.from_numpy(net_img_in).clone()
    net_img_in = net_img_in.permute(2,0,1)[None,:,:]
    net_light_in = np.load(input_light_paths[id])
    net_light_in = torch.from_numpy(net_light_in).clone() 
    net_light_in = torch.reshape(net_light_in,(-1,27))[None,:].repeat(1024,1024,1)
    net_light_in = net_light_in.permute(2,0,1)[None,:,:]
    
    net_in = torch.cat([net_img_in,net_light_in],1)

    net_gt = cv2.imread(processed_paths[id],cv2.IMREAD_COLOR).astype(np.float32)/255.
    net_gt = 2. * net_gt - 1.
    net_gt = torch.from_numpy(net_gt).clone()
    net_gt = net_gt.permute(2,0,1)[None,:,:]

    return net_in, net_gt

# some functions 
def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):        
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.fill_(1)
            module.bias.data.zero_()


###### load data ###### 
input_paths = sorted(glob(indir + "/frame_*[!_mask].png"))
processed_paths = sorted(glob(processed_dir + "/frame_*.png"))
input_light_paths = sorted(glob(light_dir + "/frame_*.npy"))

N_frames =len(input_paths)


data_in_memory = [None] * N_frames   # [None, None, ..., None]   
for id in range(min(len(input_paths), len(processed_paths))):                           
    net_in,net_gt = prepare_paired_input(id, input_paths,input_light_paths,processed_paths) 
    data_in_memory[id] = [net_in,net_gt]    


#######################
save_training_basepath = '%s/training' % outdir
if not os.path.isdir(save_training_basepath):
    os.makedirs(save_training_basepath)
# model re-initialization 
initialize_weights(net)
step = 0
###### start train ######
for epoch in range(1,maxepoch):
    save_basepath = '%s/%04d' % (outdir, epoch)
    if not os.path.exists(save_basepath):
        os.makedirs(save_basepath)
    pbar = tqdm(total=N_frames, desc="Processing epoch %d" % epoch, ascii=True)
    for id in range(N_frames): 
        
        net_in,net_gt = data_in_memory[id] 
        
        if gpu>-1:
            net_in = net_in.to('cuda')
            net_gt = net_gt.to('cuda')
        ########################
        prediction = net(net_in)
        ########################
        L = Lp_loss(prediction, net_gt)

        opt.zero_grad()
        L.backward()
        opt.step()

        step+=1
        if step % 100 == 0 :
            net_in = net_in.data[0].permute(1,2,0).to('cpu').detach().numpy()
            net_gt = net_gt.data[0].permute(1,2,0).to('cpu').detach().numpy()
            prediction = prediction.data[0].permute(1,2,0).to('cpu').detach().numpy()
            cv2.imwrite(save_training_basepath + '/step%06d_%06d.jpg' % (step, id), 
                        np.uint8(np.concatenate([(net_in[:,:,:3]+1.)/2., (prediction+1.)/2., (net_gt+1.)/2.], axis=1).clip(0,1) * 255.))  
        pbar.update(1)
    pbar.close()    

    ###### test ######
    net.train_dropout = False
    pbar = tqdm(total=N_frames, desc='test', ascii=True)
    for id in range(N_frames):

        net_in,net_gt = data_in_memory[id]
        if gpu>-1:
            net_in = net_in.to('cuda')
            net_in = net_gt.to('cuda')
        #############################
        with torch.no_grad():
            prediction = net(net_in) 
        #############################
        net_in = net_in.data[0].permute(1,2,0).to('cpu').detach().numpy()
        net_gt = net_gt.data[0].permute(1,2,0).to('cpu').detach().numpy()
        prediction = prediction.data[0].permute(1,2,0).to('cpu').detach().numpy()

        cv2.imwrite(save_basepath + '/predictions_%05d.jpg' % id, 
            np.uint8(np.concatenate([(net_in[:,:,:3]+1.)/2., (net_gt+1.)/2.,(prediction+1.)/2.],1).clip(0,1) * 255.))
        cv2.imwrite(save_basepath + '/out_main_%05d.jpg' % id, 
            np.uint8(((prediction+1.)/2).clip(0,1) * 255.))
        pbar.update(1)
    pbar.close()
    ################

    video_path = save_basepath + '.mp4'    
    files_path = save_basepath + '/out_main_%05d.jpg'
    os.system('ffmpeg -y -r 30 -i ' + files_path + ' -vcodec libx264 -pix_fmt yuv420p -r 60 -loglevel fatal ' + video_path)
    video_path = save_basepath + '_compare.mp4'    
    files_path = save_basepath + '/predictions_%05d.jpg'
    os.system('ffmpeg -y -r 30 -i ' + files_path + ' -vcodec libx264 -pix_fmt yuv420p -r 60 -loglevel fatal ' + video_path)
    net.train_dropout = True

