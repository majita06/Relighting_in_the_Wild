#ここでinput,base,ours,sabunの動画も出力する

from glob import glob
import cv2
import os
import torch
import model
model.train_baseline = False
import numpy as np
import argparse
import utils
from tqdm import tqdm
parser = argparse.ArgumentParser(description='Relighting images')

parser.add_argument('--in_dir','-i', default='./result/relighting_video/1st/91WwxuTSaYS+cluster88', help='relighting result by 1st stage')
parser.add_argument('--out_dir','-o', default='./result/relighting_video/2nd', help='relighting result by 1st stage')
parser.add_argument('--model_path', '-m', default='./trained_models/model_2nd.pth', help='model path')
parser.add_argument('--gpu', '-g', default=0, type=int, help='GPU ID (negative value means CPU)')
args = parser.parse_args()

indir = args.in_dir
gpu = args.gpu

model_path = args.model_path
input_name = os.path.basename(indir)
human_name = input_name.split('+')[0]
outdir = args.out_dir + '/' + input_name
if not os.path.exists(outdir):
    os.makedirs(outdir)

model = model.CNNAE2ResNet(train=False,albedo_decoder_channels=3)
model.load_state_dict(torch.load(model_path))
if gpu>-1:
    model.to('cuda')


frame_paths = sorted(glob(indir + '/frame_*[!_mask].png'))
mask_paths = sorted(glob('./demo/infer_video/%s/frame_*_mask.png' % human_name))
N_frames = len(frame_paths)
print("Relighting by 2nd stage...")
for i in tqdm(range(N_frames)):
    frame_orig = cv2.imread(frame_paths[i],cv2.IMREAD_COLOR).astype(np.float32)/255.
    mask = cv2.imread(mask_paths[i],cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.
    
    frame_orig = utils.square(frame_orig)
    mask= utils.square(mask)
    frame_orig = cv2.resize(frame_orig,(1024,1024))
    mask = cv2.resize(mask,(1024,1024))

    mask3 = np.stack([mask for i in range(3)],2)
    img_center_n = 2.*frame_orig-1.
    img_center_n = mask3*img_center_n  
    img_center_n = torch.from_numpy(img_center_n.astype(np.float32)).clone().to("cuda").permute(2,0,1)[None,:,:]
    ##########################################
    res = model(img_center_n)
    ##########################################
    res = res.data[0].permute(1,2,0).to("cpu").detach().numpy().copy()

    result = (frame_orig + res).clip(0.,1.)
    cv2.imwrite(outdir + ('/frame_%03d.png' % i), 255*result*mask3)

video_path = outdir + '.mp4'    
files_path = outdir + '/frame_%03d.png'
os.system('ffmpeg -y -r 30 -i ' + files_path + ' -vcodec libx264 -pix_fmt yuv420p -r 60 -loglevel fatal ' + video_path)

