#baselineを使って反射率，伝達マップ，光源，推定(再構成も)

from glob import glob
import numpy as np
import cv2
import os
import model
import torch
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import utils


parser = argparse.ArgumentParser(description='Relighting humans')
parser.add_argument('--in_dir', '-i', default='./data/test_video/VIDEO_FRAME', help='Input directory')
parser.add_argument('--out_dir', '-o', default='./result/infer_frame', help='Output directory')
parser.add_argument('--model_path', '-m', default='./trained_models/model_1st.pth', help='Model path')
args = parser.parse_args()
indir_path = args.in_dir



model_path = args.model_path
model = model.CNNAE2ResNet(train=False)
if model_path:
    model.load_state_dict(torch.load(model_path))
model.train_dropout = False
model.to("cuda")    


def infer_light_transport_albedo_and_light(img, mask):
    img = 2.*img-1.  
    img = img.transpose(2,0,1)

    mask3 = mask[None,:,:].repeat(3,axis=0).astype(np.float32)
    mask9 = mask[None,:,:].repeat(9,axis=0).astype(np.float32)

    img = torch.from_numpy(img.astype(np.float32)).clone().to("cuda")
    mask3 = torch.from_numpy(mask3.astype(np.float32)).clone().to("cuda")
    mask9 = torch.from_numpy(mask9.astype(np.float32)).clone().to("cuda")
    img_batch = img[None,:,:,:].clone()
    mask3_batch = mask3[None,:,:,:].clone()
    mask9_batch = mask9[None,:,:,:].clone()
        
    img_batch = mask3_batch * img_batch

    #########################################################
    res_transport, res_albedo_parsing, res_light = model(img_batch)
    res_albedo = res_albedo_parsing[:,:3,:,:]
    ##########################################################
    
    res_transport = (mask9_batch * res_transport).data[0]
    res_albedo = (mask3_batch * res_albedo).data[0]
    res_light = res_light.data

    res_transport = res_transport.to('cpu').detach().numpy().copy().transpose(1,2,0)
    res_albedo = res_albedo.to('cpu').detach().numpy().copy().transpose(1,2,0)
    res_light = res_light.to('cpu').detach().numpy().copy()

    return res_transport, res_albedo, res_light

#ディレクトリの PATH のみを取得


frame_paths = sorted(glob(indir_path + '/*[!_mask].png'))
mask_paths = sorted(glob(indir_path + '/*_mask.png'))
N_frames = len(mask_paths)

human_name = os.path.basename(indir_path)
outdir_path = args.out_dir + '/' + human_name
if not os.path.exists(outdir_path):
    os.makedirs(outdir_path)


print('Infer frames...')
for i in tqdm(range(N_frames)):
    basename_frame = os.path.basename(frame_paths[i]).split('.')[0]
    img_orig = cv2.imread(frame_paths[i], cv2.IMREAD_COLOR).astype(np.float32)/255.
    mask_orig = cv2.imread(mask_paths[i], cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.
    img_sq = utils.square(img_orig)
    mask_sq = utils.square(mask_orig)
    img = cv2.resize(img_sq,(1024,1024))
    mask = cv2.resize(mask_sq,(1024,1024))

    transport, albedo, light = infer_light_transport_albedo_and_light(img, mask) 
    shading = np.matmul(transport, light).clip(0.,None)
    rendering = (albedo * shading).clip(0.,1.)

    cv2.imwrite(outdir_path+'/'+basename_frame+'.png', 255.*img)
    cv2.imwrite(outdir_path+'/'+basename_frame+'_albedo.png', 255 * albedo)
    cv2.imwrite(outdir_path+'/'+basename_frame+'_mask.png', 255.*mask)
    np.save(outdir_path+'/'+basename_frame+'_light.npy', light)
    np.savez_compressed(outdir_path+'/'+basename_frame+'_transport.npz', T=transport)
    cv2.imwrite(outdir_path+'/'+basename_frame+'_rendering.jpg', 255 * rendering)
    
