from glob import glob
import numpy as np
import cv2
import os
import model
import torch
import argparse
from tqdm import tqdm
import utils


parser = argparse.ArgumentParser()
parser.add_argument('--in_dir', '-i', default='./data/test_image', help='Input directory')
parser.add_argument('--out_dir', '-o', default='./result/infer_image', help='Output directory')
parser.add_argument('--model_path', '-m', default='./trained_models/model_1st.pth', help='Model path')
parser.add_argument('--gpu', '-g', default=0, type=int, help='GPU ID (negative value means CPU)')
args = parser.parse_args()
indir_path = args.in_dir
gpu = args.gpu

model_path = args.model_path
model = model.CNNAE2ResNet(train=False)
if model_path:
    model.load_state_dict(torch.load(model_path))
if gpu>-1:
    model.to('cuda')    


def infer_light_transport_albedo_and_light(img, mask):
    img = 2.*img-1.  
    img = img.transpose(2,0,1)

    mask3 = mask[None,:,:].repeat(3,axis=0).astype(np.float32)
    mask9 = mask[None,:,:].repeat(9,axis=0).astype(np.float32)

    img = torch.from_numpy(img.astype(np.float32)).clone()
    mask3 = torch.from_numpy(mask3.astype(np.float32)).clone()
    mask9 = torch.from_numpy(mask9.astype(np.float32)).clone()
    if gpu>-1:
        img = img.to('cuda')
        mask3 = mask3.to('cuda')
        mask9 = mask9.to('cuda')
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

img_paths = sorted(glob(indir_path + '/*.jpg'))
mask_paths = sorted(glob(indir_path + '/*_mask.png'))
N_imgs = len(mask_paths)
N_masks = len(mask_paths)
print("Infer images...")
for i in tqdm(range(N_imgs)):
    img_name = os.path.basename(img_paths[i])[:-len('.jpg')]
    outdir_path = args.out_dir + '/' + img_name
    if not os.path.exists(outdir_path):
        os.makedirs(outdir_path)


    img_orig = cv2.imread(img_paths[i], cv2.IMREAD_COLOR).astype(np.float32)/255.
    mask_orig = cv2.imread(indir_path + '/' + img_name + '_mask.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.

    if img_orig.shape[:2] != mask_orig.shape:
        h = min(img_orig.shape[0],mask_orig.shape[0])
        w = min(img_orig.shape[1],mask_orig.shape[1])
        img_orig = cv2.resize(img_orig,(w,h))
        mask_orig = cv2.resize(mask_orig,(w,h))
    img_trim,mask_trim = utils.trim(img_orig,mask_orig)
    img_sq = utils.square(img_trim)
    mask_sq = utils.square(mask_trim)
    img = cv2.resize(img_sq,(1024,1024))
    mask = cv2.resize(mask_sq,(1024,1024))
    # img = img_orig
    # mask = mask_orig

    transport, albedo, light = infer_light_transport_albedo_and_light(img, mask) 
    shading = np.matmul(transport, light).clip(0.,None)
    rendering = (albedo * shading).clip(0.,1.)

    cv2.imwrite(outdir_path+'/'+img_name+'.png', 255.*img)
    cv2.imwrite(outdir_path+'/'+img_name+'_albedo.png', 255 * albedo)
    cv2.imwrite(outdir_path+'/'+img_name+'_shading.png', 255 * shading.clip(0,1))
    cv2.imwrite(outdir_path+'/'+img_name+'_mask.png', 255.*mask)
    np.save(outdir_path+'/'+img_name+'_light.npy', light)
    np.savez_compressed(outdir_path+'/'+img_name+'_transport.npz', T=transport)
    cv2.imwrite(outdir_path+'/'+img_name+'_rendering.jpg', 255 * rendering)
    
