from glob import glob
import cv2
import os
from numpy.lib.function_base import diff
import torch
import model
model.train_baseline = False
import numpy as np
import argparse
import utils
from tqdm import tqdm


parser = argparse.ArgumentParser()

parser.add_argument('--in_dir','-i', default='./result/relighting_image/1st', help='relighting result by 1st stage')
parser.add_argument('--out_dir','-o',default='./result/relighting_image/2nd', help='relighting result by 1st stage')
parser.add_argument('--model_path', '-m', default='./trained_models/model_2nd.pth', help='model epoch')
parser.add_argument('--gpu', '-g', default=0, type=int, help='GPU ID (negative value means CPU)')
args = parser.parse_args()

indir = args.in_dir
gpu = args.gpu
model_path = args.model_path

model = model.CNNAE2ResNet(train=False,albedo_decoder_channels=3)
model.load_state_dict(torch.load(model_path))
if gpu>-1:
    model.to('cuda')

img_paths = sorted(glob(indir + '/*[!.mp4]'))
print("Relighting by 2nd stage...")
for img_path in tqdm(img_paths):
    input_name = os.path.basename(img_path)
    human_name = input_name.split('+')[0]
    outdir = args.out_dir + '/' + input_name
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    frame_paths = sorted(glob(img_path + '/frame_*[!_mask].png'))
    mask = cv2.imread('./demo/infer_image/%s/%s_mask.png' % (human_name,human_name),cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.
    mask3 = np.stack([mask for i in range(3)],2)
    N_frames = len(frame_paths)
    for i in range(N_frames):
        frame_1st = cv2.imread(frame_paths[i],cv2.IMREAD_COLOR).astype(np.float32)/255.

        img_center_n = 2.*frame_1st-1.
        img_center_n = mask3*img_center_n  
        img_center_n = torch.from_numpy(img_center_n.astype(np.float32)).clone().permute(2,0,1)[None,:,:]
        if gpu>-1:
            img_center_n
        ##########################################
        res = model(img_center_n)
        ##########################################
        res = res.data[0].permute(1,2,0).to('cpu').detach().numpy().copy()
        result = (frame_1st + res).clip(0.,1.)
        
        result_trim,mask_trim = utils.trim(result,mask)
        cv2.imwrite(outdir + ('/frame_%04d.png' % i), 255*utils.black_mask(result_trim,mask_trim))

    video_path = outdir + '.mp4'    
    files_path = outdir + '/frame_%04d.png'
    os.system('ffmpeg -y -r 30 -i ' + files_path + ' -vcodec libx264 -pix_fmt yuv420p -r 60 -loglevel fatal -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" ' + video_path)
