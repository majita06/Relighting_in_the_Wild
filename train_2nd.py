import os
import argparse
import cv2
import numpy as np
np.random.seed(202108)
from glob import glob
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import torch_optimizer as optim
import utils

import model as net
net.train_baseline = False

parser = argparse.ArgumentParser(description='Residual train')
parser.add_argument('--max_epochs', default=200, type=int, help='Max number of epochs')
parser.add_argument('--train_dir', '-train', default='./data/train_human_2nd', help='Directory for training input images')
parser.add_argument('--test_dir', '-test', default='./data/test_human_2nd', help='Directory for training input images')
parser.add_argument('--out_dir', '-o', default='./result/output_2nd', help='Directory for output images')
parser.add_argument('--gpu', '-g', default=0, type=int, help='GPU ID (negative value means CPU)')
args = parser.parse_args()


max_epochs = args.max_epochs
train_dir = args.train_dir
test_dir = args.test_dir
outdir = args.out_dir
gpu = args.gpu


if not os.path.exists(outdir):
    os.makedirs(outdir)
loss_train_txt = outdir + "/loss_train.txt"
open(loss_train_txt, 'w').close()
loss_test_txt = outdir + "/loss_test.txt"
open(loss_test_txt, 'w').close()


model_save_path = outdir + '/model_%03d.pth'
opt_save_path = outdir + '/opt_%03d.pth'


net = net.CNNAE2ResNet(albedo_decoder_channels=3)
opt = optim.RAdam(net.parameters(),lr=0.001, betas=(0.5, 0.999))
T_max = 10
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max, eta_min=0.0001)
if gpu>-1:
    net.to('cuda') 

train_paths = sorted(glob(train_dir + '/*/*_rendering.png'))
test_paths = sorted(glob(test_dir + '/*/*_rendering.png'))

N_train_paths = len(train_paths)
N_test_paths = len(test_paths)

for epoch in range(max_epochs):
    print("epoch: %d\n  outdir: %s" % (epoch+1, outdir))

    L_sum = 0

    perm_img = np.random.permutation(N_train_paths)
    pbar = tqdm(total=N_train_paths, desc='  train', ascii=True)
    for i in range(N_train_paths):
        
        img_path = train_paths[perm_img[i]]
        basename = os.path.basename(img_path)[:-len("_rendering.png")]
        
        orig = cv2.imread(train_dir+'/' + basename + '/'+basename+'.png', cv2.IMREAD_COLOR).astype(np.float32)/255.
        rend_1st = cv2.imread(train_dir+'/' + basename + '/' +basename+'_rendering.png', cv2.IMREAD_COLOR).astype(np.float32)/255.
        mask = cv2.imread(train_dir+'/'+ basename + '/' +basename+'_mask.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.
        mask3 = np.stack([mask,mask,mask],2)

        orig = orig*mask3
        input = 2.*rend_1st-1.
        input = mask3 * input 

        orig = torch.from_numpy(orig).clone().to('cuda', dtype=torch.float32)
        rend_1st = torch.from_numpy(rend_1st).clone().to('cuda', dtype=torch.float32)
        input = torch.from_numpy(input).clone().to('cuda', dtype=torch.float32)

        orig = orig.permute(2,0,1)[None,:,:]
        rend_1st = rend_1st.permute(2,0,1)[None,:,:]
        input = input.permute(2,0,1)[None,:,:]

        #########################################
        res = net(input)
        ##########################################
        result = rend_1st + res

        L = F.l1_loss(result, orig)
        # L_sum += L.data


        opt.zero_grad()
        L.backward()
        opt.step()

        pbar.update(1)
    pbar.close()
    scheduler.step()

    # raw_L = (L_sum/N_train_paths).to("cpu").item()

    # with open(loss_train_txt,"a") as f:
    #     f.write(str(epoch) + " ")
    #     f.write(str(raw_L) + "\n")

    ################# TRAIN OUTPUT ##################

    # result = result.data[0].permute(1,2,0)
    # orig = orig.data[0].permute(1,2,0)
    # rend_1st = rend_1st.data[0].permute(1,2,0)

    # result = result.to('cpu').detach().numpy().copy()
    # orig = orig.to('cpu').detach().numpy().copy()
    # rend_1st = rend_1st.to('cpu').detach().numpy().copy()


    # cv2.imwrite(outdir + "/train_2nd_epoch%03d.jpg" % epoch, 255*result.clip(0,1))
    # cv2.imwrite(outdir + "/train_orig_epoch%03d.jpg" % epoch, 255*orig)
    # cv2.imwrite(outdir + "/train_1st_epoch%03d.jpg" % epoch, 255*rend_1st)
    torch.save(net.state_dict(),model_save_path % epoch)

    if (epoch-9)%20==0:
        rmse_sum = 0.
        ssim_sum = 0.

        perm_test_img = np.random.permutation(N_test_paths)
        net.train_dropout = False
        pbar = tqdm(total=N_test_paths, desc='  test', ascii=True)
        for i in range(N_test_paths):
            
            img_path = test_paths[perm_test_img[i]]
            basename = os.path.basename(img_path)[:-len("_rendering.jpg")]
            
            orig = cv2.imread(test_dir+'/'+basename+'.png', cv2.IMREAD_COLOR).astype(np.float32)/255.
            rend_1st = cv2.imread(test_dir+'/'+basename+'_rendering.jpg', cv2.IMREAD_COLOR).astype(np.float32)/255.
            mask = cv2.imread(test_dir+'/'+basename+'_mask.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.
            mask3 = np.stack([mask,mask,mask],2)  
            

            orig = mask3 * orig

            input = 2.*rend_1st-1.
            input = mask3 * input

            with torch.no_grad():
                input = torch.from_numpy(input).clone().to('cuda', dtype=torch.float32)
                input = input.permute(2,0,1)[None,:,:]

                #########################################
                res = net(input)
                res = res.data[0].permute(1,2,0)
                res = res.to("cpu").detach().numpy().copy()
                res3 = np.stack([res,res,res],2)
                ##########################################
                result = (rend_1st + res3).clip(0,1)
                result = mask3 * result
                
                rmse_sum += utils.rmse_w_mask(result,orig,mask)
                result_gray = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
                orig_gray = cv2.cvtColor(orig,cv2.COLOR_BGR2GRAY)
                ssim_sum += ssim(result_gray,orig_gray) 

            pbar.update(1)
        pbar.close()
        net.train_dropout = True

        eval_rmse = rmse_sum/N_test_paths
        eval_ssim = ssim_sum/N_test_paths


        # Loss list
        with open(loss_test_txt,"a") as f:
            f.write(str(epoch) + " ")
            f.write(str(eval_rmse) + " ")
            f.write(str(eval_ssim)+"\n")   

        cv2.imwrite(outdir + "/test_2nd_epoch%03d.jpg" % epoch, 255.*result)
        cv2.imwrite(outdir + "/test_orig_epoch%03d.jpg" % epoch, 255.*orig)
        cv2.imwrite(outdir + "/test_1st_epoch%03d.jpg" % epoch, 255.*rend_1st)
