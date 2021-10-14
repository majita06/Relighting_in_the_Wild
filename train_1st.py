from glob import glob
import numpy as np
import cv2
import os
import model
import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse
#from utils_shtools import *
import utils
import sfloss as sfl
np.random.seed(202108)

parser = argparse.ArgumentParser(description='Relighting humans')
parser.add_argument('--max_epochs', default=500, type=int, help='Max number of epochs')

#TRAIN DATA(521) 
parser.add_argument('--train_dir', '-train', default='./data/train_human_1st', help='Directory for training input images')

#TEST DATA(20)
parser.add_argument('--test_dir', '-t', default='./data/test_human_1st', help='Directory for test input images')

#OUTPUT
parser.add_argument('--out_dir', '-o', default='./result/output_1st', help='Directory for output images')

#TRAIN LIGHT(2760)
parser.add_argument('--train_light_dir', '-l0', default="./data/train_light", help='Light directory for training')

#TEST LIGHT(10)
parser.add_argument('--test_light_dir', '-l1', default="./data/test_light", help='Light directory for test')

parser.add_argument('--w_transport', '-tw0', default=1., type=float, help='')
parser.add_argument('--w_albedo', '-tw1', default=1., type=float, help='')
parser.add_argument('--w_light', '-tw2', default=1., type=float, help='')
parser.add_argument('--w_shading_transport', '-tw5', default=1., type=float, help='')
parser.add_argument('--w_shading_light', '-tw6', default=1., type=float, help='')
parser.add_argument('--w_shading_all', '-tw7', default=1., type=float, help='')
parser.add_argument('--w_rendering_albedo', '-tw8', default=1., type=float, help='')
parser.add_argument('--w_rendering_transport', '-tw9', default=1., type=float, help='')
parser.add_argument('--w_rendering_light', '-tw10', default=1., type=float, help='')
parser.add_argument('--w_rendering_albedo_transport', '-tw11', default=1., type=float, help='')
parser.add_argument('--w_rendering_transport_light', '-tw12', default=1., type=float, help='')
parser.add_argument('--w_rendering_albedo_light', '-tw13', default=1., type=float, help='')
parser.add_argument('--w_rendering_all', '-tw14', default=1., type=float, help='')
parser.add_argument('--w_parsing', '-tw15', default=1., type=float, help='')
parser.add_argument('--w_albedo_sf', '-tw16', default=1., type=float, help='')
parser.add_argument('--w_shading_sf', '-tw17', default=1., type=float, help='')

args = parser.parse_args()
max_epoch = args.max_epochs
train_dir = args.train_dir
test_dir = args.test_dir
train_light_dir = args.train_light_dir
test_light_dir = args.test_light_dir
outdir = args.out_dir
w_transport = args.w_transport
w_albedo = args.w_albedo
w_light = args.w_light
w_shading_transport = args.w_shading_transport
w_shading_light = args.w_shading_light
w_shading_all = args.w_shading_all
w_rendering_albedo = args.w_rendering_albedo
w_rendering_transport = args.w_rendering_transport
w_rendering_light = args.w_rendering_light
w_rendering_albedo_transport = args.w_rendering_albedo_transport
w_rendering_transport_light = args.w_rendering_transport_light
w_rendering_albedo_light = args.w_rendering_albedo_light
w_rendering_all = args.w_rendering_all
w_parsing = args.w_parsing
w_albedo_sf = args.w_albedo_sf
w_shading_sf = args.w_shading_sf


if not os.path.exists(outdir):
    os.makedirs(outdir)

loss_train_txt = outdir + '/loss_train.txt'
open(loss_train_txt, 'w').close()

loss_test_txt = outdir + '/loss_test.txt'
open(loss_test_txt, 'w').close()

model_save_path = outdir + '/model_%03d.pth'
opt_save_path = outdir + '/opt_%03d.pth'



train_fpath  = sorted(glob(train_dir+"/*_tex.png"))
train_light_fpath = glob(train_light_dir+'/*.npy')
test_fpath = glob(test_dir+"/*_tex.png")
test_light_fpath = glob(test_light_dir+'/*.npy')
train_parsing_fpath = sorted(glob(train_dir+'/*_parsing.png'))


sf_loss = sfl.SpatialFrequencyLoss(num_channels=3)

model = model.CNNAE2ResNet()
opt = torch.optim.Adam(model.parameters(),lr=0.0002, betas=(0.5, 0.999))
model = model.to("cuda")
torch.backends.cudnn.benchmark = True

def infer_light_transport_albedo_and_light(img, mask):
    with torch.no_grad():
        input = 2. * img - 1.
        mask3 = np.stack([mask,mask,mask],2)
        mask9 = np.stack([mask for i in range(9)],2)

        input = torch.from_numpy(input).clone().to('cuda', dtype=torch.float).permute(2,0,1)[None,:,:,:]
        mask3 = torch.from_numpy(mask3).clone().to('cuda', dtype=torch.float).permute(2,0,1)[None,:,:,:]
        mask9 = torch.from_numpy(mask9).clone().to('cuda', dtype=torch.float).permute(2,0,1)[None,:,:,:]

        input = mask3 * input

        transport_hat, albedo_parsing_hat, light_hat = model(input)

        transport_hat = (mask9 * transport_hat).data[0].permute(1,2,0)
        albedo_hat = (mask3 * albedo_parsing_hat[:,:3,:,:]).data[0].permute(1,2,0)
        light_hat = light_hat.data
        parsing_hat = (mask3[:,:1,:,:] * albedo_parsing_hat[:,3:,:,:]).data[0].permute(1,2,0)
        

        transport_hat = transport_hat.to('cpu').detach().numpy().copy()
        albedo_hat = albedo_hat.to('cpu').detach().numpy().copy()
        light_hat = light_hat.to('cpu').detach().numpy().copy()
        parsing_hat = parsing_hat.to('cpu').detach().numpy().copy()

    return transport_hat, albedo_hat, light_hat, parsing_hat



N_pixel = 1024**2
eps = 1e-6 
def focal_loss(input, target):
    input = input.clamp(eps, 1.-eps)
    target = target.clamp(eps, 1.-eps)
    loss = ((-(1.-input)**2 * target * torch.log(input) \
        - input**2 * (1.-target) * torch.log(1.-input)).sum())/N_pixel
    return loss

N_train_img = len(train_fpath)
N_train_light = len(train_light_fpath)
N_train_total = N_train_img

N_test_img = len(test_fpath)
N_test_light = len(test_light_fpath)
N_test_total = N_test_img*N_test_light
print("Preloading %d lights ..." % (N_train_light + N_test_light))

train_lights = []
train_lights_basename = []
for train_light_path in train_light_fpath:
    light_basename = os.path.basename(train_light_path)
    light = np.load(train_light_path)
    light = torch.from_numpy(light).clone().to("cuda", dtype=torch.float)
    train_lights.append(light)
    train_lights_basename.append(light_basename)

test_lights = []
test_lights_basename = []
for test_light_path in test_light_fpath:
    light_basename = os.path.basename(test_light_path)
    light = np.load(test_light_path)
    light = torch.from_numpy(light).clone().to("cuda", dtype=torch.float)
    test_lights.append(light)
    test_lights_basename.append(light_basename)

L1_list_train = np.empty(0)
L2_list_train = np.empty(0)
L3_list_train = np.empty(0)
L4_list_train = np.empty(0)
L5_list_train = np.empty(0)
L1_list_test = np.empty(0)
L2_list_test = np.empty(0)
L3_list_test = np.empty(0)
L4_list_test = np.empty(0)

b_train=0 


for epoch in range(max_epoch):
    print("epoch: %d\n  outdir: %s" % (epoch+1, outdir))
    
    L_sum = 0.
    L_transport_sum = 0.
    L_albedo_sum = 0.
    L_light_sum = 0.

    perm_img = np.random.permutation(N_train_img)
    perm_light = np.random.permutation(N_train_light)

    pbar = tqdm(total=N_train_total, desc='  train', ascii=True)
    for bi in range(N_train_img):
        for i in perm_img[bi:bi+1]:
            albedo = cv2.imread(train_fpath[i], cv2.IMREAD_COLOR).astype(np.float32) / 255.
            transport = np.load(train_fpath[i][:-len('_tex.png')]+'_transport.npz')['T']
            mask = cv2.imread(train_fpath[i][:-len('_tex.png')]+'_mask.png', cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
            mask3 = np.stack([mask,mask,mask],2)
            mask9 = np.stack([mask for i in range(9)],2)
            parsing = cv2.imread(train_fpath[i][:-len('_tex.png')]+'_parsing.png', cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
            albedo = torch.from_numpy(albedo).clone().to('cuda', dtype=torch.float)
            transport = torch.from_numpy(transport).clone().to('cuda', dtype=torch.float)
            mask = torch.from_numpy(mask).clone().to('cuda', dtype=torch.float)
            mask3 = torch.from_numpy(mask3).clone().to('cuda', dtype=torch.float)
            mask9 = torch.from_numpy(mask9).clone().to('cuda', dtype=torch.float)
            parsing = torch.from_numpy(parsing).clone().to('cuda', dtype=torch.float)

            albedo = albedo.permute(2,0,1)[None,:,:,:]
            transport = transport.permute(2,0,1)[None,:,:,:]
            mask = mask[None,None,:,:]
            mask3 = mask3.permute(2,0,1)[None,:,:,:]
            mask9 = mask9.permute(2,0,1)[None,:,:,:]
            parsing = mask * parsing[None,None,:,:]

            light = train_lights[perm_light[b_train]]

            shading = torch.einsum('bchw,cd->bdhw',transport,light).clip(0,None)
            rendering = (albedo * shading).clip(0,1)


            input = 2. * rendering - 1.
            input = mask3 * input

            ###########################################################
            transport_hat, albedo_parsing_hat, light_hat = model(input)
            ###########################################################
            albedo_hat = (mask3 * albedo_parsing_hat[:,:3,:,:])
            parsing_hat = (mask3[:,:1,:,:] * albedo_parsing_hat[:,3:,:,:])
            transport_hat = mask9 * transport_hat

            L_transport = F.l1_loss(transport_hat, transport)
            L_albedo = F.l1_loss(albedo_hat, albedo)
            L_albedo_sf = sf_loss(albedo_hat, albedo)
            L_parsing = focal_loss(parsing_hat, parsing)
            L_light = F.l1_loss(torch.unsqueeze(light_hat,0), torch.unsqueeze(light,0))

            shading_transport_hat = torch.einsum('bchw,cd->bdhw',transport_hat, light)
            L_shading_transport = F.l1_loss(shading_transport_hat, shading)

            shading_light_hat = torch.einsum('bchw,cd->bdhw',transport, light_hat)
            L_shading_light = F.l1_loss(shading_light_hat, shading)

            shading_all_hat = torch.einsum('bchw,cd->bdhw',transport_hat, light_hat)

            L_shading_all = F.l1_loss(shading_all_hat, shading)
            L_shading_all_sf = sf_loss(shading_all_hat, shading)

            rendering_albedo_hat = (albedo_hat * shading)
            L_rendering_albedo = F.l1_loss(rendering_albedo_hat, rendering)
                    
            rendering_transport_hat = (albedo * shading_transport_hat)
            L_rendering_transport = F.l1_loss(rendering_transport_hat, rendering)
                    
            rendering_light_hat = (albedo * shading_light_hat)
            L_rendering_light = F.l1_loss(rendering_light_hat, rendering)
                    
            rendering_albedo_transport_hat = (albedo_hat * shading_transport_hat)
            L_rendering_albedo_transport = F.l1_loss(rendering_albedo_transport_hat, rendering)
                    
            rendering_transport_light_hat = (albedo * shading_all_hat)
            L_rendering_transport_light = F.l1_loss(rendering_transport_light_hat, rendering)

            rendering_albedo_light_hat = (albedo_hat * shading_light_hat)
            L_rendering_albedo_light = F.l1_loss(rendering_albedo_light_hat, rendering)
                    
            rendering_all_hat = (albedo_hat * shading_all_hat)
            L_rendering_all = F.l1_loss(rendering_all_hat, rendering)
                    
            L = w_transport * L_transport + w_albedo * L_albedo + w_light * L_light +\
                w_shading_transport * L_shading_transport + w_shading_light * L_shading_light + w_shading_all * L_shading_all +\
                w_rendering_albedo * L_rendering_albedo + w_rendering_transport * L_rendering_transport + w_rendering_light * L_rendering_light + \
                w_rendering_albedo_transport * L_rendering_albedo_transport + w_rendering_transport_light * L_rendering_transport_light + w_rendering_albedo_light * L_rendering_albedo_light +\
                w_rendering_all * L_rendering_all + \
                w_parsing * L_parsing + \
                w_albedo_sf * L_albedo_sf + w_shading_sf * L_shading_all_sf
            opt.zero_grad()  
            L.backward()
            opt.step()
                    
            L_sum += L_transport.data + L_albedo.data + L_light.data + \
                L_shading_transport.data + L_shading_light.data + L_shading_all.data + L_rendering_albedo.data + \
                L_rendering_transport.data + L_rendering_light.data + L_rendering_albedo_transport.data + \
                L_rendering_transport_light.data + L_rendering_albedo_light.data + L_rendering_all.data + \
                L_albedo_sf.data + L_shading_all_sf.data
            L_transport_sum += L_transport.data
            L_albedo_sum += L_albedo.data
            L_light_sum += L_light.data

            b_train += 1
            if b_train == N_train_light: 
                b_train = 0

            pbar.update(1)
            
    pbar.close()


    raw_L_sum = (L_sum / N_train_total).to("cpu").item()
    raw_L_transport = (L_transport_sum / N_train_total).to("cpu").item()
    raw_L_albedo = (L_albedo_sum / N_train_total).to("cpu").item()
    raw_L_light = (L_light_sum / N_train_total).to("cpu").item()


    with open(loss_train_txt,"a") as f:
        f.write(str(epoch) + " ")
        f.write(str(raw_L_sum) + " ")
        f.write(str(raw_L_albedo) + " ")
        f.write(str(raw_L_transport) + " ")
        f.write(str(raw_L_light)+"\n")


    if (epoch+1)%10 == 0:
        albedo = cv2.imread(train_fpath[perm_img[0]], cv2.IMREAD_COLOR).astype(np.float32) / 255.
        mask = cv2.imread(train_fpath[perm_img[0]][:-len('_tex.png')]+"_mask.png", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
        mask3 = np.stack([mask,mask,mask],2)
        transport = np.load(train_fpath[perm_img[0]][:-len('_tex.png')]+"_transport.npz")['T']
        light = np.load(train_light_fpath[perm_light[0]])
        light_basename = os.path.basename(train_light_fpath[perm_light[0]])

        shading = np.matmul(transport, light).clip(0,None)
        rendering = (albedo*shading).clip(0,1)

        transport_hat, albedo_hat, light_hat, parsing_hat = infer_light_transport_albedo_and_light(rendering, mask)
        shading_hat = np.matmul(transport_hat, light_hat).clip(0,None)
        rendering_hat = (albedo_hat*shading_hat).clip(0,1)

        cv2.imwrite(outdir + "/train_albedo_epoch%03d.png" % epoch, 255*albedo_hat)
        np.savez_compressed(outdir + "/train_transport_epoch%03d.npz" % epoch, T=transport_hat)
        cv2.imwrite(outdir + '/train_shading_epoch%03d.png' % epoch, 255*shading_hat.clip(0,1))
        cv2.imwrite(outdir + '/train_rendering_epoch%03d.png' % epoch, 255*rendering_hat)
        np.save(outdir + "/train_light_epoch%03d.npy" % epoch, light_hat)
        cv2.imwrite(outdir + "/train_parsing_epoch%03d.png" % epoch, 255*parsing_hat)

        torch.save(model.state_dict(),model_save_path % epoch)
        torch.save(opt.state_dict(),opt_save_path % epoch)
        

        #test
        model.train_dropout = False

        L_sum = 0.
        L_transport_sum = 0.
        L_shading_sum = 0.
        L_albedo_sum = 0.
        L_light_sum = 0.
        L_rendering_sum = 0.

        perm_img = np.random.permutation(N_test_img)
        perm_light = np.random.permutation(N_test_light)
        pbar = tqdm(total=N_test_total, desc='  test', ascii=True)
        for i in range(N_test_img):
            albedo = cv2.imread(test_fpath[i], cv2.IMREAD_COLOR).astype(np.float32) / 255.
            transport = np.load(test_fpath[i][:-len('_tex.png')]+'_transport.npz')['T']
            mask = cv2.imread(test_fpath[i][:-len('_tex.png')]+'_mask.png', cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.

            for j in range(N_test_light):
                light = np.load(test_light_fpath[j])
                shading = np.matmul(transport,light).clip(0,None)
                rendering = (albedo * shading).clip(0,1)
                transport_hat, albedo_hat, light_hat, _ = infer_light_transport_albedo_and_light(rendering, mask)
                shading_hat = np.matmul(transport_hat,light_hat).clip(0,None)
                rendering_hat = (albedo_hat * shading_hat).clip(0,1)

                L_transport_sum += utils.rmse_w_mask(transport_hat,transport,mask)
                L_albedo_sum += utils.rmse_w_mask(albedo_hat,albedo,mask)
                L_light_sum += np.sqrt(np.mean((light_hat-light)**2))
                L_shading_sum += utils.rmse_w_mask(shading_hat,shading,mask)
                L_rendering_sum += utils.rmse_w_mask(rendering_hat,rendering,mask)
            pbar.update(1)
        pbar.close()
            
        with open(loss_test_txt,"a") as f:
            f.write(str(epoch) + " ")
            f.write(str(L_rendering_sum/N_test_total) + " ")
            f.write(str(L_shading_sum/N_test_total) + " ")
            f.write(str(L_albedo_sum/N_test_total) + " ")
            f.write(str(L_light_sum/N_test_total)+" ")
            f.write(str(L_transport_sum/N_test_total)+"\n")
        
        model.train_dropout = True