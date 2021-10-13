import numpy as np
import OpenEXR, Imath, array
import cv2
import torch
import os
from time import time
from utils_shtools import *
from pyshtools.rotate import djpi2, SHRotateRealCoef
from pyshtools.expand import SHExpandDH



def cpu_to_cuda(img):
    img = torch.from_numpy(img).clone().to('cuda', dtype=torch.float)
    return img

def cuda_to_cpu(img):
    img = img.to('cpu').detach().numpy().copy()
    return img


st = 0
def start_time():
    global st
    torch.cuda.synchronize()
    st = time()
def end_time():
    torch.cuda.synchronize()
    t = time() - st
    t_str = 'time: %02f sec' % t
    if t >= 100:
        t /= 60.
        t_str = 'time: %02f min' % t
        if t > 100:
            t /= 60.
            t_str = 'time: %02f h' % t
    print(t_str)

def rotation_y_map(envmap,theta):
    _,w,_=envmap.shape
    envmap_rot = np.zeros_like(envmap)
    cc = int((theta/(2*np.pi))*w)
    envmap_rot[:,:w-cc,:] = envmap[:,cc:,:].copy()
    envmap_rot[:,w-cc:,:] = envmap[:,:cc,:].copy()
    return envmap_rot

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_mask(exr_path):
    img_exr = OpenEXR.InputFile(exr_path)
    pix_type = Imath.PixelType(Imath.PixelType.FLOAT)
    a_str= img_exr.channels('RGBA', pix_type)[3]
    alpha = np.array(array.array('f', a_str))

    data_win = img_exr.header()['dataWindow']
    size = (data_win.max.x - data_win.min.x + 1,
            data_win.max.y - data_win.min.y + 1)
    mask = np.array(alpha)
    mask = mask.reshape(size[1], size[0])
    return mask

def rot_vec(vec,theta):
    rot_mat = [[np.cos(theta),-np.sin(theta)],
                [np.sin(theta),np.cos(theta)]]
    vec_aft = np.matmul(rot_mat,vec)
    return vec_aft







def env_rotate_torch(env,theta):
    _,_,_,w=env.shape
    env_rot = torch.zeros_like(env).to('cuda', dtype=torch.float)
    cc = int((theta/(2*np.pi))*w)
    env_rot[:,:,:,:w-cc] = env[:,:,:,cc:].clone()
    env_rot[:,:,:,w-cc:] = env[:,:,:,:cc].clone()
    return env_rot


def trim(img, mask, padding_x=5, padding_y=5):
    mask_ids = np.where(mask>0)
    y_max = min(max(mask_ids[0])+padding_y, img.shape[0])
    y_min = max(min(mask_ids[0])-padding_y, 0)
    x_max = min(max(mask_ids[1])+padding_x, img.shape[1])
    x_min = max(min(mask_ids[1])-padding_x, 0)
    # if (y_max - y_min) % 2 == 1:
    #     y_max -= 1
    # if (x_max - x_min) % 2 == 1:
    #     x_max -= 1
        
    # p = min(x_min,img.shape[1]-x_max)
    # q=int(abs((y_max - y_min)-(x_max - x_min))/2)
    # x_l=x_min-min(p,q)
    # x_r=x_max+min(p,q)
    return img[y_min:y_max,x_min:x_max,:], mask[y_min:y_max,x_min:x_max]

def square(img):
    if len(img.shape) == 3:
        h,w,c = img.shape
    elif len(img.shape) == 2:
        h,w = img.shape
    if h > w:
        vertical = True
        long_side = h
        short_side = w
    else:
        vertical = False
        long_side = w
        short_side = h
    pad = (long_side-short_side)//2

    if len(img.shape) == 3:
        sq = np.zeros((long_side,long_side,c))
        if vertical:
            sq[0:long_side, pad:pad+short_side, :] += img
        else:
            sq[pad:pad+short_side,0:long_side, :] += img
    elif len(img.shape) == 2:
        sq = np.zeros((long_side,long_side))
        if vertical:
            sq[0:long_side, pad:pad+short_side] += img
        else:
            sq[pad:pad+short_side,0:long_side] += img
   

    sq = sq.astype(np.float32)
    return sq


def load_checkpoint(model, optimizer, filename):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))
    return model, optimizer, start_epoch


def white_mask(img,mask):
    mask3 = np.stack([mask for i in range(3)],2)
    img[mask3<0.9] = 1.
    return img
def black_mask(img,mask):
    mask3 = np.stack([mask for i in range(3)],2)
    return img * mask3

def make_sphere():
    img_size = 256
    row = 256
    col = 256
    y = np.linspace(-1, 1, img_size)
    z = np.linspace(1, -1, img_size)
    y, z = np.meshgrid(y, z)
    mag = np.sqrt(y**2 + z**2)
    mask = mag <=1
    x = np.sqrt(1 - (y*mask)**2 - (z*mask)**2)
    x = x * mask
    y = y * mask
    z = z * mask
    normal = np.concatenate((x[...,None], y[...,None], z[...,None]), axis=2)
    normal = np.reshape(normal, (-1, 3))

    norm_X = normal[:,0]
    norm_Y = normal[:,1]
    norm_Z = normal[:,2]
    sphere_transport = np.zeros((normal.shape[0], 9)) #(1024*1024, 9)
    att= np.pi*np.array([1, 2.0/3.0, 1/4.0])
    sphere_transport[:,0] = 0.5/np.sqrt(np.pi)*att[0]
    sphere_transport[:,1] = np.sqrt(3)/2/np.sqrt(np.pi)*norm_Y*att[1]
    sphere_transport[:,2] = np.sqrt(3)/2/np.sqrt(np.pi)*norm_Z*att[1]
    sphere_transport[:,3] = np.sqrt(3)/2/np.sqrt(np.pi)*norm_X*att[1]
    sphere_transport[:,4] = np.sqrt(15)/2/np.sqrt(np.pi)*norm_Y*norm_X*att[2]
    sphere_transport[:,5] = np.sqrt(15)/2/np.sqrt(np.pi)*norm_Y*norm_Z*att[2]
    sphere_transport[:,6] = np.sqrt(5)/4/np.sqrt(np.pi)*(3*norm_Z**2-1)*att[2]
    sphere_transport[:,7] = np.sqrt(15)/2/np.sqrt(np.pi)*norm_X*norm_Z*att[2]
    sphere_transport[:,8] = np.sqrt(15)/4/np.sqrt(np.pi)*(norm_X**2-norm_Y**2)*att[2]
    sphere_transport = np.reshape(sphere_transport, (row,col,9))


    return sphere_transport,mask

SH_DEGREE = 2
dj = djpi2(SH_DEGREE)
def sh_rotate(light,theta):
    L_matrix_R = shtools_sh2matrix(light[:,0], SH_DEGREE)
    L_matrix_G = shtools_sh2matrix(light[:,1], SH_DEGREE)
    L_matrix_B = shtools_sh2matrix(light[:,2], SH_DEGREE)
    rotL_matrix_R = SHRotateRealCoef(L_matrix_R, np.array([0,0,theta]),dj)
    rotL_matrix_G = SHRotateRealCoef(L_matrix_G, np.array([0,0,theta]),dj)
    rotL_matrix_B = SHRotateRealCoef(L_matrix_B, np.array([0,0,theta]),dj)
    light[:,0] = shtools_matrix2vec(rotL_matrix_R)
    light[:,1] = shtools_matrix2vec(rotL_matrix_G)
    light[:,2] = shtools_matrix2vec(rotL_matrix_B)  
    return light 

def normalize_light(light3):
    SH_DEGREE = 2
    coeffs_matrix_R = shtools_sh2matrix(light3[:,0], SH_DEGREE)
    tmp_envMap_R = MakeGridDH(coeffs_matrix_R, lmax=255, lmax_calc=SH_DEGREE, sampling=2, norm=4)
    coeffs_matrix_G = shtools_sh2matrix(light3[:,1], SH_DEGREE)
    tmp_envMap_G = MakeGridDH(coeffs_matrix_G, lmax=255, lmax_calc=SH_DEGREE, sampling=2, norm=4)
    coeffs_matrix_B = shtools_sh2matrix(light3[:,2], SH_DEGREE)
    tmp_envMap_B = MakeGridDH(coeffs_matrix_B, lmax=255, lmax_calc=SH_DEGREE, sampling=2, norm=4)
    tmp_envMap = 0.299*tmp_envMap_R + 0.587*tmp_envMap_G + 0.114*tmp_envMap_B  # from opencv document
    tmp_SH = pyshtools.expand.SHExpandDH(tmp_envMap, sampling=2, lmax_calc=2, norm=4)
    light_gray = shtools_matrix2vec(tmp_SH)
    if light_gray[0] > 1.0 or light_gray[0] < 0.5:
        factor = (np.random.rand()*0.2 + 0.7)/light_gray[0] #np.random.rand()
        for i in range(3):
            light = light3[:,i]
            coeffs_matrix = shtools_sh2matrix(light, SH_DEGREE)
            tmp_envMap = MakeGridDH(coeffs_matrix, lmax=255, lmax_calc=SH_DEGREE, sampling=2, norm=4)
            tmp_envMap = tmp_envMap*factor
            tmp_SH = SHExpandDH(tmp_envMap, sampling=2, lmax_calc=2, norm=4)
            light = shtools_matrix2vec(tmp_SH)
            light3[:,i] = light

    return light3


def rmse_w_mask(a,b,mask):
    maskx = mask
    if a.shape[2] == 3:
        maskx = np.stack([mask,mask,mask],2)
    if a.shape[2] == 9:
        maskx = np.stack([mask for i in range(9)],2)
    mse = np.sum((a[maskx>0]-b[maskx>0])**2)/np.sum(maskx>0)
    rmse = np.sqrt(mse)
    return rmse

def normal2SHbasis(normal_map):
    if not 'torch' in str(normal_map.dtype):
        transport = np.empty((normal_map.shape[0], normal_map.shape[1], 9), dtype=np.float32)
        normal_sqr = normal_map**2
        mask = np.sum(normal_sqr, axis=2) != 0.
        coeff = 3.141593 * 0.282095
        transport[:,:,0] = coeff * mask
        # note: OpenCV records (x,y,z) in BGR format, so x and z should be swapped
        coeff = 2.094395 * 0.488603
        transport[:,:,1:4] = coeff * normal_map[:,:,::-1]
        coeff = 0.785398
        transport[:,:,4] = coeff * normal_map[:,:,2] * normal_map[:,:,1]
        transport[:,:,5] = coeff * normal_map[:,:,1] * normal_map[:,:,0]
        transport[:,:,6] = coeff * normal_map[:,:,0] * normal_map[:,:,2]
        coeff = 0.785398 * 0.315392
        transport[:,:,7] = coeff * (3. * normal_sqr[:,:,0] - 1.)
        coeff = 0.785398
        transport[:,:,8] = coeff * (normal_sqr[:,:,2] - normal_sqr[:,:,1])
    else:
        transport = torch.empty_like(normal_map).repeat_interleave(3,dim=0)
        normal_sqr = normal_map**2
        mask = torch.sum(normal_sqr, axis=2) != 0.
        coeff = 3.141593 * 0.282095
        transport[:,:,0] = coeff * mask
        # note: OpenCV records (x,y,z) in BGR format, so x and z should be swapped
        coeff = 2.094395 * 0.488603
        transport[:,:,1:4] = coeff * normal_map[:,:,::-1]
        coeff = 0.785398
        transport[:,:,4] = coeff * normal_map[:,:,2] * normal_map[:,:,1]
        transport[:,:,5] = coeff * normal_map[:,:,1] * normal_map[:,:,0]
        transport[:,:,6] = coeff * normal_map[:,:,0] * normal_map[:,:,2]
        coeff = 0.785398 * 0.315392
        transport[:,:,7] = coeff * (3. * normal_sqr[:,:,0] - 1.)
        coeff = 0.785398
        transport[:,:,8] = coeff * (normal_sqr[:,:,2] - normal_sqr[:,:,1])
    return transport
