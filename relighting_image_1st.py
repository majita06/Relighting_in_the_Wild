from glob import glob
import numpy as np
import cv2
import os
import argparse
from tqdm import tqdm
import utils




parser = argparse.ArgumentParser()
parser.add_argument('--in_dir', '-i', default='./result/infer_image' , help='Input directory')
parser.add_argument('--out_dir', '-o', default='./result/relighting_image/1st', help='Output directory')
parser.add_argument('--light_path', '-l', default='./data/test_light/cluster32.npy', help='Input directory')
args = parser.parse_args()

indir = args.in_dir
outdir = args.out_dir
if not os.path.exists(outdir):
    os.makedirs(outdir)
light_path = args.light_path


light_name = os.path.splitext(os.path.basename(light_path))[0]

sphere_transport,sphere_mask = utils.make_sphere()




light = np.load(light_path)
img_paths = sorted(glob(indir + '/*'))
N_imgs = len(img_paths)
print("Relighting by 1st stage...")
for img_path in tqdm(img_paths):
    img_name = os.path.basename(img_path)
    save_basepath = outdir + '/' + img_name +'+'+ light_name
    if not os.path.exists(save_basepath):
        os.makedirs(save_basepath)

    albedo_path = glob(img_path + '/*_albedo.png')[0]
    mask_path = glob(img_path + '/*_mask.png')[0]
    transport_path = glob(img_path + '/*_transport.npz')[0]

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.
    albedo = cv2.imread(albedo_path, cv2.IMREAD_COLOR).astype(np.float32)/255.
    transport = np.load(transport_path)['T']

    N_rotations = 128
    theta = 2*np.pi/N_rotations


    for i in range(N_rotations):
        # rotation light 
        light_rotate = utils.sh_rotate(light,theta)
        shading = np.matmul(transport, light_rotate).clip(0.,None)
        rendering = (albedo * shading).clip(0.,1.)
        sphere = np.matmul(sphere_transport,light_rotate).clip(0.,1.)
        np.save(save_basepath+("/frame_%04d.npy" % i), light_rotate)
        cv2.imwrite(save_basepath + ('/frame_%04d.png' % i), 255.*rendering)
        cv2.imwrite(save_basepath + ('/sphere_%04d.png' % i), 255.*utils.black_mask(sphere,sphere_mask))


    video_path = save_basepath + '.mp4'
    files_path = save_basepath + '/frame_%04d.png'
    os.system('ffmpeg -y -r 30 -i ' + files_path + ' -vcodec libx264 -pix_fmt yuv420p -r 60 -loglevel fatal ' + video_path)
    video_path = outdir + '/' + light_name + '.mp4'
    files_path = save_basepath + '/sphere_%04d.png'
    os.system('ffmpeg -y -r 30 -i ' + files_path + ' -vcodec libx264 -pix_fmt yuv420p -r 60 -loglevel fatal ' + video_path)


