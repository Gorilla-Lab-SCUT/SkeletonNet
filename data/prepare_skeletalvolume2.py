import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import multiprocessing
from plyfile import PlyData, PlyElement
import mcubes
from voxel2layer import *
import h5py

def load_pointcloud(in_file):
    plydata = PlyData.read(in_file)
    vertices = np.stack([
        plydata['vertex']['x'],
        plydata['vertex']['y'],
        plydata['vertex']['z']
    ], axis=1)
    return vertices

parser = argparse.ArgumentParser()
parser.add_argument('--num_processes', type=int, default=6, help='parallelism')
parser.add_argument('--vx_res', type=int, default=256)
parser.add_argument('--saveobj', action='store_true')
parser.add_argument('--cats', type=str, default='03001627')

FLAGS = parser.parse_args()
id1,id2,id3 = generate_indices(FLAGS.vx_res)
if FLAGS.vx_res == 256:
    maxpool3d = nn.MaxPool3d(kernel_size=5, stride=1, padding=2)
elif FLAGS.vx_res == 64 or FLAGS.vx_res == 128:
    maxpool3d = nn.MaxPool3d(kernel_size=3, stride=1, padding=1)
maxpool3d = maxpool3d.cuda()

raw_pointcloud_dir = './data/raw_surface_pointcloud'
upsample_skeleton_dir = './data/upsample_skeleton'
outroot = './data/ShapeNetV1_skeleal_volume'
filelist_dir = './data/filelists'
# cats = ['04379243', '02958343', '03001627', '02691156', '04256520', '04090263', '03636649', '04530566', \
#         '02828884', '03691459', '02933112', '03211117', '04401088']
#'03001627_04379243_02958343_02691156_04256520_04090263_03636649_04530566_02828884_03691459_02933112_03211117_04401088'
cats = (FLAGS.cats).split('_')

cat_mods = []
for cat in cats:
    train_lst = os.path.join(filelist_dir, '%s_train.lst'%cat)
    test_lst = os.path.join(filelist_dir, '%s_test.lst'%cat)
    train_lst = open(train_lst, 'r').readlines()
    train_lst = [f.strip() for f in train_lst]
    test_lst = open(test_lst, 'r').readlines()
    test_lst = [f.strip() for f in test_lst]
    mods = train_lst + test_lst
    for mod in mods:
        cat_mods.append((cat, mod))

def worker(cat_mod):
    t0 = time.time()
    cat,mod = cat_mod
    ###load loc, scale
    pointcloud_file = os.path.join(raw_pointcloud_dir, cat, mod, 'pointcloud.npz')
    pointcloud = np.load(pointcloud_file)
    raw_loc = pointcloud['loc']
    raw_scale = pointcloud['scale']
    ##load upsampled skeleton points and apply scale and translation
    skeleton_file = os.path.join(upsample_skeleton_dir, cat, mod+'.ply')
    skeleton = PlyData.read(skeleton_file)['vertex'].data
    points = skeleton.view(np.dtype('float32')).reshape(-1,7)[:,0:3]
    points = ((points * raw_scale) + raw_loc).astype('f4')
    print('raw_scale', raw_scale, 'raw_loc', raw_loc)
    print('the number of points:', len(points))
    points = np.require(points,'float32','C')

    #voxelization
    voxel_data = points * FLAGS.vx_res + (FLAGS.vx_res+1.0)/2.0
    # discard voxels that fall outsvx_resdims
    xyz = (voxel_data.T).astype(np.int)
    valid_ix = ~np.any((xyz < 0) | (xyz >= FLAGS.vx_res), 0)
    xyz = xyz[:, valid_ix]
    voxel_data = np.zeros((FLAGS.vx_res, FLAGS.vx_res, FLAGS.vx_res), dtype='bool')
    voxel_data[tuple(xyz)] = True
    print('the number voxels before maxpool:', voxel_data.sum())

    #maxpooling for coarsening
    with torch.no_grad():
        voxel_data = torch.from_numpy(voxel_data[None, None, :, :, :].astype('float32'))
        voxel_data = voxel_data.cuda()
        voxel_max  = maxpool3d(voxel_data)
        voxel_max  = torch.squeeze(voxel_max.data).cpu().numpy()
    print('the number of voxels after maxpool:', voxel_max.sum())

    #hole fill
    shape_layer = encode_shape(voxel_max, num_layers=1, id1=id1, id2=id2, id3=id3)
    voxels = decode_shape(shape_layer, id1=id1, id2=id2, id3=id3)
    print('the number of voxels after holefilled:', voxels.sum())

    outdir = os.path.join(outroot, cat, mod)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if FLAGS.saveobj:
        #ouput .obj
        outfile = os.path.join(outdir, '%d_max_fill.obj'%FLAGS.vx_res)
        vertices, faces = mcubes.marching_cubes(voxels, 0)
        mcubes.export_obj(vertices, faces, outfile)
        print(outfile, vertices.shape, faces.shape, time.time()-t0)
    else:
        # output .npz
        outfile = os.path.join(outdir, '%d_max_fill.h5'%FLAGS.vx_res)
        f1 = h5py.File(outfile, 'w')
        f1.create_dataset('occupancies', data=voxels.astype('bool'), compression='gzip', compression_opts=4)
        f1.close()
        print(outfile, voxels.shape, time.time()-t0)

pool = multiprocessing.Pool(FLAGS.num_processes)
results = []
results = pool.map(worker, cat_mods)
pool.close()
pool.join()
for x in results:
    print(x)
print('HAVE FINISHED!!!')