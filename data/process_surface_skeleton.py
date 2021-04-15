"""
This file conduct data proprocessing as below:
    1: read the pointcloud.npz to get the dense point cloud (10w) with correct normals, and its translation and scale
    2: sample points and normals from the dense point cloud
    3: use the translation and scale to transform all of the point cloud to align with the ShapeNetCore.V1
"""
import os
import sys
import numpy as np
import h5py
from plyfile import PlyData, PlyElement
def load_pointcloud(in_file):
    plydata = PlyData.read(in_file)
    vertices = np.stack([
        plydata['vertex']['x'],
        plydata['vertex']['y'],
        plydata['vertex']['z']
    ], axis=1)
    return vertices

sys.path.append('/data1/tang.jiapeng/3d_recon/occupancy_networks')
from im2mesh.utils.libkdtree import KDTree

raw_pointcloud_dir = '/data1/tang.jiapeng/3d_recon/occupancy_networks/data/ShapeNet'
surface_skeleton_dir = '/data1/tang.jiapeng/surface_skeleton'
filelist_dir = './filelists'
out_dir = '/data1/tang.jiapeng/ShapeNetV1_surface_skeleton'
cats = ['04379243', '02958343', '03001627', '02691156', '04256520', '04090263', '03636649', '04530566', \
        '02828884', '03691459', '02933112', '03211117', '04401088']
total_number = 16384

for cat in cats:
    train_lst = os.path.join(filelist_dir, '%s_train.lst'%cat)
    test_lst = os.path.join(filelist_dir, '%s_test.lst'%cat)
    train_lst = open(train_lst, 'r').readlines()
    train_lst = [f.strip() for f in train_lst]
    test_lst = open(test_lst, 'r').readlines()
    test_lst = [f.strip() for f in test_lst]
    mods = train_lst + test_lst

    for mod in mods:
        pointcloud_file = os.path.join(raw_pointcloud_dir, cat, mod, 'pointcloud.npz')
        pointcloud = np.load(pointcloud_file)
        raw_points = pointcloud['points']
        raw_normals = pointcloud['normals']
        raw_loc = pointcloud['loc']
        raw_scale = pointcloud['scale']

        surface_file = os.path.join(surface_skeleton_dir, cat, 'surface', mod+'.ply')
        surface = PlyData.read(surface_file)['vertex'].data
        surface_points = surface.view(np.dtype('float32')).reshape(-1,7)[:,0:3]
        ##because normal smoothing in skeletonization, we need to get correct surface normals by KDTree
        kdtree = KDTree(raw_points)
        _, idx = kdtree.query(surface_points, k=1)
        surface_normals = raw_normals[idx]

        skeleton_file = os.path.join(surface_skeleton_dir, cat, 'skeleton', mod+'.ply')
        skeleton = PlyData.read(skeleton_file)['vertex'].data
        skeleton = skeleton.view(np.dtype('float32')).reshape(-1,7)[:,0:3]

        line_file = os.path.join(surface_skeleton_dir, cat, 'classify_line', mod+'.ply')
        line = PlyData.read(line_file)['vertex'].data
        line = line.view(np.dtype('float32')).reshape(-1,7)[:,0:3]

        square_file = os.path.join(surface_skeleton_dir, cat, 'classify_square', mod+'.ply')
        square = PlyData.read(square_file)['vertex'].data
        square = square.view(np.dtype('float32')).reshape(-1,7)[:,0:3]

        choice = np.random.choice(raw_points.shape[0], total_number)
        sampled_points = (raw_points[choice]).astype('f4')
        sampled_points_v1 = ((sampled_points * raw_scale) + raw_loc).astype('f4')
        sampled_normals_v1 = (raw_normals[choice]).astype('f4')
        surface_points_v1 = ((surface_points * raw_scale) + raw_loc).astype('f4')
        surface_normals_v1 = (surface_normals).astype('f4')
        skeleton_v1 = ((skeleton * raw_scale) + raw_loc).astype('f4')
        line_v1 = ((line * raw_scale) + raw_loc).astype('f4')
        square_v1 = ((square * raw_scale) + raw_loc).astype('f4')

        out_cat_mod_dir = os.path.join(out_dir, cat, mod)
        if not os.path.exists(out_cat_mod_dir):
            os.makedirs(out_cat_mod_dir)
        sample_h5 = os.path.join(out_cat_mod_dir, 'sample.h5')
        f1 = h5py.File(sample_h5, 'w')
        f1.create_dataset('onet_v1_points', data=sampled_points_v1, compression='gzip', compression_opts=4)
        f1.create_dataset('onet_v1_normals', data=sampled_normals_v1, compression='gzip', compression_opts=4)
        f1.create_dataset('surface_v1_points', data=surface_points_v1, compression='gzip', compression_opts=4)
        f1.create_dataset('surface_v1_normals', data=surface_normals_v1, compression='gzip', compression_opts=4)
        f1.close()
        print(sample_h5, sampled_points_v1.shape, sampled_normals_v1.shape, surface_points_v1.shape, surface_normals_v1.shape)

        skeleton_h5 = os.path.join(out_cat_mod_dir, 'skeleton.h5')
        f1 = h5py.File(skeleton_h5, 'w')
        f1.create_dataset('line', data=line_v1, compression='gzip', compression_opts=4)
        f1.create_dataset('square', data=square_v1, compression='gzip', compression_opts=4)
        f1.create_dataset('skeleton',data=skeleton_v1, compression='gzip', compression_opts=4)
        f1.close()
        print(skeleton_h5, line_v1.shape, square_v1.shape, skeleton_v1.shape)