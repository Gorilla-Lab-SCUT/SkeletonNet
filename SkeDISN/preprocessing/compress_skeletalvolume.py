import os
import numpy as np
import h5py

skevoldir = '/data/tang.jiapeng/ShapeNetV1_skeleal_volume'
compress_skevoldir = '/data/tang.jiapeng/ShapeNetV1_skeleal_volume_compress'
if not os.path.exists(compress_skevoldir):
    os.mkdir(compress_skevoldir)

def compress_volume(in_path, out_path):
    voxel = h5py.File(in_path, 'r')['occupancies'][:]
    voxel_compress = np.packbits(voxel).view(np.int64)
    h5 = h5py.File(out_path, 'w')
    h5.create_dataset('occupancies', data=voxel_compress, compression='gzip', compression_opts=4)
    h5.close()

for cat in sorted(os.listdir(skevoldir)):
    cat_skevoldir = os.path.join(skevoldir, cat)
    cat_compress_skevoldir = os.path.join(compress_skevoldir, cat)
    for mod in sorted(os.listdir(cat_skevoldir)):
        cat_mod_skevoldir = os.path.join(skevoldir, cat, mod)
        cat_mod_compress_skevoldir= os.path.join(compress_skevoldir, cat, mod)
        if not os.path.exists(cat_mod_compress_skevoldir):
            os.makedirs(cat_mod_compress_skevoldir)

        ## compress skeletal volume 64
        filename = '64_max_fill.h5'
        in_path = os.path.join(cat_mod_skevoldir, filename)
        out_path = os.path.join(compress_skevoldir, filename)
        compress_volume(in_path, out_path, filename)

        ## compress skeletal volume 128
        filename = '128_max_fill.h5'
        in_path = os.path.join(cat_mod_skevoldir, filename)
        out_path = os.path.join(compress_skevoldir, filename)
        compress_volume(in_path, out_path, filename)

        ## compress skeletal volume 256
        filename = '256_max_fill.h5'
        in_path = os.path.join(cat_mod_skevoldir, filename)
        out_path = os.path.join(compress_skevoldir, filename)
        compress_volume(in_path, out_path, filename)
        print(' %s %s has finishd'%(cat, mod))