import os
import time
import numpy as np 
import argparse
import openmesh 
import multiprocessing

cat_dicts = {
    "plane": "02691156",
    "car": "02958343",
    "chair": "03001627",
    "table": "04379243",
    "bench": "02828884",
    "couch": "04256520",
    "cellphone": "04401088",
    "watercraft": "04530566",
    "firearm": "04090263",
    "monitor": "03211117",
    "lamp": "03636649",
    "speaker": "03691459",
    "cabinet": "02933112"
    }

parser = argparse.ArgumentParser()
parser.add_argument('--num_processes', type=int, default=10, help='parallelism')
parser.add_argument('--basemesh_dir', type=str, default='')
parser.add_argument('--info_dir', type=str, default='')
FLAGS = parser.parse_args()
n_processes = FLAGS.num_processes
basemesh_dir = args.basemesh_dir
info_dir = args.info_dir

fns = []
for cat in sorted(os.listdir(basemesh_dir)):
    cat_basemesh_dir = os.path.join(basemesh_dir, cat)
    for file in sorted(os.listdir(cat_basemesh_dir)):
        mod = file[:-4]
        fns.append((cat, mod))
print(len(fns))

def read_mesh(cat_mod):
    cat, mod = cat_mod
    t = time.time()
    objfile = os.path.join(basemesh_dir, cat, mod + '.obj')
    cat_info_dir = os.path.join(info_dir, cat)
    if not os.path.exists(cat_info_dir):
        os.makedirs(cat_info_dir)
    outfile = os.path.join(cat_info_dir, mod +'.npz')
    if os.path.exists(outfile):
        print(outfile, 'has exists!!!')
    else:
        trimesh = openmesh.read_trimesh(objfile)
        vertices = trimesh.points()
        vertices = vertices.astype('float32')

        halfedges = trimesh.halfedge_vertex_indices()
        halfedges = halfedges.astype('int32')

        faces = trimesh.face_vertex_indices()
        faces = faces.astype('int32')
        np.savez(outfile, vertices=vertices, edges=halfedges, faces=faces)
        print(outfile, time.time()-t0)

pool = multiprocessing.Pool(processes=n_processes)
for idx,fn in enumerate(fns):
    if n_processes >1:
        pool.apply_async(read_mesh,args=(fn,))
    else:
        read_mesh(fn)

if n_processes >1:
    pool.close()
    pool.join()
