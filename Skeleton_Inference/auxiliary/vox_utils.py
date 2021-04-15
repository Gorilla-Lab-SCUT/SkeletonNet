from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F
import sys
#dirname = os.path.dirname(__file__)
sys.path.append(os.path.join('..', 'data'))
from voxel2layer import holefill_cpu
from voxel2layer_torch import holefill_gpu

def eval_iou(prediction, groudtruth, th=0.4):
    prediction = F.softmax(prediction, dim=1)
    prediction = torch.ge(prediction[:, 1, :, :, :], th)
    prediction = prediction.type(torch.cuda.FloatTensor)
    groudtruth = groudtruth.type(torch.cuda.FloatTensor)
    
    inter = torch.min(prediction, groudtruth).sum(3).sum(2).sum(1)
    union = torch.max(prediction, groudtruth).sum(3).sum(2).sum(1)
    batch_iou = inter/union
    return prediction, batch_iou

def eval_iou_pre_rec(prediction, groudtruth, th=0.4):
    prediction = F.softmax(prediction, dim=1)
    prediction = torch.ge(prediction[:, 1, :, :, :], th)
    prediction = prediction.type(torch.cuda.FloatTensor)
    groudtruth = groudtruth.type(torch.cuda.FloatTensor)
    
    inter = torch.min(prediction, groudtruth).sum(3).sum(2).sum(1)
    union = torch.max(prediction, groudtruth).sum(3).sum(2).sum(1)
    batch_iou = inter/union
    
    prediction_sum = prediction.sum(3).sum(2).sum(1)
    gt_sum = groudtruth.sum(3).sum(2).sum(1)
    batch_pre = inter/(prediction_sum+1e-6)
    batch_rec = inter/gt_sum
    return prediction, batch_iou, batch_pre, batch_rec
####################################
def combine_patches(batch_size, output, npatch=4, patch_res=36, padding=2, vx_res=128):
    npatch2 = npatch * npatch
    npatch3 = npatch * npatch * npatch
    mul = int(vx_res / npatch)

    output_softmax = F.softmax(output, dim=1)
    output_softmax = output_softmax.reshape(batch_size, npatch3, 2, patch_res, patch_res, patch_res)
    prediction = torch.zeros((batch_size, vx_res+2*padding, vx_res+2*padding, vx_res+2*padding), \
        dtype=torch.float,device=output.device)
    count = torch.zeros((batch_size, vx_res+2*padding, vx_res+2*padding, vx_res+2*padding), \
        dtype=torch.float,device=output.device)
    for i in range(npatch):
        for j in range(npatch):
            for k in range(npatch):
                x1, x2 = i * mul, i * mul + patch_res
                y1, y2 = j * mul, j * mul + patch_res
                z1, z2 = k * mul, k * mul + patch_res
                patch_prob = output_softmax[:, i*npatch2 + j*npatch + k, 1, :, :, :]
                prediction[:, x1:x2, y1:y2, z1:z2] += patch_prob
                count[:, x1:x2, y1:y2, z1:z2] += torch.ones(patch_prob.size(), dtype=torch.float32, device=output.device)
    avg_prediction = prediction / count
    if padding!=0:
        prediction = avg_prediction[:, padding:-padding, padding:-padding, padding:-padding]
    else:
        prediction = avg_prediction
    return prediction

def eval_iou_res128(batch_size, output, gt, th, npatch=4, patch_res=36, padding=2, vx_res=128):
    prediction = combine_patches(batch_size, output, npatch, patch_res, padding, vx_res)
    prediction = torch.ge(prediction, th)
    prediction = prediction.type(torch.cuda.FloatTensor)
    groudtruth = gt.type(torch.cuda.FloatTensor)

    inter = torch.min(prediction, groudtruth).sum(3).sum(2).sum(1)
    union = torch.max(prediction, groudtruth).sum(3).sum(2).sum(1)
    batch_iou = inter/union
    return prediction, batch_iou

def eval_iou_pre_rec_res256(batch_size, output, gt, th, npatch=4, patch_res=72, padding=4, vx_res=256):
    prediction = combine_patches(batch_size, output, npatch, patch_res, padding, vx_res)
    prediction = torch.ge(prediction, th)
    prediction = prediction.type(torch.cuda.FloatTensor)
    groudtruth = gt.type(torch.cuda.FloatTensor)

    inter = torch.min(prediction, groudtruth).sum(3).sum(2).sum(1)
    union = torch.max(prediction, groudtruth).sum(3).sum(2).sum(1)
    batch_iou = inter/union

    prediction_sum = prediction.sum(3).sum(2).sum(1)
    gt_sum = groudtruth.sum(3).sum(2).sum(1)
    batch_pre = inter/(prediction_sum+1e-6)
    batch_rec = inter/gt_sum
    return prediction, batch_iou, batch_pre, batch_rec


import h5py
def save_voxel_h5py(voxels, outfile):
    f1 = h5py.File(outfile, 'w')
    f1.create_dataset('occupancies', data=voxels.astype('bool'), compression='gzip', compression_opts=4)
    f1.close()
    print(outfile, voxels.shape)

sys.path.append('external')
import libmcubes
import libsimplify
import trimesh
def save_volume_obj(outdir, catname, modname, pred, gt, holefill=True, save_gt_obj=False):
    #save pred.obj
    outdir_cat = os.path.join(outdir, catname)
    if not os.path.exists(outdir_cat):
        os.mkdir(outdir_cat)
    if holefill:
        t0 = time.time()
        print('fill hole before', pred.sum().item(), pred.shape)
        shape_layers_cpu, pred = holefill_cpu(pred.cpu().numpy().astype('bool'))
        shape_layers_cpu = torch.from_numpy(shape_layers_cpu.astype(np.int32)).to(torch.int16)
        print('fill hole cpu after', pred.sum(), time.time()-t0)
        print('fill hole cpu consuem %f s'%(time.time()-t0))
        voxels = pred.astype('bool')
    else:
        voxels = pred.cpu().numpy().astype('bool')
    n_x, n_y, n_z = voxels.shape
    voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))  

    objfile = os.path.join(outdir_cat, modname+'_pred.obj')
    vertices, triangles = libmcubes.marching_cubes(voxels, 0.5)
    vertices = (vertices-0.5)/ n_x - 0.5
    mesh = trimesh.Trimesh(vertices, triangles, vertex_normals=None, process=False)
    #mesh.export(objfile)
    
    objfile = os.path.join(outdir_cat, modname+'_pred_simplify.obj')
    mesh = libsimplify.simplify_mesh(mesh, 10000)
    mesh.export(objfile)

    if save_gt_obj:
        #save gt.obj
        objfile = os.path.join(outdir_cat, modname+'_gt.obj')
        voxels = gt.cpu().numpy().astype('bool')
        n_x, n_y, n_z = voxels.shape
        voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
        vertices, triangles = libmcubes.marching_cubes(voxels, 0.5)
        vertices = (vertices-0.5)/ n_x - 0.5
        mesh = trimesh.Trimesh(vertices, triangles, vertex_normals=None, process=False)
        #mesh.export(objfile)

        objfile = os.path.join(outdir_cat, modname+'_gt_simplify.obj')
        mesh = libsimplify.simplify_mesh(mesh, 10000)
        mesh.export(objfile)

sys.path.append('external')
import libmcubes
import libsimplify
import trimesh
def save_mc_simplify_obj(outdir, catname, modname, pred, gt, holefill=True):
    #save pred.obj
    outdir_cat = os.path.join(outdir, catname)
    if not os.path.exists(outdir_cat):
        os.mkdir(outdir_cat)
    objfile = os.path.join(outdir_cat, modname+'.obj')

    if holefill:
        shape_layers_cpu, pred = holefill_cpu(pred.cpu().numpy().astype('bool'))
        voxels = pred.astype('bool')
    else:
        voxels = pred.cpu().numpy().astype('bool')
    n_x, n_y, n_z = voxels.shape
    voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0)) 
    
    vertices, triangles = libmcubes.marching_cubes(voxels, 0.5)
    vertices = (vertices-0.5)/ n_x - 0.5
    mesh = trimesh.Trimesh(vertices, triangles, vertex_normals=None, process=False)
    
    if len(triangles)>200000:
        nfaces = int(0.05 * len(triangles))
    elif len(triangles)>=100000 and len(triangles)<200000:
        nfaces = int(0.1 * len(triangles))
    else:
        nfaces = int(0.15 * len(triangles))
    mesh = libsimplify.simplify_mesh(mesh, nfaces)
    if len(mesh.faces>20000):
        mesh = libsimplify.simplify_mesh(mesh, 20000)

    mesh.export(objfile)
    print('holefill is ', holefill,objfile)

from plyio import *
def save_skeleton_ply(outdir, catname, modname, points):
    #save skeletal points .ply
    outdir_cat = os.path.join(outdir, catname)
    if not os.path.exists(outdir_cat):
        os.mkdir(outdir_cat)
    plyfile = os.path.join(outdir_cat, modname+'_ske.ply')
    write_ply(filename= plyfile, points=pd.DataFrame((points.cpu().data.squeeze()).numpy()), as_text=True)

def save_skeleton_upsample_ply(outdir, catname, modname, points):
    #save skeletal points .ply
    outdir_cat = os.path.join(outdir, catname)
    if not os.path.exists(outdir_cat):
        os.mkdir(outdir_cat)
    plyfile = os.path.join(outdir_cat, modname+'_ske_up.ply')
    write_ply(filename= plyfile, points=pd.DataFrame((points.cpu().data.squeeze()).numpy()), as_text=True)