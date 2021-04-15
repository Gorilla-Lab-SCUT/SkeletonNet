import argparse
import numpy as np
import random
import torch
import socket
import trimesh
import os
import sys
import h5py
sys.path.append("./auxiliary/")
from config import ROOT_PC, ROO_SPLIT
sys.path.append("./extension/")
import dist_chamfer as ext
distChamfer = ext.chamferDist()

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='3', help='GPU to use [default: GPU 0]')
parser.add_argument('--view_num', type=int, default=1, help="how many views do you want to create for each obj")
parser.add_argument('--num_points_gt', type=int, default=5000, help='Sample Point Number for each obj to test[default: 10000]')
parser.add_argument('--num_points_pred', type=int, default=2600, help='Sample Point Number for each obj to test[default: 10000]')
parser.add_argument('--category', default="chair", help='Which single class to train on [default: None]')
parser.add_argument('--log_dir', default='./skeleton_eval', help='Log dir [default: log]')
parser.add_argument('--cal_dir', type=str, default="", help="target obj directory that needs to be tested")

FLAGS = parser.parse_args()
print('pid: %s'%(str(os.getpid())))
print(FLAGS)

GPU_INDEX = FLAGS.gpu
LOG_DIR = FLAGS.log_dir
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_INDEX
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

name = (FLAGS.cal_dir).split('/')[-1] + '_cd'
name += '_pts%d'%FLAGS.num_points_pred
LOG_FOUT = open(os.path.join(LOG_DIR, '%s.txt'%name), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def build_file_dict(dir):
    file_dict = {}
    for file in sorted(os.listdir(dir)):
        full_path = os.path.join(dir, file)
        if os.path.isfile(full_path):
            if len(file.split('_')) !=2:
                continue
            if file.split('_')[-1]!="pred.ply" and file.split('_')[-1]!="ske.ply":
                continue
            if file.split('_')[-1]=="pred.ply":
                obj_id = file[:-9]
            if file.split('_')[-1]=="ske.ply":
                obj_id = file[:-8]
            print(full_path)
            if obj_id in file_dict.keys():
                file_dict[obj_id].append(full_path)
            else:
                file_dict[obj_id] = [full_path]
    return file_dict

def cd_all(cats, pred_dir, gt_dir, test_lst_dir):
    for cat_nm, cat_id in cats.items():
        pred_dir_cat = os.path.join(pred_dir, cat_id)
        gt_dir_cat = os.path.join(gt_dir, cat_id)
        test_lst_f = os.path.join(test_lst_dir, cat_id+"_test.lst")
        cd_cat(cat_id, cat_nm, pred_dir_cat, gt_dir_cat, test_lst_f)
    print("done!")

def cd_cat(cat_id, cat_nm, pred_dir, gt_dir, test_lst_f):
    pred_dict = build_file_dict(pred_dir)
    sum_cf_loss = 0.
    count = 0
    with open(test_lst_f, "r") as f:
        test_objs = f.readlines()
        for obj_id in test_objs:
            obj_id = obj_id.rstrip('\r\n')
            if obj_id not in pred_dict:
                continue
            elif len(pred_dict[obj_id]) == 0: #if the predicted mesh do not exists, skip
                continue
            src_path = os.path.join(gt_dir, obj_id, "skeleton.h5")
            verts_batch = np.zeros((FLAGS.view_num+1, FLAGS.num_points_gt, 3), dtype=np.float32)
            pointcloud_h5 = h5py.File(src_path, 'r')
            pointcloud = pointcloud_h5['skeleton'][:].astype(np.float32)
            #choice = np.random.randint(pointcloud.shape[0], size=FLAGS.num_points_gt)
            #verts_batch[0, ...] = pointcloud[choice,...]
            verts_batch[0, ...] = pointcloud[:FLAGS.num_points_gt, ...]

            pred_path_lst = pred_dict[obj_id]
            pred_path_lst = random.sample(pred_path_lst, FLAGS.view_num)
            for i in range(len(pred_path_lst)):
                pred_mesh_fl = pred_path_lst[i]
                mesh1 = trimesh.load_mesh(pred_mesh_fl)
                num_left = FLAGS.num_points_pred - mesh1.vertices.shape[0]
                if num_left>0:
                    samples = np.zeros((0,3), dtype=np.float32)
                    choice = np.random.randint(mesh1.vertices.shape[0], size=num_left)
                    points = mesh1.vertices[choice, ...]
                    samples = np.append(samples, points, axis=0)
                    verts_batch[i+1, :FLAGS.num_points_pred] = np.append(samples, mesh1.vertices, axis=0)
                else:
                    choice = np.arange(FLAGS.num_points_pred)
                    points = mesh1.vertices[choice, ...]
                    verts_batch[i+1, :FLAGS.num_points_pred] = points
            avg_cf_loss_val = get_chamfer_distance(verts_batch)
            sum_cf_loss += avg_cf_loss_val
            print(str(count) + " ", "cat_id {}, obj_id {}, avg cf:{}".format(cat_id, obj_id, str(avg_cf_loss_val)))
            count +=1
    print("cat_nm:{}, cat_id:{}, avg_cf:{}, count:{}".format(cat_nm, cat_id, sum_cf_loss/count, count))
    log_string("cat_nm:{}, cat_id:{}, avg_cf:{}, count:{}".format(cat_nm, cat_id, sum_cf_loss/count, count))

def get_chamfer_distance(sampled_pc):
    src_pc = np.tile(np.expand_dims(sampled_pc[0,:,:], axis=0), (FLAGS.view_num, 1, 1))
    pred = sampled_pc[1:, :FLAGS.num_points_pred, :]

    pred, src_pc = torch.from_numpy(pred).float(), torch.from_numpy(src_pc).float()
    pred, src_pc = pred.cuda(), src_pc.cuda()
    dist1, dist2 = distChamfer(pred, src_pc)
    dist1, dist2 = dist1.data.cpu().numpy(), dist2.data.cpu().numpy()
    cf_loss_views = (np.mean(dist1, axis=1) + np.mean(dist2, axis=1)) * 1000
    avg_cf_loss = np.mean(cf_loss_views)
    return avg_cf_loss


if __name__ == "__main__":
    cats_all = {
        "watercraft": "04530566",
        "rifle": "04090263",
        "display": "03211117",
        "lamp": "03636649",
        "speaker": "03691459",
        "chair": "03001627",
        "bench": "02828884",
        "cabinet": "02933112",
        "car": "02958343",
        "airplane": "02691156",
        "sofa": "04256520",
        "table": "04379243",
        "phone": "04401088"
    }
    if FLAGS.category == "all":
        cats=cats_all

    elif FLAGS.category == "four":
        cats ={
                "airplane": "02691156",
                "chair": "03001627",
                "car": "02958343",
                "table": "04379243"
            }
    else:
        cats={FLAGS.category: cats_all[FLAGS.category]}
    cd_all(cats, FLAGS.cal_dir, ROOT_PC, ROO_SPLIT)