import argparse
import numpy as np
import random
import tensorflow as tf
import socket
import trimesh
import os
import sys
import h5py
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(BASE_DIR) # model
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'data'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'preprocessing'))
from tensorflow.contrib.framework.python.framework import checkpoint_utils

import models.tf_ops.nn_distance.tf_nndistance as tf_nndistance
import create_file_lst
slim = tf.contrib.slim

parser = argparse.ArgumentParser()
lst_dir, cats, all_cats, raw_dirs = create_file_lst.get_all_info()

parser.add_argument('--truethreshold', type=float, default=1.0, help='if distance smaller than this value, its true')
parser.add_argument('--baselines', action='store_true')
parser.add_argument('--mesh_sample', action='store_true')
###
parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 32]')
parser.add_argument('--log_dir', default='checkpoint/exp_200', help='Log dir [default: log]')
parser.add_argument('--test_lst_dir', default=lst_dir, help='test mesh data list')
parser.add_argument('--num_points_gt', type=int, default=10000, help='Sample Point Number for each obj to test[default: 10000]')
parser.add_argument('--num_points_pred', type=int, default=10000, help='Sample Point Number for each obj to test[default: 10000]')
parser.add_argument('--category', default="all", help='Which single class to train on [default: None]')
parser.add_argument('--view_num', type=int, default=1, help="how many views do you want to create for each obj")
###
parser.add_argument('--cal_dir', type=str, default="", help="target obj directory that needs to be tested")

FLAGS = parser.parse_args()
print('pid: %s'%(str(os.getpid())))
print(FLAGS)


BATCH_SIZE = FLAGS.batch_size
GPU_INDEX = FLAGS.gpu
LOG_DIR = FLAGS.log_dir
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_INDEX

if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
if FLAGS.baselines:
    name = (FLAGS.cal_dir).split('/')[-1] + '_fscore_alignv1'
else:
    name = (FLAGS.cal_dir).split('/')[-3] + '_fscore_alignv1'
if FLAGS.mesh_sample:
    name += '_meshsample%d'%FLAGS.num_points_pred
else:
    name += '_pts%d'%FLAGS.num_points_pred
LOG_FOUT = open(os.path.join(LOG_DIR, '%s.txt'%name), 'a')
LOG_FOUT.write(str(FLAGS)+'\n')

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

# info = {'rendered_dir': raw_dirs["renderedh5_dir"],
#         'sdf_dir': raw_dirs["sdf_dir"],
#         'gt_marching_cube':raw_dirs['norm_mesh_dir']}
# print(info)

sdf_dir = '/data/tang.jiapeng/3d_recon/DISN/preprocessing/SDF_v1'

def build_file_dict(dir, baselines):
    file_dict = {}
    for file in os.listdir(dir):
        full_path = os.path.join(dir, file)
        if os.path.isfile(full_path):
            if baselines:
                obj_id = file[:-4]
            else:
                obj_id = file.split("_")[1]
            if obj_id in file_dict.keys():
                file_dict[obj_id].append(full_path)
            else:
                file_dict[obj_id] = [full_path]
    return file_dict

def cal_f_score_all_cat(cats, pred_dir, gt_dir, test_lst_dir,threshold_lst, side_len, baselines=False):
    precision_lst = []
    recall_lst = []
    cnt_lst = []
    for cat_nm, cat_id in cats.items():
        pred_dir_cat = os.path.join(pred_dir, cat_id)
        gt_dir_cat = os.path.join(gt_dir, cat_id)
        test_lst_f = os.path.join(test_lst_dir, cat_id + "_test.lst")
        thresholds = np.asarray(threshold_lst, dtype=np.float32) * 0.001 * side_len
        precision_avg, recall_avg, cnt \
            = f_score_cat(cat_id, cat_nm, pred_dir_cat, gt_dir_cat, test_lst_f, thresholds, baselines)
        f_score_avg = 2 * (precision_avg * recall_avg) / (precision_avg + recall_avg)
        precision_lst.append(precision_avg)
        recall_lst.append(recall_avg)
        cnt_lst.append(cnt)
        print("{}, {}, precision_avg {}, recal_avg{}, f_score_avg{}, count {}"
              .format(cat_nm, cat_id, precision_avg, recall_avg, f_score_avg, cnt))
        log_string("{}, {}, precision_avg {}, recal_avg{}, f_score_avg{}, count {}"
              .format(cat_nm, cat_id, precision_avg, recall_avg, f_score_avg, cnt))
    print("done!")
    precision = np.asarray(precision_lst) # 13 * 5
    recall = np.asarray(recall_lst)
    pre_w_avg = np.average(precision, axis=0, weights=cnt_lst)
    rec_w_avg = np.average(recall, axis=0, weights=cnt_lst)
    f_score = 2 * (pre_w_avg * rec_w_avg) / (pre_w_avg + rec_w_avg)
    print("pre_w_avg {}, rec_w_avg {}, f_score {}".format(pre_w_avg, rec_w_avg, f_score))
    #log_string("pre_w_avg {}, rec_w_avg {}, f_score {}".format(pre_w_avg, rec_w_avg, f_score))

def f_score_cat(cat_id, cat_nm, pred_dir, gt_dir, test_lst_f, thresholds, baselines=False):
    pred_dict = build_file_dict(pred_dir, baselines)
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            sess = tf.Session(config=config)
            sampled_pc = tf.placeholder(tf.float32, shape=(FLAGS.batch_size + 1, FLAGS.num_points_gt, 3))
            #
            dists_forward_sqrt, dists_backward_sqrt = get_points_distance(sampled_pc)
            count = 0
            precision_sum = 0
            recall_sum = 0
            with open(test_lst_f, "r") as f:
                test_objs = f.readlines()
                for obj_id in test_objs:
                    obj_id = obj_id.rstrip('\r\n')
                    if obj_id not in pred_dict:
                        continue
                    elif len(pred_dict[obj_id]) == 0: #if the predicted mesh do not exists, skip
                        continue
                    pred_pnt_dir = os.path.join(os.path.dirname(pred_dir),
                        "pnt_{}_{}".format(FLAGS.num_points_pred, cat_id))
                    forfl = os.path.join(pred_pnt_dir, "for_dist_{}.txt".format(obj_id))
                    backfl = os.path.join(pred_pnt_dir, "bac_dist_{}.txt".format(obj_id))
                    if not os.path.exists(forfl):
                        src_path = os.path.join(gt_dir, obj_id, "sample.h5")
                        verts_batch = np.zeros((FLAGS.view_num+1, FLAGS.num_points_gt, 3), dtype=np.float32)
                        pointcloud_h5 = h5py.File(src_path, 'r')
                        pointcloud = pointcloud_h5['surface_v1_points'][:].astype(np.float32)
                        # norm_params = pointcloud_h5['norm_params'][:].astype(np.float32)
                        # centroid, m = norm_params[:3], norm_params[3]
                        choice = np.random.randint(pointcloud.shape[0], size=FLAGS.num_points_gt)
                        gt_pnts = pointcloud[choice,...]

                        verts_batch = np.zeros((FLAGS.view_num + 1, FLAGS.num_points_gt, 3), dtype=np.float32)
                        verts_batch[0, ...] = gt_pnts
                        sdf_h5_file = os.path.join(sdf_dir, cat_id, obj_id, 'ori_sample.h5')
                        centroid, m = get_norm_params(sdf_h5_file)

                        pred_path_lst = pred_dict[obj_id]
                        for i in range(len(pred_path_lst)):
                            pred_mesh_fl = pred_path_lst[i]
                            mesh1 = trimesh.load_mesh(pred_mesh_fl)
                            try:
                                if FLAGS.mesh_sample:
                                    num_left = FLAGS.num_points_pred - mesh1.vertices.shape[0]
                                    if num_left>0:
                                        samples = np.zeros((0,3), dtype=np.float32)
                                        points, index = trimesh.sample.sample_surface(mesh1, num_left)
                                        samples = np.append(samples, points, axis=0)
                                        verts_batch[i+1, :FLAGS.num_points_pred] = normalize_points(np.append(samples, mesh1.vertices, axis=0), centroid, m)
                                    else:
                                        #choice = np.random.randint(mesh1.vertices.shape[0], size=FLAGS.num_points_pred)
                                        choice = np.arange(FLAGS.num_points_pred)
                                        vertices = mesh1.vertices[choice, ...]
                                        verts_batch[i+1, :FLAGS.num_points_pred] = normalize_points(vertices, centroid, m)
                                else:
                                    num_left = FLAGS.num_points_pred - mesh1.vertices.shape[0]
                                    if num_left>0:
                                        samples = np.zeros((0,3), dtype=np.float32)
                                        choice = np.random.randint(mesh1.vertices.shape[0], size=num_left)
                                        points = mesh1.vertices[choice, ...]
                                        samples = np.append(samples, points, axis=0)
                                        verts_batch[i+1, :FLAGS.num_points_pred] = normalize_points(np.append(samples, mesh1.vertices, axis=0), centroid, m)
                                    else:
                                        #choice = np.random.randint(mesh1.vertices.shape[0], size=FLAGS.num_points_pred)
                                        choice = np.arange(FLAGS.num_points_pred)
                                        points = mesh1.vertices[choice, ...]
                                        verts_batch[i+1, :FLAGS.num_points_pred] = normalize_points(points, centroid, m)
                            except:
                                continue
                                    
                        if FLAGS.batch_size == FLAGS.view_num:
                            feed_dict = {sampled_pc: verts_batch}
                            dists_forward_sqrt_val, dists_backward_sqrt_val = sess.run([dists_forward_sqrt, dists_backward_sqrt], feed_dict=feed_dict)
                        else:
                            raise NotImplementedError
                        #np.savetxt(forfl, dists_forward_sqrt_val)
                        #np.savetxt(backfl, dists_backward_sqrt_val)
                    else:
                        dists_forward_sqrt_val = np.loadtxt(forfl)
                        dists_backward_sqrt_val = np.loadtxt(backfl)
                    dists_forward_sqrt_val = np.tile(dists_forward_sqrt_val, [thresholds.shape[0], 1])
                    dists_backward_sqrt_val = np.tile(dists_backward_sqrt_val, [thresholds.shape[0], 1])
                    pre_sum_val = np.sum(np.less(dists_forward_sqrt_val, thresholds), axis=1)
                    rec_sum_val = np.sum(np.less(dists_backward_sqrt_val, thresholds), axis=1)
                    precision = pre_sum_val / (dists_forward_sqrt_val.shape[1])
                    recall = rec_sum_val / (dists_backward_sqrt_val.shape[1])
                    f_score = 2 * (precision * recall) / (precision + recall)
                    print(" count:{}, cat_id {}, obj_id {}:\n pre_sum {},\n rec_sum {},\n precision {},\n recall {},\n fscore {}\n"
                          .format(count, cat_id, obj_id, pre_sum_val, rec_sum_val, precision, recall, f_score))
                    precision_sum += precision
                    recall_sum += recall
                    count += 1
    return precision_sum/count, recall_sum/count, count

# def get_points_distance(sampled_pc):
#     src_pc = tf.tile(tf.expand_dims(sampled_pc[0, :, :], axis=0), (FLAGS.batch_size, 1, 1))
#     if sampled_pc.get_shape().as_list()[0] == 2:
#         pred = tf.expand_dims(sampled_pc[1, :, :], axis=0)
#     else:
#         pred = sampled_pc[1:, :, :]
#     print(src_pc.get_shape())
#     print(pred.get_shape())
#     dists_forward, _, dists_backward, _ = tf_nndistance.nn_distance(pred, src_pc)
#     dists_forward_sqrt = tf.sqrt(dists_forward)
#     dists_backward_sqrt = tf.sqrt(dists_backward)
#     dists_forward_sqrt = tf.reshape(dists_forward_sqrt, [-1])
#     dists_backward_sqrt = tf.reshape(dists_backward_sqrt, [-1])
#     return dists_forward_sqrt, dists_backward_sqrt

def get_points_distance(sampled_pc):
    src_pc = tf.tile(tf.expand_dims(sampled_pc[0, :, :], axis=0), (FLAGS.batch_size, 1, 1))
    if sampled_pc.get_shape().as_list()[0] == 2:
        pred = tf.expand_dims(sampled_pc[1, :FLAGS.num_points_pred, :], axis=0)
    else:
        pred = sampled_pc[1:, :FLAGS.num_points_pred, :]
    print(src_pc.get_shape())
    print(pred.get_shape())
    dists_forward, _, dists_backward, _ = tf_nndistance.nn_distance(pred, src_pc)
    dists_forward= tf.reshape(dists_forward, [-1])
    dists_backward = tf.reshape(dists_backward, [-1])
    return dists_forward, dists_backward

import h5py
def get_norm_params(sdf_h5_file):
    with h5py.File(sdf_h5_file, 'r') as h5_f:
        norm_params = h5_f['norm_params'][:]
        center, m, = norm_params[:3], norm_params[3]
    return center, m

def normalize_points(points, center, m):
    return points * m + center[None, :]


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
    elif FLAGS.category == "clean":
        cats ={"cabinet": "02933112",
                "display": "03211117",
                "speaker": "03691459",
                "rifle": "04090263",
                "watercraft": "04530566"
                }
    elif FLAGS.category == "five":
        cats ={
                "airplane": "02691156",
                "bench": "02828884",
                "chair": "03001627",
                "rifle": "04090263",
                "table": "04379243"
                }
    elif FLAGS.category == "clean":
        cats = {
        # "cabinet": "02933112",
        # "display": "03211117",
        # "speaker": "03691459",
        "rifle": "04090263",
        "watercraft": "04530566"
    }
    else:
        cats={FLAGS.category: cats_all[FLAGS.category]}

    cal_f_score_all_cat(cats, FLAGS.cal_dir, '/data/tang.jiapeng/ShapeNetV1_surface_skeleton', \
        FLAGS.test_lst_dir, [[0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9], [1.0], [2.0]], FLAGS.truethreshold, FLAGS.baselines) # th *1e-3