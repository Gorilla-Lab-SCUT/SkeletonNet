import argparse
import numpy as np
import random
import os
import sys
import h5py
import pymesh
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(BASE_DIR) # model
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'data'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'preprocessing'))
import create_file_lst

parser = argparse.ArgumentParser()
lst_dir, cats, all_cats, raw_dirs = create_file_lst.get_all_info()
parser.add_argument('--name', type=str, default='SkeDISN', help='name of comparative method')
parser.add_argument('--res', type=int, default=64, help="voxelize resolution")
parser.add_argument('--explicit', action='store_true', help='generated mesh files: explicit: modelname.obj, implicit: modelname_index.obj')
###
parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 32]')
parser.add_argument('--cal_dir', type=str, default="", help="target obj directory that needs to be tested")
parser.add_argument('--log_dir', default='checkpoint/exp_200', help='Log dir [default: log]')
parser.add_argument('--test_lst_dir', default=lst_dir, help='test mesh data list')
parser.add_argument('--category', default="all", help='Which single class to train on [default: None]')
parser.add_argument('--view_num', type=int, default=1, help="how many views do you want to create for each obj")

FLAGS = parser.parse_args()
print('pid: %s'%(str(os.getpid())))
print(FLAGS)

BATCH_SIZE = FLAGS.batch_size
GPU_INDEX = FLAGS.gpu
LOG_DIR = FLAGS.log_dir
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_INDEX

if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
outfile_name = FLAGS.name + '_iou_disndata_res%d'%FLAGS.res
LOG_FOUT = open(os.path.join(LOG_DIR, '%s.txt'%outfile_name), 'a')
LOG_FOUT.write(str(FLAGS)+'\n')

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

info = {'rendered_dir': raw_dirs["renderedh5_dir"],
        'sdf_dir': raw_dirs["sdf_dir"],
        'gt_marching_cube':raw_dirs['norm_mesh_dir']}
print(info)

sdf_dir = info['sdf_dir']
norm_mesh_dir = info['gt_marching_cube']

def build_file_dict(dir, explicit):
    file_dict = {}
    for file in os.listdir(dir):
        full_path = os.path.join(dir, file)
        if os.path.isfile(full_path):
            if explicit:
                obj_id = file[:-4]
            else:
                obj_id = file.split("_")[1]
            if obj_id in file_dict.keys():
                file_dict[obj_id].append(full_path)
            else:
                file_dict[obj_id] = [full_path]
    return file_dict

def iou_all(cats, pred_dir, gt_dir, test_lst_dir, explicit=False):
    results = []
    for cat_nm, cat_id in cats.items():
        pred_dir_cat = os.path.join(pred_dir, cat_id)
        gt_dir_cat = os.path.join(gt_dir, cat_id)
        test_lst_f = os.path.join(test_lst_dir, cat_id + "_test.lst")
        iou_avg, best_iou_pred_lst, cnt = iou_cat(cat_id, pred_dir_cat, gt_dir_cat, test_lst_f, explicit)
        print("cat_nm: {}, cat_id: {}, iou_avg: {}".format(cat_nm, cat_id, iou_avg))
        log_string("{}, {}, iou_avg {}, count {}".format(cat_nm, cat_id, iou_avg, cnt)) 
        results.append(iou_avg)
    print("mean iou: {}".format( np.array(results).sum() / float(len(results))) )
    log_string("mean iou: {} ".format( np.array(results).sum() / float(len(results))) )
    print("done!")

def iou_cat(cat_id, pred_dir, gt_dir, test_lst_f, explicit=False):
    pred_dict = build_file_dict(pred_dir, explicit)
    iou_sum = 0.0
    count = 0.0
    best_iou_pred_lst = []
    with open(test_lst_f, "r") as f:
        test_objs = f.readlines()
        for obj_id in test_objs:
            obj_id = obj_id.rstrip('\r\n')
            src_path = os.path.join(gt_dir, obj_id, "isosurf.obj")
            if obj_id not in pred_dict.keys():
                print("skip error obj id, no key:", obj_id)
                continue
            pred_path_lst = pred_dict[obj_id]
            if len(pred_path_lst) == 0:
                print("skip error obj id:", obj_id)
                continue

            pred_path = sorted(pred_path_lst)[0]
            result_iou = iou_pymesh(src_path, pred_path)
            if result_iou>0:
                iou_sum += result_iou
                count += 1
                avg_iou = result_iou
            else:
                continue
            print("cat_id:", cat_id, "count:", int(count), "obj_id iou avg: ", avg_iou)
            print(pred_path)
    return iou_sum / count, best_iou_pred_lst, count

def iou_pymesh(mesh_src, mesh_pred):
    dim=FLAGS.res

    mesh1 = pymesh.load_mesh(mesh_src)
    grid1 = pymesh.VoxelGrid(2./dim)
    grid1.insert_mesh(mesh1)
    grid1.create_grid()

    ind1 = ((grid1.mesh.vertices + 1.1) / 2.4 * dim).astype(np.int)
    v1 = np.zeros([dim, dim, dim])
    v1[ind1[:,0], ind1[:,1], ind1[:,2]] = 1

    mesh2 = pymesh.load_mesh(mesh_pred)
    grid2 = pymesh.VoxelGrid(2./dim)
    grid2.insert_mesh(mesh2)
    grid2.create_grid()

    ind2 = ((grid2.mesh.vertices + 1.1) / 2.4 * dim).astype(np.int)
    v2 = np.zeros([dim, dim, dim])
    v2[ind2[:,0], ind2[:,1], ind2[:,2]] = 1

    print(v1.shape, v2.shape)
    intersection = np.sum(np.logical_and(v1, v2))
    union = np.sum(np.logical_or(v1, v2))
    return float(intersection) / union

if __name__ == "__main__":


############################################################

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
        cats = cats_all
    elif FLAGS.category == "clean":
        cats = {
                "watercraft": "04530566",
                "rifle": "04090263"
                }
    else:
        cats = {FLAGS.category: cats_all[FLAGS.category]}


    iou_all(cats, FLAGS.cal_dir, norm_mesh_dir, FLAGS.test_lst_dir, FLAGS.explicit)
