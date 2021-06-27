import argparse
from datetime import datetime
import numpy as np
import random
import tensorflow as tf
import socket
import os
import sys
import h5py
import struct
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR) # model
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'cam_est'))
sys.path.append(os.path.join(BASE_DIR, 'data'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'preprocessing'))
from tensorflow.contrib.framework.python.framework import checkpoint_utils
import model_add_skevox_pix3d_alignv1 as model
import model_cam as model_cam
from concurrent.futures import ThreadPoolExecutor
import create_file_lst
import cv2
slim = tf.contrib.slim
lst_dir, cats, all_cats, raw_dirs = create_file_lst.get_all_info()

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='checkpoint/exp_200', help='Log dir [default: log]')
parser.add_argument('--category', default="all", help='Which single class to train on [default: None]')
#define the network
parser.add_argument('--img_h', type=int, default=137, help='Image Height')
parser.add_argument('--img_w', type=int, default=137, help='Image Width')
parser.add_argument('--num_classes', type=int, default=1024, help='vgg dim')
parser.add_argument('--sdf_res', type=int, default=256, help='sdf grid')
parser.add_argument('--num_points', type=int, default=1, help='Point Number [default: 2048]')
parser.add_argument('--num_sample_points', type=int, default=1, help='Sample Point Number [default: 2048]')
#network architecture
###
parser.add_argument('--restore_modeldisn', default='', help='restore_model')
parser.add_argument('--ske_local_patch', action='store_true')
parser.add_argument('--ske_local_patch_share', action='store_true')
###
parser.add_argument('--threedcnn', action='store_true')
parser.add_argument('--img_feat_onestream', action='store_true')
parser.add_argument('--img_feat_twostream', action='store_true')
parser.add_argument('--binary', action='store_true')
parser.add_argument('--tanh', action='store_true')
#training paradiagm: nepochs, lr, lr_decay, bn, bn_decay, optimizer, restore_model
parser.add_argument('--max_epoch', type=int, default=1, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate [default: 0.001]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.9, help='Decay rate for lr decay [default: 0.7]')
#dataset: if use cam_est , data_augmentation, if rot, cat_limit, if multi_view
parser.add_argument('--cam_est', action='store_true', help="if you are using the estimated camera image h5")
parser.add_argument('--alpha', action='store_true')
parser.add_argument('--augcolorfore', action='store_true')
parser.add_argument('--augcolorback', action='store_true')
parser.add_argument('--backcolorwhite', action='store_true')
parser.add_argument('--grayscale', action='store_true')  ###same as implcit decoder
parser.add_argument('--rot', action='store_true')
parser.add_argument('--cat_limit', type=int, default=12000, help="balance each category, 2000 * 6 = 12000")
parser.add_argument('--multi_view', action='store_true')
#
parser.add_argument('--test_lst_dir', default=lst_dir, help='test mesh data list')
parser.add_argument('--mask_tp', type=str, default="neg_two_sides")
parser.add_argument('--mask_rt', type=int, default=40000)
parser.add_argument('--iso', type=float, default=0.0, help='iso value')
parser.add_argument('--create_obj', action='store_true', help="create_obj or test accuracy on test set")
parser.add_argument('--view_num', type=int, default=1, help="how many views do you want to create for each obj")
##
##
parser.add_argument('--use_predvox', action='store_true')
parser.add_argument('--predvox_dir', type=str, default='gen_4cats_local')
parser.add_argument('--test_allset', action='store_true')
parser.add_argument('--threshold', type=float, default=0.5, help='threshold for binarization')
parser.add_argument('--cat', type=str, default="", help='chair')
parser.add_argument('--demo_image_path', type=str, default="", help='demo_image_path')
#
parser.add_argument('--in1out0', action='store_true')

FLAGS = parser.parse_args()
print('pid: %s'%(str(os.getpid())))
print(FLAGS)

EPOCH_CNT = 0
NUM_POINTS = FLAGS.num_points
BATCH_SIZE = FLAGS.batch_size
RESOLUTION = FLAGS.sdf_res+1
TOTAL_POINTS = RESOLUTION * RESOLUTION * RESOLUTION
if FLAGS.img_feat_twostream:
    SPLIT_SIZE = int(np.ceil(TOTAL_POINTS / 200000.0))
else:
    SPLIT_SIZE = int(np.ceil(TOTAL_POINTS / 274625.0))
NUM_SAMPLE_POINTS = int(np.ceil(TOTAL_POINTS / SPLIT_SIZE))+1
GPU_INDEX = FLAGS.gpu
PRETRAINED_MODEL_PATH = FLAGS.log_dir
LOG_DIR = FLAGS.log_dir
SDF_WEIGHT = 10.

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_INDEX

if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

if FLAGS.cam_est:
    RESULT_OBJ_PATH = os.path.join("./demo/pix3d_add_skevox/", FLAGS.cat)
    print("RESULT_OBJ_PATH: ",RESULT_OBJ_PATH)
else:
    if FLAGS.ske_local_patch or FLAGS.ske_local_patch_share:
        RESULT_OBJ_PATH = os.path.join("./demo/pix3d_add_skevox/", FLAGS.cat)
        print("RESULT_OBJ_PATH: ",RESULT_OBJ_PATH)
    else:
        RESULT_OBJ_PATH = os.path.join("./demo/pix3d_no_skevox/", FLAGS.cat)
        print("RESULT_OBJ_PATH: ",RESULT_OBJ_PATH)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_test.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

IMG_SIZE = FLAGS.img_h
HOSTNAME = socket.gethostname()
print("HOSTNAME:", HOSTNAME)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


if FLAGS.threedcnn:
    info = {'rendered_dir': raw_dirs["renderedh5_dir_v2"],
            'sdf_dir': raw_dirs["3dnnsdf_dir"]}
elif FLAGS.img_feat_onestream or FLAGS.img_feat_twostream:
    info = {'rendered_dir': raw_dirs["renderedh5_dir"],
            'sdf_dir': raw_dirs["sdf_dir"]}
    #if FLAGS.cam_est:
    #    info['rendered_dir']= raw_dirs["renderedh5_dir_est"]
else:
    info = {'rendered_dir': raw_dirs["renderedh5_dir_v2"],
            'sdf_dir': raw_dirs['sdf_dir_v2']}

# cam_gt=[326.421594487, 29.0316186116, 0, 0.790311739218, 25]

def create():
    log_string(LOG_DIR)

    batch_data = read_img_get_transmat()

    input_pls = model.placeholder_inputs(BATCH_SIZE, NUM_POINTS, (IMG_SIZE, IMG_SIZE),
                        num_sample_pc=NUM_SAMPLE_POINTS, scope='inputs_pl', FLAGS=FLAGS)
    is_training_pl = tf.placeholder(tf.bool, shape=())
    print(is_training_pl)
    batch = tf.Variable(0, name='batch')

    print("--- Get model and loss")
    # Get model and loss

    end_points = model.get_model(input_pls, NUM_POINTS, is_training_pl, bn=False,FLAGS=FLAGS)

    loss, end_points = model.get_loss(end_points,
        sdf_weight=SDF_WEIGHT, num_sample_points=NUM_SAMPLE_POINTS, FLAGS=FLAGS)
    # Create a session
    gpu_options = tf.GPUOptions() # per_process_gpu_memory_fraction=0.99
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)

    init = tf.global_variables_initializer()
    sess.run(init)

    ######### Loading Checkpoint ###############
    saver = tf.train.Saver([v for v in tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES) if
                            ('lr' not in v.name) and ('batch' not in v.name)])
    ckptstate = tf.train.get_checkpoint_state(PRETRAINED_MODEL_PATH)
    if FLAGS.img_feat_twostream:
        load_model(sess, FLAGS.restore_modeldisn, ['sdfprediction', 'fold1', 'fold2', 'sdfprediction_imgfeat', 'vgg_16'], strict=True)

    if ckptstate is not None:
        LOAD_MODEL_FILE = os.path.join(PRETRAINED_MODEL_PATH, os.path.basename(ckptstate.model_checkpoint_path))
        try:
            # load_model(sess, PRETRAINED_PN_MODEL_FILE, ['refpc_reconstruction','sdfprediction','vgg_16'], strict=True)
            if FLAGS.ske_local_patch_share or FLAGS.ske_local_patch:
                load_model(sess, LOAD_MODEL_FILE, ['local_patch_encode','sdfprediction_voxfeat'], strict=True)
            with NoStdStreams():
                saver.restore(sess, LOAD_MODEL_FILE)
            print("Model loaded in file: %s" % LOAD_MODEL_FILE)
        except:
            print("Fail to load overall modelfile: %s" % PRETRAINED_MODEL_PATH)
    ###########################################

    ops = {'input_pls': input_pls,
           'is_training_pl': is_training_pl,
           'loss': loss,
           'step': batch,
           'end_points': end_points}

    test_one_epoch(sess, ops, batch_data)


class NoStdStreams(object):
    def __init__(self,stdout = None, stderr = None):
        self.devnull = open(os.devnull,'w')
        self._stdout = stdout or self.devnull or sys.stdout
        self._stderr = stderr or self.devnull or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush(); self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush(); self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()

def load_model(sess, LOAD_MODEL_FILE, prefixs, strict=False):

    vars_in_pretrained_model = dict(checkpoint_utils.list_variables(LOAD_MODEL_FILE))
    # print(vars_in_pretrained_model)
    vars_in_defined_model = []

    for var in tf.trainable_variables():
        if isinstance(prefixs, list):
            for prefix in prefixs:
                if (var.op.name.startswith(prefix)) and (var.op.name in vars_in_pretrained_model.keys()) and ('logits' not in var.op.name):
                    if (list(var.shape) == vars_in_pretrained_model[var.op.name]):
                        vars_in_defined_model.append(var)
        else:     
            if (var.op.name.startswith(prefixs)) and (var.op.name in vars_in_pretrained_model.keys()) and ('logits' not in var.op.name):
                if (list(var.shape) == vars_in_pretrained_model[var.op.name]):
                    vars_in_defined_model.append(var)
    saver = tf.train.Saver(vars_in_defined_model)
    try:
        saver.restore(sess, LOAD_MODEL_FILE)
        print( "Model loaded in file: %s" % (LOAD_MODEL_FILE))
    except:
        if strict:
            print( "Fail to load modelfile: %s" % LOAD_MODEL_FILE)
            return False
        else:
            print( "Fail loaded in file: %s" % (LOAD_MODEL_FILE))
            return True

    return True

def cam_evl(img_arr):
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            input_pls = model_cam.placeholder_inputs(1, NUM_POINTS, (IMG_SIZE, IMG_SIZE), num_pc=NUM_POINTS,
                                                 num_sample_pc=1, scope='inputs_pl')
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0, name='batch')

            print("--- cam Get model_cam and loss")
            # Get model and loss

            end_points = model_cam.get_model(input_pls, NUM_POINTS, is_training_pl, img_size=(IMG_SIZE, IMG_SIZE), bn=False, wd=2e-3, FLAGS=FLAGS)
            loss, end_points = model_cam.get_loss(end_points, sdf_weight=SDF_WEIGHT, FLAGS=FLAGS)
            tf.summary.scalar('loss', loss)

            print("--- Get training operator")
            # Get training operator

            # Create a session
            config = tf.ConfigProto()
            gpu_options = tf.GPUOptions()  # per_process_gpu_memory_fraction=0.99)
            config = tf.ConfigProto(gpu_options=gpu_options)
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            sess = tf.Session(config=config)
            # Init variables
            init = tf.global_variables_initializer()
            sess.run(init)

            ######### Loading Checkpoint ###############

            saver = tf.train.Saver([v for v in tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES) if
                                    ('lr' not in v.name) and ('batch' not in v.name)])
            ckptstate = tf.train.get_checkpoint_state(FLAGS.cam_log_dir)

            if ckptstate is not None:
                LOAD_MODEL_FILE = os.path.join(FLAGS.cam_log_dir, os.path.basename(ckptstate.model_checkpoint_path))
                try:
                    with NoStdStreams():
                        saver.restore(sess, LOAD_MODEL_FILE)
                    print("model_cam loaded in file: %s" % LOAD_MODEL_FILE)
                except:
                    print("Fail to load overall modelfile: %s" % LOAD_MODEL_FILE)
            ###########################################

            ops = {'input_pls': input_pls,
                   'is_training_pl': is_training_pl,
                   'step': batch,
                   'end_points': end_points}

            is_training = False
            batch_data = img_arr

            feed_dict = {ops['is_training_pl']: is_training,
                         ops['input_pls']['imgs']: batch_data}

            pred_trans_mat_val = sess.run(ops['end_points']['pred_trans_mat'], feed_dict=feed_dict)
            print("pred_trans_mat_val", pred_trans_mat_val)
            return pred_trans_mat_val


def project(data, points, newimg_size):
    import numpy as np
    p_3d = np.concatenate((points, np.ones((points.shape[0], 1), dtype=points.dtype)), axis=1)
    p_3d = p_3d.T #4xn
    RT = np.append(data['rot_mat'], np.resize(data['trans_mat'], [3, 1]), axis=1)
    # Here we convert focal length in mm into focal length in pixels.
    f_pix = data['focal_length'] / 32. * data['img_size'][0]
    width, height = newimg_size[0], newimg_size[1]
    K = np.array([[f_pix, 0., width / 2.0], [0, f_pix, height / 2.0], [0., 0., 1.]])
    p_2d = np.dot(np.dot(K, RT), p_3d)
    p_2d = p_2d / p_2d[2, :]
    # Convert u, v into conventional image coordinates (from +u: leftward, +v: upward to +u: rightward, +v: downward.
    p_2d = np.array(newimg_size)[:, None] - p_2d[:2, :]
    return p_2d

import openmesh
import trimesh
import json
def read_img_get_transmat():
    img_file = FLAGS.demo_image_path #"./demo/03001627_17e916fc863540ee3def89b32cef8e45_20.png"

    data_list = json.load(open('/data/tang.jiapeng/pix3d/pix3d.json'))
    K, RT, norm_mat= None, None, None
    center, m = None, None
    batch_vox256_reshape64, batch_vox128_reshape32, batch_vox64_reshape16 = None, None, None
    for i in range(len(data_list)):
        data = data_list[i]
        cat  = data['category']
        img_path = data['img']
        _, _, name = img_path.split('/')
        idx, _ = name.split('.')
        if cat == FLAGS.cat and img_path[:-4] == FLAGS.demo_image_path[:-4]:
            mask_path = data['mask']
            model_path = data['model']
            masked_img_path = os.path.join('/data/tang.jiapeng/pix3d/masked_img', cat, idx+'.png')
            rot_mat = np.array(data['rot_mat'])
            trans_mat = np.array(data['trans_mat'])
            cam_pos = data['cam_position']
            img_size = data['img_size']
            focal_len = data['focal_length']
            w, h = img_size[0], img_size[1]
            #img_arr = cv2.imread(masked_img_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)[:, :, :3]

            mask = cv2.imread('/data/tang.jiapeng/pix3d/'+mask_path)
            img = cv2.imread('/data/tang.jiapeng/pix3d/'+img_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)[:, :, :3]
            print(model_path)
            print(img.shape, mask.shape)
            if img.shape[0]!=mask.shape[0] or img.shape[1]!=mask.shape[1]:
                h = img.shape[0]
                w = img.shape[1]
            mask = cv2.resize(mask, (w,h), interpolation = cv2.INTER_AREA) / 255 ##before no 255
            img_arr = img * mask

            rot_mat = np.array([[0, 0, 1, 0],
                                [0, 1, 0, 0],
                                [-1, 0, 0, 0],
                                [0, 0, 0, 1]], dtype='float32')

            RT = np.append(data['rot_mat'], np.resize(data['trans_mat'], [3, 1]), axis=1) #3x4
            if w>h:
                newimg_arr = np.zeros((w, w, 3), dtype='f4')
                start = int((w - h) / 2.0)
                newimg_arr[start: start+h, 0:w, :] = img_arr[0:h, 0:w, :]
                newimg_size = [w, w]
                f_pix = data['focal_length'] / 32. * 137.0
            else:
                newimg_arr = np.zeros((h, h, 3), dtype='f4')
                start = int((h - w) / 2.0)
                newimg_arr[0:h, start:start+w, :] = img_arr[0:h, 0:w, :]
                newimg_size = [h, h]
                f_pix = data['focal_length'] / 32. * w * 137.0 / float(h)
            K = np.array([[f_pix, 0., 137 / 2.0], [0, f_pix, 137 / 2.0], [0., 0., 1.]])
            img_arr = cv2.resize(newimg_arr, (137, 137))

            def get_vox64(vox64_dir):
                vox64_file = os.path.join(vox64_dir, '64_max_fill.h5')
                vol64 = h5py.File(vox64_file, 'r')['occupancies'][:]
                vox64 = vol64.astype('float32')#.reshape(64, 64, 64
                return vox64

            def get_vox128(vox128_dir):
                vox128_file = os.path.join(vox128_dir, '128_max_fill.h5')
                vol128 = h5py.File(vox128_file, 'r')['occupancies'][:]
                vox128 = vol128.astype('float32')#.reshape(128, 128, 128)
                return vox128

            def get_vox256(vox256_dir):
                vox256_file = os.path.join(vox256_dir, '256_max_fill.h5')
                vol256 = h5py.File(vox256_file, 'r')['occupancies'][:]
                vox256 = (vol256).astype('float32')#.reshape(256, 256, 256)
                return vox256
            vox_dir = '/data/tang.jiapeng/SkeletonBridge-recon/demo/demo_pix3d_%s/%s'%(FLAGS.cat, idx)
            vox64 = get_vox64(vox_dir)
            vox128 = get_vox128(vox_dir)
            vox256 = get_vox256(vox_dir)
            print(vox_dir, vox256.shape)
            #[262144, 4, 4, 4]
            vox256_reshape64 = vox256.reshape(64, 4, 64, 4, 64, 4)
            vox256_reshape64 = np.transpose(vox256_reshape64, (0, 2, 4, 1, 3, 5))
            batch_vox256_reshape64 = vox256_reshape64[None, ...]
            #[32768, 4, 4, 4]
            vox128_reshape32 = vox128.reshape(32, 4, 32, 4, 32, 4)
            vox128_reshape32 = np.transpose(vox128_reshape32, (0, 2, 4, 1, 3, 5))
            batch_vox128_reshape32 = vox128_reshape32[None, ...]
            #[4096,  4, 4, 4]
            vox64_reshape16 = vox64.reshape(16, 4, 16, 4, 16, 4)
            vox64_reshape16 = np.transpose(vox64_reshape16, (0, 2, 4, 1, 3, 5))
            batch_vox64_reshape16 = vox64_reshape16[None, ...]
        else:
            continue
    trans_mat = np.linalg.multi_dot([K, RT, rot_mat])
    batch_img = np.asarray([img_arr.astype(np.float32) / 255.0])
    batch_data = {}
    batch_data['img'] = batch_img
    if FLAGS.cam_est:
        print("here we use our cam est network to estimate cam parameters:")
        batch_data['trans_mat'] = cam_evl(batch_img)
    else:
        print("here we use gt cam parameters")
        batch_data['trans_mat'] = np.transpose(trans_mat)[None, :, :]
    batch_data['vox64_reshape16'] = batch_vox64_reshape16
    batch_data['vox128_reshape32'] = batch_vox128_reshape32
    batch_data['vox256_reshape64'] = batch_vox256_reshape64
    #batch_data['sdf_params'] = np.array([[-1, -1, -1, 1,  1, 1]])
    box_len = 1.0
    batch_data['sdf_params'] = np.array([[-box_len/2.0, -box_len/2.0, -box_len/2.0, box_len/2.0, box_len/2.0, box_len/2.0]])  # only useful if we want to compare it with gt, here random set some value
    return batch_data

def test_one_epoch(sess, ops, batch_data):
    """ ops: dict mapping from string to tf ops """
    is_training = False

    # Shuffle train samples
    log_string(str(datetime.now()))
    losses = {}
    for lossname in ops['end_points']['losses'].keys():
        losses[lossname] = 0

    with ThreadPoolExecutor(max_workers=4) as executor:
        print(SPLIT_SIZE, NUM_SAMPLE_POINTS, SPLIT_SIZE* NUM_SAMPLE_POINTS, TOTAL_POINTS)
        extra_pts = np.zeros((1, SPLIT_SIZE * NUM_SAMPLE_POINTS - TOTAL_POINTS, 3), dtype=np.float32)
        batch_points = np.zeros((SPLIT_SIZE, 0, NUM_SAMPLE_POINTS, 3), dtype=np.float32)
        batch_vox256_patch4 = np.zeros((SPLIT_SIZE, 0, NUM_SAMPLE_POINTS, 4, 4, 4), dtype=np.float32)
        batch_vox128_patch4 = np.zeros((SPLIT_SIZE, 0, NUM_SAMPLE_POINTS, 4, 4, 4), dtype=np.float32)
        batch_vox64_patch4 = np.zeros((SPLIT_SIZE, 0, NUM_SAMPLE_POINTS, 4, 4, 4), dtype=np.float32)
        if not FLAGS.threedcnn:
            for b in range(BATCH_SIZE):
                #print(batch_data)
                sdf_params = batch_data['sdf_params'][b]
                x_ = np.linspace(sdf_params[0], sdf_params[3], num=RESOLUTION)
                y_ = np.linspace(sdf_params[1], sdf_params[4], num=RESOLUTION)
                z_ = np.linspace(sdf_params[2], sdf_params[5], num=RESOLUTION)
                z, y, x = np.meshgrid(z_, y_, x_, indexing='ij')
                x = np.expand_dims(x, 3)
                y = np.expand_dims(y, 3)
                z = np.expand_dims(z, 3)
                all_pts = np.concatenate((x, y, z), axis=3).astype(np.float32)
                all_pts = all_pts.reshape(1, -1, 3)
                all_pts = np.concatenate((all_pts, extra_pts), axis=1).reshape(SPLIT_SIZE, 1, -1, 3)
                print('all_pts', all_pts.shape)
                batch_points = np.concatenate((batch_points, all_pts), axis=1)

                # norm_params = batch_data['norm_params'][b]
                # center, m, = norm_params[:3], norm_params[3]
                vox256_patch4 = np.zeros((SPLIT_SIZE, NUM_SAMPLE_POINTS, 4, 4, 4), dtype=np.float32)
                vox128_patch4 = np.zeros((SPLIT_SIZE, NUM_SAMPLE_POINTS, 4, 4, 4), dtype=np.float32)
                vox64_patch4 = np.zeros((SPLIT_SIZE, NUM_SAMPLE_POINTS, 4, 4, 4), dtype=np.float32)
                for j in range(SPLIT_SIZE):
                    x_left, x_right = -0.5, 0.5
                    y_left, y_right = -0.5, 0.5
                    z_left, z_right = -0.5, 0.5
                    x_sp16, y_sp16, z_sp16 = (x_right - x_left)/15.0, (y_right - y_left)/15.0, (z_right-z_left)/15.0
                    x_sp32, y_sp32, z_sp32 = (x_right - x_left)/31.0, (y_right - y_left)/31.0, (z_right-z_left)/31.0
                    x_sp64, y_sp64, z_sp64 = (x_right - x_left)/63.0, (y_right - y_left)/63.0, (z_right-z_left)/63.0

                    split_pointsf = batch_points[j, 0]
                    coord_x_16, coord_y_16, coord_z_16 = ((split_pointsf[:, 0]-x_left)/x_sp16).astype('int32'), \
                                                         ((split_pointsf[:, 1]-y_left)/y_sp16).astype('int32'), \
                                                         ((split_pointsf[:, 2]-z_left)/z_sp16).astype('int32')
                    coord_x_32, coord_y_32, coord_z_32 = ((split_pointsf[:, 0]-x_left)/x_sp32).astype('int32'), \
                                                         ((split_pointsf[:, 1]-y_left)/y_sp32).astype('int32'), \
                                                         ((split_pointsf[:, 2]-z_left)/z_sp32).astype('int32')
                    coord_x_64, coord_y_64, coord_z_64 = ((split_pointsf[:, 0]-x_left)/x_sp64).astype('int32'), \
                                                         ((split_pointsf[:, 1]-y_left)/y_sp64).astype('int32'), \
                                                         ((split_pointsf[:, 2]-z_left)/z_sp64).astype('int32')
                    coord_x_16, coord_y_16, coord_z_16 = np.clip(coord_x_16, 0, 15), np.clip(coord_y_16, 0, 15), np.clip(coord_z_16, 0, 15)
                    coord_x_32, coord_y_32, coord_z_32 = np.clip(coord_x_32, 0, 31), np.clip(coord_y_32, 0, 31), np.clip(coord_z_32, 0, 31)
                    coord_x_64, coord_y_64, coord_z_64 = np.clip(coord_x_64, 0, 63), np.clip(coord_y_64, 0, 63), np.clip(coord_z_64, 0, 63)
                    vox256_patch4[j] = batch_data['vox256_reshape64'][b, coord_x_64, coord_y_64, coord_z_64]
                    vox128_patch4[j] = batch_data['vox128_reshape32'][b, coord_x_32, coord_y_32, coord_z_32]
                    vox64_patch4[j] = batch_data['vox64_reshape16'][b,  coord_x_16, coord_y_16, coord_z_16]
                    print(vox256_patch4[j].sum(), vox128_patch4[j].sum(), vox64_patch4[j].sum())
                print(batch_vox128_patch4.shape, vox128_patch4.shape)
                batch_vox64_patch4 = np.concatenate((batch_vox64_patch4, np.expand_dims(vox64_patch4, axis=1)), axis=1)
                batch_vox128_patch4 = np.concatenate((batch_vox128_patch4, np.expand_dims(vox128_patch4, axis=1)), axis=1)
                batch_vox256_patch4 = np.concatenate((batch_vox256_patch4, np.expand_dims(vox256_patch4, axis=1)), axis=1)

        pred_sdf_val_all = np.zeros((SPLIT_SIZE, BATCH_SIZE, NUM_SAMPLE_POINTS, 2 if FLAGS.binary else 1))

        for sp in range(SPLIT_SIZE):
            if FLAGS.ske_local_patch or FLAGS.ske_local_patch_share:
                feed_dict = {ops['is_training_pl']: is_training,
                         ops['input_pls']['sample_pc']: batch_points[sp,...].reshape(BATCH_SIZE, -1, 3),
                         ops['input_pls']['sample_pc_rot']: batch_points[sp,...].reshape(BATCH_SIZE, -1, 3),
                         ops['input_pls']['imgs']: batch_data['img'],
                         ops['input_pls']['trans_mat']: batch_data['trans_mat'],
                         
                         ops['input_pls']['ske_vox64_patch4']:batch_vox64_patch4[sp, ...].reshape(BATCH_SIZE, -1, 4, 4, 4),
                         ops['input_pls']['ske_vox128_patch4']:batch_vox128_patch4[sp, ...].reshape(BATCH_SIZE, -1, 4, 4, 4),
                         ops['input_pls']['ske_vox256_patch4']:batch_vox256_patch4[sp, ...].reshape(BATCH_SIZE, -1, 4, 4, 4)}
            else:
                feed_dict = {ops['is_training_pl']: is_training,
                         ops['input_pls']['sample_pc']: batch_points[sp,...].reshape(BATCH_SIZE, -1, 3),
                         ops['input_pls']['sample_pc_rot']: batch_points[sp,...].reshape(BATCH_SIZE, -1, 3),
                         ops['input_pls']['imgs']: batch_data['img'],
                         ops['input_pls']['trans_mat']: batch_data['trans_mat']}

            output_list = [ops['end_points']['pred_sdf'], ops['end_points']['ref_img'],
                           ops['end_points']['sample_img_points']]
            pred_sdf_val, ref_img_val, sample_img_points_val = sess.run(output_list, feed_dict=feed_dict)
            pred_sdf_val_all[sp,:,:,:] = pred_sdf_val
        pred_sdf_val_all = np.swapaxes(pred_sdf_val_all,0,1) # B, S, NUM SAMPLE, 1 or 2
        pred_sdf_val_all = pred_sdf_val_all.reshape((BATCH_SIZE,-1,2 if FLAGS.binary else 1))[:, :TOTAL_POINTS, :]
        if FLAGS.binary:
            expo = np.exp(pred_sdf_val_all)
            prob = expo[:,:,1] / np.sum(expo, axis = 2)
            result = (FLAGS.threshold - prob) ####/ 10.
            print("result.shape", result.shape)
        else:
            result = pred_sdf_val_all / SDF_WEIGHT
        for b in range(BATCH_SIZE):
            print("submit create_obj")
            executor.submit(create_obj, result[b], batch_data['sdf_params'][b], RESULT_OBJ_PATH, FLAGS.iso)


def to_binary(res, pos, pred_sdf_val_all, sdf_file):
    f_sdf_bin = open(sdf_file, 'wb')

    f_sdf_bin.write(struct.pack('i', -res))  # write an int
    f_sdf_bin.write(struct.pack('i', res))  # write an int
    f_sdf_bin.write(struct.pack('i', res))  # write an int

    positions = struct.pack('d' * len(pos), *pos)
    f_sdf_bin.write(positions)
    val = struct.pack('=%sf'%pred_sdf_val_all.shape[0], *(pred_sdf_val_all))
    f_sdf_bin.write(val)
    f_sdf_bin.close()

def create_obj(pred_sdf_val, sdf_params, dir, i):
    obj_nm =(FLAGS.demo_image_path).split('/')[-1][:-4]
    if FLAGS.ske_local_patch or FLAGS.ske_local_patch_share:
        cube_obj_file = os.path.join(dir, obj_nm+"_disnske.obj")
    else:
        cube_obj_file = os.path.join(dir, obj_nm+"_disn.obj")
    sdf_file = os.path.join(dir, obj_nm+".dist")
    to_binary((RESOLUTION-1), sdf_params, pred_sdf_val, sdf_file)
    create_one_cube_obj("./isosurface/computeMarchingCubes", i, sdf_file, cube_obj_file)
    command_str = "rm -rf " + sdf_file
    print("command:", command_str)
    os.system(command_str)

def create_one_cube_obj(marching_cube_command, i, sdf_file, cube_obj_file):
    command_str = marching_cube_command + " " + sdf_file + " " + cube_obj_file + " -i " + str(i)
    print("command:", command_str)
    os.system(command_str)
    return cube_obj_file

def get_sdf_h5(sdf_h5_file, cat_id, obj):
    h5_f = h5py.File(sdf_h5_file, 'r')
    try:
        if ('pc_sdf_original' in h5_f.keys()
                and 'pc_sdf_sample' in h5_f.keys()
                and 'norm_params' in h5_f.keys()):
            ori_sdf = h5_f['pc_sdf_original'][:].astype(np.float32)
            # sample_sdf = np.reshape(h5_f['pc_sdf_sample'][:],(ori_sdf.shape[0], -1 ,4)).astype(np.float32)
            sample_sdf = h5_f['pc_sdf_sample'][:].astype(np.float32)
            ori_pt = ori_sdf[:,:3]#, ori_sdf[:,3]
            ori_sdf_val = None
            if sample_sdf.shape[1] == 4:
                sample_pt, sample_sdf_val = sample_sdf[:,:3], sample_sdf[:,3]
            else:
                sample_pt, sample_sdf_val = None, sample_sdf[:, 0]
            norm_params = h5_f['norm_params'][:]
            sdf_params = h5_f['sdf_params'][:]
        else:
            raise Exception(cat_id, obj, "no sdf and sample")
    finally:
        h5_f.close()
    return ori_pt, ori_sdf_val, sample_pt, sample_sdf_val, norm_params, sdf_params

if __name__ == "__main__":


    # 1. create all categories / some of the categories:
    create()



    # 2. create single obj, just run python -u create_sdf.py

    # ori_pt, ori_sdf_val, sample_pt, sample_sdf_val, norm_params, sdf_params = \
    #     get_sdf_h5("/ssd1/datasets/ShapeNet/SDF_full/64_expr_1.2/03001627/47cd848a5584867b1e8791c225564ae0/ori_sample.h5",
    #                 "03001627", "47cd848a5584867b1e8791c225564ae0")
    # create_obj(sample_sdf_val, sdf_params, "send/",
    #            "03001627", "97cd4ed02e022ce7174150bd56e389a8", "111", 0.00)

