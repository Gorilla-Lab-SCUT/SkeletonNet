import numpy as np
import cv2
import random
import math
import os
import threading
import queue
import sys
import h5py
import copy
import time
import multiprocessing

FETCH_BATCH_SIZE = 32
BATCH_SIZE = 32
HEIGHT = 192
WIDTH = 256
POINTCLOUDSIZE = 16384
OUTPUTPOINTS = 1024
REEBSIZE = 1024


def get_filelist(lst_dir, maxnverts, minsurbinvox, cats, cats_info, type):
    for cat in cats:
        cat_id = cats_info[cat]
    inputlistfile = os.path.join(lst_dir, cat_id + type + ".lst")
    with open(inputlistfile, 'r') as f:
        lines = f.read().splitlines()
        file_lst = [[cat_id, line.strip()] for line in lines]
    return file_lst

class Pt_sdf_img(threading.Thread):
    def __init__(self, FLAGS, listinfo=None, info=None, qsize=64, cats_limit=None, shuffle=True):
        super(Pt_sdf_img, self).__init__()
        self.queue = queue.Queue(10000)
        self.list = list()
        self.stopped = False
        self.bno = 0
        self.epoch = 0
        self.listinfo = listinfo
        self.num_points = FLAGS.num_points
        self.gen_num_pt = FLAGS.num_sample_points
        self.batch_size = FLAGS.batch_size
        self.img_dir = info['rendered_dir']
        self.sdf_dir = info['sdf_dir']
        self.vox64_dir = info['ske_vox_dir']
        self.vox128_dir = info['ske_vox_dir']
        self.vox256_dir = info['ske_vox_dir']
        self.patch_res1 = 4
        self.patch_res2 = 8
        self.patch_res3 = 16
        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 60000
        self.data_num = len(self.listinfo)
        self.FLAGS = FLAGS
        self.shuffle = shuffle
        self.num_batches = int(self.data_num / self.FLAGS.batch_size)
        self.cats_limit, self.epoch_amount = self.set_cat_limit(cats_limit)
        self.data_order = list(range(len(listinfo)))
        self.order = self.data_order

    def set_cat_limit(self, cats_limit):
        epoch_amount = 0
        for cat, amount in cats_limit.items():
            cats_limit[cat] = min(self.FLAGS.cat_limit, amount)
            epoch_amount += cats_limit[cat]
        print("epoch_amount ", epoch_amount)
        print("cats_limit ", cats_limit)
        return cats_limit, epoch_amount

    def get_sdf_h5_filenm(self, cat_id, obj):
        return os.path.join(self.sdf_dir, cat_id, obj, "ori_sample.h5")

    def get_img_dir(self, cat_id, obj):
        img_dir = os.path.join(self.img_dir, cat_id, obj)
        return img_dir, None

    def getitem(self, index):
        cat_id, obj, num = self.listinfo[index]
        sdf_file = self.get_sdf_h5_filenm(cat_id, obj)
        ori_pt, ori_sdf_val, sample_pt, sample_sdf_val, norm_params, sdf_params\
            = self.get_sdf_h5(sdf_file, cat_id, obj)
        img_dir, img_file_lst = self.get_img_dir(cat_id, obj)
        vox64_dir = self.get_vox64_dir(cat_id, obj)
        vox128_dir = self.get_vox128_dir(cat_id, obj)
        vox256_dir = self.get_vox256_dir(cat_id, obj)
        return ori_pt, ori_sdf_val, sample_pt, sample_sdf_val, norm_params,\
               sdf_params, img_dir, img_file_lst, cat_id, obj, num, vox64_dir, vox128_dir, vox256_dir

    def __len__(self):
        return self.epoch_amount

    #get the skeletal volume of 64 and 256 
    def get_vox64_dir(self, cat_id, obj):
        vol64_dir = os.path.join(self.vox64_dir, cat_id, obj)
        return vol64_dir

    def get_vox128_dir(self, cat_id, obj):
        vol128_dir = os.path.join(self.vox128_dir, cat_id, obj)
        return vol128_dir

    def get_vox256_dir(self, cat_id, obj):
        vox256_dir = os.path.join(self.vox256_dir, cat_id, obj)
        return vox256_dir

    def get_vox64(self, vox64_dir, num):
        vox64_file = os.path.join(vox64_dir, '64_max_fill.h5')
        vox64 = h5py.File(vox64_file, 'r')['occupancies'][:]
        vox64 = np.unpackbits(vox64.view(np.uint8)).reshape(64, 64, 64)
        return vox64

    def get_vox128(self, vox128_dir, num):
        vox128_file = os.path.join(vox128_dir, '128_max_fill.h5')
        vox128 = h5py.File(vox128_file, 'r')['occupancies'][:]
        vox128 = np.unpackbits(vox128.view(np.uint8)).reshape(128, 128, 128)
        return vox128

    def get_vox256(self, vox256_dir, num):
        vox256_file = os.path.join(vox256_dir, '256_max_fill.h5')
        vox256 = h5py.File(vox256_file, 'r')['occupancies'][:]
        vox256 = np.unpackbits(vox256.view(np.uint8)).reshape(256, 256, 256)
        return vox256

    def get_sdf_h5(self, sdf_h5_file, cat_id, obj):
        h5_f = h5py.File(sdf_h5_file, 'r')
        try:
            if ('pc_sdf_original' in h5_f.keys()
                    and 'pc_sdf_sample' in h5_f.keys()
                    and 'norm_params' in h5_f.keys()):
                ori_sdf = h5_f['pc_sdf_original'][:].astype(np.float32)
                sample_sdf = h5_f['pc_sdf_sample'][:].astype(np.float32)
                ori_pt = ori_sdf[:,:3]
                ori_sdf_val = None
                if sample_sdf.shape[1] == 4:
                    sample_pt, sample_sdf_val = sample_sdf[:, :3], sample_sdf[:, 3]
                else:
                    sample_pt, sample_sdf_val = None, sample_sdf[:, 0]
                norm_params = h5_f['norm_params'][:]
                sdf_params = h5_f['sdf_params'][:]
            else:
                raise Exception(cat_id, obj, "no sdf and sample")
        finally:
            h5_f.close()
        return ori_pt, ori_sdf_val, sample_pt, sample_sdf_val, norm_params, sdf_params


    def get_img(self, img_dir, num):
        img_h5 = os.path.join(img_dir, "%02d.h5"%num)
        cam_mat, cam_pos, trans_mat, obj_rot_mat, regress_mat = None, None, None, None, None
        with h5py.File(img_h5, 'r') as h5_f:
            #if self.FLAGS.img_feat_onestream or self.FLAGS.img_feat_twostream:
            if True:
                trans_mat = h5_f["trans_mat"][:].astype(np.float32)
                obj_rot_mat = h5_f["obj_rot_mat"][:].astype(np.float32)
                regress_mat = h5_f["regress_mat"][:].astype(np.float32)
            # else:
            #     cam_mat, cam_pos = h5_f["cam_mat"][:].astype(np.float32), h5_f["cam_pos"][:].astype(np.float32)
            if self.FLAGS.alpha:
                img_arr = h5_f["img_arr"][:].astype(np.float32)
                img_arr[:, :, :4] = img_arr[:,:,:4] / 255.
            else:
                img_raw = h5_f["img_arr"][:]
                img_arr = img_raw[:, :, :3]
                if self.FLAGS.augcolorfore or self.FLAGS.augcolorback:
                    r_aug = 60 * np.random.rand() - 30
                    g_aug = 60 * np.random.rand() - 30
                    b_aug = 60 * np.random.rand() - 30
                if self.FLAGS.augcolorfore:
                    img_arr[img_raw[:, :, 3] != 0, 0] + r_aug
                    img_arr[img_raw[:, :, 3] != 0, 1] + g_aug
                    img_arr[img_raw[:, :, 3] != 0, 2] + b_aug
                if self.FLAGS.backcolorwhite:
                    img_arr[img_raw[:, :, 3] == 0] = [255, 255, 255]
                if self.FLAGS.augcolorback:
                    img_arr[img_raw[:, :, 3] == 0, 0] + r_aug
                    img_arr[img_raw[:, :, 3] == 0, 1] + g_aug
                    img_arr[img_raw[:, :, 3] == 0, 2] + b_aug
                if self.FLAGS.grayscale:
                    r, g, b = img_raw[:,:,0:1], img_raw[:,:,1:2], img_raw[:,:,2:3]
                    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
                    img_arr = 255.0 - gray
                img_arr = np.clip(img_arr, 0, 255)
                img_arr = img_arr.astype(np.float32) / 255.

            return img_arr, cam_mat, cam_pos, trans_mat, obj_rot_mat, regress_mat

    def degree2rad(self, params):
        params[0] = np.deg2rad(params[0] + 180.0)
        params[1] = np.deg2rad(params[1])
        params[2] = np.deg2rad(params[2])
        return params

    def unit(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def camera_info(self, param):
        az_mat = self.get_az(param[0])
        el_mat = self.get_el(param[1])
        inl_mat = self.get_inl(param[2])
        cam_mat = np.transpose(np.matmul(np.matmul(inl_mat, el_mat), az_mat))
        cam_pos = self.get_cam_pos(param)
        return cam_mat, cam_pos

    def get_cam_pos(self, param):
        camX = 0
        camY = 0
        camZ = param[3]
        cam_pos = np.array([camX, camY, camZ])
        return -1 * cam_pos

    def get_az(self, az):
        cos = np.cos(az)
        sin = np.sin(az)
        mat = np.asarray([cos, 0.0, sin, 0.0, 1.0, 0.0, -1.0*sin, 0.0, cos], dtype=np.float32)
        mat = np.reshape(mat, [3,3])
        return mat
    #
    def get_el(self, el):
        cos = np.cos(el)
        sin = np.sin(el)
        mat = np.asarray([1.0, 0.0, 0.0, 0.0, cos, -1.0*sin, 0.0, sin, cos], dtype=np.float32)
        mat = np.reshape(mat, [3,3])
        return mat
    #
    def get_inl(self, inl):
        cos = np.cos(inl)
        sin = np.sin(inl)
        # zeros = np.zeros_like(inl)
        # ones = np.ones_like(inl)
        mat = np.asarray([cos, -1.0*sin, 0.0, sin, cos, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        mat = np.reshape(mat, [3,3])
        return mat

    def get_single(self, index, cnt):
        #t0 = time.time()
        for i in range(index+cnt, index+cnt+1):
            single_obj = self.getitem(self.order[i])
            if single_obj == None:
                raise Exception("single mesh is None!")
            ori_pt, ori_sdf_val, sample_pt, sample_sdf_val, norm_params, sdf_params, \
                img_dir, img_file_lst, cat_id, obj, num, vox64_dir, vox128_dir, vox256_dir = single_obj
            img, cam_mat, cam_pos, trans_mat, obj_rot_mat, regress_mat = self.get_img(img_dir, num)
            ######
            x_left, x_right = -0.5, 0.5
            y_left, y_right = -0.5, 0.5
            z_left, z_right = -0.5, 0.5
            x_sp16, y_sp16, z_sp16 = (x_right - x_left)/15.0, (y_right-y_left)/15.0, (z_right-z_left)/15.0
            x_sp32, y_sp32, z_sp32 = (x_right - x_left)/31.0, (y_right-y_left)/31.0, (z_right-z_left)/31.0
            x_sp64, y_sp64, z_sp64 = (x_right - x_left)/63.0, (y_right-y_left)/63.0, (z_right-z_left)/63.0
            ###### compute voxel size

            cf_ref_choice = np.random.randint(ori_pt.shape[0], size=self.num_points)
            self.batch_pc[cnt, :, :] = ori_pt[cf_ref_choice, :]
            if self.gen_num_pt > sample_pt.shape[0]:
                choice = np.random.randint(sample_pt.shape[0], size=self.gen_num_pt)
            else:
                choice = np.asarray(random.sample(range(sample_pt.shape[0]), self.gen_num_pt), dtype=np.int32)
            self.batch_sdf_pt[cnt, ...] = sample_pt[choice, :]
            self.batch_sdf_val[cnt, :, 0] = sample_sdf_val[choice]
            
            ########### load multi-scale skeletal volume
            vox64 = self.get_vox64(vox64_dir, num)
            self.batch_vox64[cnt, ...] = vox64
            vox128 = self.get_vox128(vox128_dir, num)
            vox256 = self.get_vox256(vox256_dir, num)
            #[262144, 4, 4, 4]
            vox256_reshape64 = vox256.reshape(64, 4, 64, 4, 64, 4)
            vox256_reshape64 = np.transpose(vox256_reshape64, (0, 2, 4, 1, 3, 5))
            #[32768, 4, 4, 4]
            vox128_reshape32 = vox128.reshape(32, 4, 32, 4, 32, 4)
            vox128_reshape32 = np.transpose(vox128_reshape32, (0, 2, 4, 1, 3, 5))
            #[4096,  4, 4, 4]
            vox64_reshape16 = vox64.reshape(16, 4, 16, 4, 16, 4)
            vox64_reshape16 = np.transpose(vox64_reshape16, (0, 2, 4, 1, 3, 5))
            center, m, = norm_params[:3], norm_params[3]
            pts = (sample_pt[choice, :]) * m + center[None, :] ### align with shapenet v1
            coord_x16, coord_y16, coord_z_16 = ((pts[:, 0]-x_left)/x_sp16).astype('int32'), ((pts[:, 1]-y_left)/y_sp16).astype('int32'), ((pts[:, 2] - z_left)/z_sp16).astype('int32')
            coord_x32, coord_y32, coord_z_32 = ((pts[:, 0]-x_left)/x_sp32).astype('int32'), ((pts[:, 1]-y_left)/y_sp32).astype('int32'), ((pts[:, 2] - z_left)/z_sp32).astype('int32')
            coord_x64, coord_y64, coord_z_64 = ((pts[:, 0]-x_left)/x_sp64).astype('int32'), ((pts[:, 1]-y_left)/y_sp64).astype('int32'), ((pts[:, 2] - z_left)/z_sp64).astype('int32')
            coord_x16, coord_y16, coord_z_16 = np.clip(coord_x16, 0, 15), np.clip(coord_y16, 0, 15), np.clip(coord_z_16, 0, 15)
            coord_x32, coord_y32, coord_z_32 = np.clip(coord_x32, 0, 31), np.clip(coord_y32, 0, 31), np.clip(coord_z_32, 0, 31)
            coord_x64, coord_y64, coord_z_64 = np.clip(coord_x64, 0, 63), np.clip(coord_y64, 0, 63), np.clip(coord_z_64, 0, 63)

            vox256_patch4 = vox256_reshape64[coord_x64, coord_y64, coord_z_64]
            vox128_patch4 = vox128_reshape32[coord_x32, coord_y32, coord_z_32]
            vox64_patch4 = vox64_reshape16[coord_x16, coord_y16, coord_z_16]
            self.batch_vox256_patch4[cnt, ...] = vox256_patch4.reshape(-1, 4, 4, 4)
            self.batch_vox128_patch4[cnt, ...] = vox128_patch4.reshape(-1, 4, 4, 4)
            self.batch_vox64_patch4[cnt, ...] = vox64_patch4.reshape(-1, 4, 4, 4)
            ####
            if self.FLAGS.rot:
                self.batch_sdf_pt_rot[cnt, ...] = np.dot(sample_pt[choice, :], obj_rot_mat)
            else:
                self.batch_sdf_pt_rot[cnt, ...] = sample_pt[choice, :]
            self.batch_norm_params[cnt, ...] = norm_params
            self.batch_sdf_params[cnt, ...] = sdf_params
            self.batch_img[cnt, ...] = img.astype(np.float32)
            self.batch_regress_mat[cnt, ...] = regress_mat
            self.batch_trans_mat[cnt, ...] = trans_mat
            self.batch_cat_id.append(cat_id)
            self.batch_obj_nm.append(obj)
            self.batch_view_id.append(num)
        #print(index, cnt, time.time()-t0)

    def get_batch(self, index):
        #t0 = time.time()
        if index + self.batch_size > self.epoch_amount:
            index = index + self.batch_size - self.epoch_amount
        self.batch_pc = np.zeros((self.batch_size, self.num_points, 3)).astype(np.float32)
        self.batch_sdf_pt = np.zeros((self.batch_size, self.gen_num_pt, 3)).astype(np.float32)
        self.batch_sdf_pt_rot = np.zeros((self.batch_size, self.gen_num_pt, 3)).astype(np.float32)
        self.batch_sdf_val = np.zeros((self.batch_size, self.gen_num_pt, 1)).astype(np.float32)
        #coarse volume 64, multi scale local patches extracted from skeletal volume 256
        self.batch_vox64 = np.zeros((self.batch_size, 64, 64, 64)).astype(np.float32)
        self.batch_vox64_reshape16 = np.zeros((self.batch_size, 16, 16, 16, 4, 4, 4)).astype(np.float32)
        self.batch_vox128_reshape32 = np.zeros((self.batch_size, 32, 32, 32, 4, 4, 4 )).astype(np.float32)
        self.batch_vox256_reshape64 = np.zeros((self.batch_size, 64, 64, 64, 4, 4, 4)).astype(np.float32)
        self.batch_vox256_patch4 = np.zeros((self.batch_size, self.gen_num_pt, 4, 4, 4)).astype(np.float32)
        self.batch_vox128_patch4 = np.zeros((self.batch_size, self.gen_num_pt, 4, 4, 4)).astype(np.float32)
        self.batch_vox64_patch4 = np.zeros((self.batch_size, self.gen_num_pt, 4, 4, 4)).astype(np.float32)
        #
        self.batch_norm_params = np.zeros((self.batch_size, 4)).astype(np.float32)
        self.batch_sdf_params = np.zeros((self.batch_size, 6)).astype(np.float32)
        if self.FLAGS.alpha:
            self.batch_img = np.zeros((self.batch_size, self.FLAGS.img_h, self.FLAGS.img_w, 4), dtype=np.float32)
        if self.FLAGS.grayscale:
            self.batch_img = np.zeros((self.batch_size, self.FLAGS.img_h, self.FLAGS.img_w, 1), dtype=np.float32)
        else:
            self.batch_img = np.zeros((self.batch_size, self.FLAGS.img_h, self.FLAGS.img_w, 3), dtype=np.float32)
        self.batch_regress_mat = np.zeros((self.batch_size, 4, 3), dtype=np.float32)
        self.batch_trans_mat = np.zeros((self.batch_size, 4, 3), dtype=np.float32)
        self.batch_cat_id = []
        self.batch_obj_nm = []
        self.batch_view_id = []

        for cnt in range(self.FLAGS.batch_size):
            self.get_single(index, cnt)

        batch_data = {}
        batch_data['pc'] = self.batch_pc
        batch_data['sdf_pt'] = self.batch_sdf_pt
        batch_data['sdf_pt_rot'] = self.batch_sdf_pt_rot
        batch_data['sdf_val'] = self.batch_sdf_val
        batch_data['vox64'] = self.batch_vox64
        batch_data['vox64_reshape16'] = self.batch_vox64_reshape16
        batch_data['vox128_reshape32'] = self.batch_vox128_reshape32
        batch_data['vox256_reshape64'] = self.batch_vox256_reshape64
        batch_data['vox64_patch4'] = self.batch_vox64_patch4
        batch_data['vox128_patch4'] = self.batch_vox128_patch4
        batch_data['vox256_patch4'] = self.batch_vox256_patch4
        batch_data['norm_params'] = self.batch_norm_params
        batch_data['sdf_params'] = self.batch_sdf_params
        batch_data['img'] = self.batch_img
        batch_data['trans_mat'] = self.batch_trans_mat
        batch_data['cat_id'] = self.batch_cat_id
        batch_data['obj_nm'] = self.batch_obj_nm
        batch_data['view_id'] = self.batch_view_id
        return batch_data

    def refill_data_order(self):
        temp_order = copy.deepcopy(self.data_order)
        cats_quota = {key: value for key, value in self.cats_limit.items()}
        np.random.shuffle(temp_order)
        pointer = 0
        epoch_order=[]
        while len(epoch_order) < self.epoch_amount:
            cat_id, _, _ = self.listinfo[temp_order[pointer]]
            if cats_quota[cat_id] > 0:
                epoch_order.append(temp_order[pointer])
                cats_quota[cat_id]-=1
            pointer+=1
        return epoch_order

    def work(self, epoch, index):
        if index == 0 and self.shuffle:
            self.order = self.refill_data_order()
            print("data order reordered!")
        return self.get_batch(index)

    def run(self):
        while (self.bno // (self.num_batches* self.batch_size)) < self.FLAGS.max_epoch and not self.stopped:
            self.queue.put(self.work(self.bno // (self.num_batches* self.batch_size),
                                     self.bno % (self.num_batches * self.batch_size)))
            self.bno += self.batch_size

    def fetch(self):
        if self.stopped:
            return None
        # else:
        #     print("queue length", self.queue.qsize())
        return self.queue.get()

    def shutdown(self):
        self.stopped = True
        while not self.queue.empty():
            self.queue.get()