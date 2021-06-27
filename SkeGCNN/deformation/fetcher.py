import numpy as np
import cPickle as pickle
import scipy.sparse as sp
import networkx as nx
import threading
import Queue
import os
import sys
import cv2
import math
import time
import h5py
from utils import getBlenderProj, rot_mat
from config import ROOT_IMG, ROOT_LABEL
np.random.seed(123)

class DataFetcher(threading.Thread):
    def __init__(self, filelist_root='../sharedata/filelists', basemesh_root='./data/allcats_basemesh256', \
        cat_list=None, train=True, num_views=6, idx=0):
        super(DataFetcher, self).__init__()
        self.stopped = False
        self.queue = Queue.Queue(64)
        self.filelist_root = filelist_root
        self.img_root = ROOT_IMG
        self.label_root = ROOT_LABEL
        self.basemesh_root = basemesh_root
        if cat_list == 'all':
            self.cat_list = ['04379243', '03001627' , '02958343', '02691156', '04256520', '04090263', '03636649', '04530566', \
                '02828884', '03691459', '02933112', '03211117', '04401088']
        else:
            self.cat_list = cat_list.split('_')

        self.pkl_list = []
        for cat in self.cat_list:
            if train:
                file_list = os.path.join(self.filelist_root, cat+'_train.lst')
                fns = open(file_list, 'r').readlines()
                fns = [f.strip() for f in fns]
                for mod in fns:
                    basemesh_path = os.path.join(self.basemesh_root, cat, mod+'.npz')
                    if os.path.exists(basemesh_path):
                        for seq in range(num_views):
                            self.pkl_list.append((cat, mod, seq))
            else:
                file_list = os.path.join(self.filelist_root, cat+'_test.lst')
                fns = open(file_list, 'r').readlines()
                fns = [f.strip() for f in fns]
                for mod in fns:
                    basemesh_path = os.path.join(self.basemesh_root, cat, mod+'.npz')
                    if os.path.exists(basemesh_path):
                        self.pkl_list.append((cat, mod, idx))
        self.index = 0
        self.number = len(self.pkl_list)
        np.random.shuffle(self.pkl_list)

    def work(self, idx):
        cat, mod, seq = self.pkl_list[idx]
        
        # load image file
        img_path = os.path.join(self.img_root, cat, mod, 'rendering', '%02d.png'%seq)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (224,224))
        img_inp = img.astype('float32')/255.0

        # load camera information
        metadata_path = os.path.join(self.img_root, cat, mod, 'rendering', "rendering_metadata.txt")
        params = np.loadtxt(metadata_path)
        param = params[seq, ...].astype(np.float32)
        az, el, distance_ratio = param[0], param[1], param[3]
        K, RT = getBlenderProj(az, el, distance_ratio, img_w=224, img_h=224)
        trans_mat = np.linalg.multi_dot([RT, rot_mat])
        trans_mat_right = np.transpose(trans_mat)

        # load information file
        basemesh_path = os.path.join(self.basemesh_root, cat, mod+'.npz')
        info = {}
        infomesh = np.load(basemesh_path)
        info['vertices'] = infomesh['vertices']
        info['faces'] = infomesh['faces']
        info['edges'] = infomesh['edges']

        #load label file
        label_h5 = os.path.join(self.label_root, cat, mod, 'sample.h5')
        h5_f = h5py.File(label_h5, 'r')
        points = h5_f['surface_v1_points'][:].astype('float32')
        normals = h5_f['surface_v1_normals'][:].astype('float32')
        choice = np.random.choice(len(points), 10000)
        points = points[choice, :]
        normals = normals[choice, :]
        label = np.concatenate([points, normals], axis=1)
        return img_inp[:,:,:3], trans_mat_right, label, info, cat+'_'+mod+'_%02d'%seq
        
    def run(self):
        while self.index < 9000000 and not self.stopped:
            self.queue.put(self.work(self.index % self.number))
            self.index += 1
            if self.index % self.number == 0:
                np.random.shuffle(self.pkl_list)
    
    def fetch(self):
        if self.stopped:
            return None
        return self.queue.get()
    
    def shutdown(self):
        self.stopped = True
        while not self.queue.empty():
            self.queue.get()

if __name__ == '__main__':
    #file_list = sys.argv[1]
    data = DataFetcher()
    data.start()

    i = 0
    for i in xrange(data.number):
        image,cat_mat, point_normal,info,data_id = data.fetch()
        print(image.shape)
        print(point_normal.shape)
        print(cat_mat.shape)
        print(data_id)
        print(i,data.number)
    data.stopped = True
