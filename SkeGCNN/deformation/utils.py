import os
import sys
import numpy as np
import scipy.sparse as sp
import trimesh
import cv2

def getBlenderProj(az, el, distance_ratio, img_w=137, img_h=137):
    """Calculate 4x3 3D to 2D projection matrix given viewpoint parameters."""
    F_MM = 35.  # Focal length
    SENSOR_SIZE_MM = 32.
    PIXEL_ASPECT_RATIO = 1.  # pixel_aspect_x / pixel_aspect_y
    RESOLUTION_PCT = 100.
    SKEW = 0.
    CAM_MAX_DIST = 1.75
    CAM_ROT = np.asarray([[1.910685676922942e-15, 4.371138828673793e-08, 1.0],
                      [1.0, -4.371138828673793e-08, -0.0],
                      [4.371138828673793e-08, 1.0, -4.371138828673793e-08]])

    # Calculate intrinsic matrix.
# 2 atan(35 / 2*32)
    scale = RESOLUTION_PCT / 100
    # print('scale', scale)
    f_u = F_MM * img_w * scale / SENSOR_SIZE_MM
    f_v = F_MM * img_h * scale * PIXEL_ASPECT_RATIO / SENSOR_SIZE_MM
    # print('f_u', f_u, 'f_v', f_v)
    u_0 = img_w * scale / 2
    v_0 = img_h * scale / 2
    K = np.matrix(((f_u, SKEW, u_0), (0, f_v, v_0), (0, 0, 1)))

    # Calculate rotation and translation matrices.
    # Step 1: World coordinate to object coordinate.
    sa = np.sin(np.radians(-az))
    ca = np.cos(np.radians(-az))
    se = np.sin(np.radians(-el))
    ce = np.cos(np.radians(-el))
    R_world2obj = np.transpose(np.matrix(((ca * ce, -sa, ca * se),
                                          (sa * ce, ca, sa * se),
                                          (-se, 0, ce))))

    # Step 2: Object coordinate to camera coordinate.
    R_obj2cam = np.transpose(np.matrix(CAM_ROT))
    R_world2cam = R_obj2cam * R_world2obj
    cam_location = np.transpose(np.matrix((distance_ratio * CAM_MAX_DIST,
                                           0,
                                           0)))
    # print('distance', distance_ratio * CAM_MAX_DIST)
    T_world2cam = -1 * R_obj2cam * cam_location

    # Step 3: Fix blender camera's y and z axis direction.
    R_camfix = np.matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
    R_world2cam = R_camfix * R_world2cam
    T_world2cam = R_camfix * T_world2cam

    RT = np.hstack((R_world2cam, T_world2cam))

    return K, RT

def get_rotate_matrix(rotation_angle1):
    cosval = np.cos(rotation_angle1)
    sinval = np.sin(rotation_angle1)

    rotation_matrix_x = np.array([[1, 0,        0,      0],
                                  [0, cosval, -sinval, 0],
                                  [0, sinval, cosval, 0],
                                  [0, 0,        0,      1]])
    rotation_matrix_y = np.array([[cosval, 0, sinval, 0],
                                  [0,       1,  0,      0],
                                  [-sinval, 0, cosval, 0],
                                  [0,       0,  0,      1]])
    rotation_matrix_z = np.array([[cosval, -sinval, 0, 0],
                                  [sinval, cosval, 0, 0],
                                  [0,           0,  1, 0],
                                  [0,           0,  0, 1]])
    scale_y_neg = np.array([
        [1, 0,  0, 0],
        [0, -1, 0, 0],
        [0, 0,  1, 0],
        [0, 0,  0, 1]
    ])

    neg = np.array([
        [-1, 0,  0, 0],
        [0, -1, 0, 0],
        [0, 0,  -1, 0],
        [0, 0,  0, 1]
    ])
    return np.linalg.multi_dot([neg, rotation_matrix_z, rotation_matrix_z, scale_y_neg, rotation_matrix_x])
rot_mat = get_rotate_matrix(-np.pi / 2)

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def normalize_adj(features):
    # normalizes symetric, binary adj matrix such that sum of each row is 1 
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)

def construct_feed_dict(img_inp, trans_mat_right, basemesh, labels, placeholders):
    """Construct feed dictionary."""
    coord = basemesh['vertices']
    edges = basemesh['edges']
    faces = basemesh['faces']

    vertex_size = len(coord)
    edge_size = len(edges)
    iden = sp.eye(vertex_size)
    adj = sp.coo_matrix((np.ones(edge_size,dtype='float32'),
                        (edges[:,0],edges[:,1])),shape=(vertex_size,vertex_size))
    support = [sparse_to_tuple(iden), normalize_adj(adj)]

    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['features']: coord})
    feed_dict.update({placeholders['img_inp']: img_inp})
    feed_dict.update({placeholders['trans_mat']: trans_mat_right})
    feed_dict.update({placeholders['edges']: edges})
    feed_dict.update({placeholders['faces']: faces})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: coord[1].shape})
    feed_dict.update({placeholders['dropout']: 0.})
    return feed_dict

def export_mesh(img_inp, label, basemesh, cat_mod, out1, out2, savedir):
    cat, mod, seq = cat_mod.split('_')
    if not os.path.exists(savedir+'/generation/'+cat):
        os.makedirs(savedir+'/generation/'+cat)

    mesh_file = savedir+'/generation/%s/%s.obj'%(cat,mod)
    vertices = out2
    faces = basemesh['faces']
    mesh = trimesh.Trimesh(vertices, faces, vertex_normals=None, process=False)
    mesh.export(mesh_file)
    print('vertices:', out2.shape, 'faces:', faces.shape, mesh_file)


def export_img_mesh(img_inp, label, basemesh, cat_mod, out1, out2, savedir):
    if not os.path.exists(savedir+'/img'):
        os.mkdir(savedir+'/img')
    if not os.path.exists(savedir+'/predict'):
        os.mkdir(savedir+'/predict')
    img_file = savedir+'/img/%s.png'%cat_mod
    cv2.imwrite(img_file, img_inp*255)

    mesh_file = savedir+'/predict/%s.obj'%cat_mod
    vertices = out2
    faces = basemesh['faces']
    mesh = trimesh.Trimesh(vertices, faces, vertex_normals=None, process=False)
    mesh.export(mesh_file)
    print('vertices:', out2.shape, 'faces:', faces.shape, mesh_file)

from plyfile import PlyData, PlyElement
def export_pointcloud(vertices, normals, out_file, as_text=True):
    assert(vertices.shape[1] == 3 and normals.shape[1]==3)
    vertices = np.array(vertices).astype(np.float32)
    normals = np.array(normals).astype(np.float32)
    vertices_normals = np.ascontiguousarray(np.concatenate([vertices, normals], axis=1))
    vector_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')]
    vertices_normals = vertices_normals.view(dtype=vector_dtype).flatten()
    plyel = PlyElement.describe(vertices_normals, 'vertex')
    plydata = PlyData([plyel], text=as_text)
    plydata.write(out_file)

def load_pointcloud(in_file):
    plydata = PlyData.read(in_file)
    vertices = np.stack([
        plydata['vertex']['x'],
        plydata['vertex']['y'],
        plydata['vertex']['z']
    ], axis=1)

    normals = np.stack([
        plydata['vertex']['nx'],
        plydata['vertex']['ny'],
        plydata['vertex']['nz']
    ], axis=1)
    return vertices, normals

def export_groundtruth(label, cat_mod, savedir):
    cat, mod, seq = cat_mod.split('_')
    if not os.path.exists(savedir+'/generation/'+cat):
        os.makedirs(savedir+'/generation/'+cat)

    pts_file = savedir+'/generation/%s/%s_gt.ply'%(cat,mod)
    export_pointcloud(label[:, 0:3], label[:, 3:6], pts_file)
