import os
import random
import numpy as np

cats_dict = {
        "watercraft": "04530566",
        "firearm": "04090263",
        "monitor": "03211117",
        "lamp": "03636649",
        "speaker": "03691459",
        "chair": "03001627",
        "bench": "02828884",
        "cabinet": "02933112",
        "car": "02958343",
        "plane": "02691156",
        "couch": "04256520",
        "table": "04379243",
        "cellphone": "04401088"
    }

import math
def normalize(vector, eps=1e-5):
    if len(vector.shape) == 1:
        radius = np.linalg.norm(vector)
        return vector / (radius + eps)
    elif len(vector.shape) == 2:
        radius = np.linalg.norm(vector, axis=1).max()
        return vector / (radius + eps)

def camera_rotation(azimuth, elevation, distance):
    azi = math.radians(azimuth)
    ele = math.radians(elevation)
    dis = distance
    eye = (dis * math.cos(ele) * math.cos(azi),
        dis * math.sin(ele),
        dis * math.cos(ele) * math.sin(azi))
    eye = np.asarray(eye)
    at = np.array([0, 0, 0], dtype='float32')
    up = np.array([0, 1, 0], dtype='float32')
    z_axis = normalize(eye - at, eps=1e-5)  # forward
    x_axis = normalize(np.cross(up, z_axis), eps=1e-5)  # left
    y_axis = normalize(np.cross(z_axis, x_axis), eps=1e-5)  # up
    # rotation matrix: [3, 3]
    R = np.concatenate((x_axis[None, :], y_axis[None, :], z_axis[None, :]), axis=0)
    return R

#initialize the weighs of the network for Convolutional layers and batchnorm layers
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def adjust_learning_rate(optimizer, epoch, phase):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if (epoch%phase==(phase-1)):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']/10.

##forzen the bn parameters
def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm1d') != -1:
        m.eval()

class AverageValueMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def random_sample_pointset(point_set, num):
    if len(point_set) <= num:
        times = num/len(point_set)
        point_set = np.repeat(point_set,times,0)
        left = num-len(point_set)
        point_left = point_set[np.random.choice(point_set.shape[0],left)]
        point_set = np.concatenate((point_set,point_left),axis=0)
    else:
        point_set = point_set[np.random.choice(point_set.shape[0], num)]
    return point_set

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


CHUNK_SIZE = 150
lenght_line = 60
def my_get_n_random_lines(path):
    #MY_CHUNK_SIZE = lenght_line * (n+2)
    #lenght = os.stat(path).st_size
    lines=[]
    with open(path, 'r') as file:
        for line in file.readlines()[17:]:
            lines.append(line)
            #file.seek(random.randint(400, lenght - MY_CHUNK_SIZE))
            #chunk = file.read(MY_CHUNK_SIZE)
            #lines = chunk.split(os.linesep)
        return lines

def get_random_color(pastel_factor = 0.5):
    return [(x+pastel_factor)/(1.0+pastel_factor) for x in [random.uniform(0,1.0) for i in [1,2,3]]]

def color_distance(c1,c2):
    return sum([abs(x[0]-x[1]) for x in zip(c1,c2)])

def generate_new_color(existing_colors,pastel_factor = 0.5):
    max_distance = None
    best_color = None
    for i in range(0,100):
        color = get_random_color(pastel_factor = pastel_factor)
        if not existing_colors:
            return color
        best_distance = min([color_distance(color,c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color

#Example:
def get_colors(num_colors=10):
  colors = []
  for i in range(0,num_colors):
      colors.append(generate_new_color(colors,pastel_factor = 0.9))
  for i in range(0,num_colors):
      for j in range(0,3):
        colors[i][j] = int(colors[i][j]*256)
      colors[i].append(255)
  return colors


#CODE from 3D R2N2
def image_transform(img, crop_x, crop_y, crop_loc=None, color_tint=None):
    """
    Takes numpy.array img
    """

    # Slight translation
    if not crop_loc:
        crop_loc = [np.random.randint(0, crop_y), np.random.randint(0, crop_x)]

    if crop_loc:
        cr, cc = crop_loc
        height, width, _ = img.shape
        img_h = height - crop_y
        img_w = width - crop_x
        img = img[cr:cr + img_h, cc:cc + img_w]
        # depth = depth[cr:cr+img_h, cc:cc+img_w]

    if np.random.rand() > 0.5:
        img = img[:, ::-1, ...]

    return img


def crop_center(im, new_height, new_width):
    height = im.shape[0]  # Get dimensions
    width = im.shape[1]
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2
    return im[top:bottom, left:right]


def add_random_color_background(im, color_range):
    r, g, b = [np.random.randint(color_range[i][0], color_range[i][1] + 1) for i in range(3)]
    if isinstance(im, Image.Image):
        im = np.array(im)

    if im.shape[2] > 3:
        # If the image has the alpha channel, add the background
        alpha = (np.expand_dims(im[:, :, 3], axis=2) == 0).astype(np.float)
        im = im[:, :, :3]
        bg_color = np.array([[[r, g, b]]])
        im = alpha * bg_color + (1 - alpha) * im

    return im


def preprocess_img(im, train=True):
    # add random background
    # im = add_random_color_background(im, cfg.TRAIN.NO_BG_COLOR_RANGE if train else
                                     # cfg.TEST.NO_BG_COLOR_RANGE)

    # If the image has alpha channel, remove it.
    im_rgb = np.array(im)[:, :, :3].astype(np.float32)
    if train:
        t_im = image_transform(im_rgb, 17, 17)
    else:
        t_im = crop_center(im_rgb, 224, 224)

    # Scale image
    t_im = t_im / 255.

    return t_im


if __name__ == '__main__':

  #To make your color choice reproducible, uncomment the following line:
  #random.seed(10)

  colors = get_colors(10)
  print("Your colors:",colors)
