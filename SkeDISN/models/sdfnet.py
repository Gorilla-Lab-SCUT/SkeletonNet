import tensorflow as tf
import tf_util

#### DISN --  SDF / Occ prediction based on global and/or local image features
def get_sdf_basic2(src_pc, globalfeats, is_training, batch_size, num_point, bn, bn_decay, wd=None):

    net2 = tf_util.conv2d(tf.expand_dims(src_pc,2), 64, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training, 
                            weight_decay=wd, scope='fold1/conv1')
    net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training, 
        weight_decay=wd, scope='fold1/conv2')
    net2 = tf_util.conv2d(net2, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training, 
        weight_decay=wd, scope='fold1/conv3')

    globalfeats = tf.reshape(globalfeats, [batch_size, 1, 1, -1])
    globalfeats_expand = tf.tile(globalfeats, [1, src_pc.get_shape()[1], 1, 1])
    print( 'net2', net2.shape)
    print( 'globalfeats_expand', globalfeats_expand.shape)
    concat = tf.concat(axis=3, values=[net2, globalfeats_expand])

    net2 = tf_util.conv2d(concat, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training, 
        weight_decay=wd, scope='fold2/conv1')
    net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training, 
        weight_decay=wd, scope='fold2/conv2')
    pred = tf_util.conv2d(net2, 1, [1,1], padding='VALID', stride=[1,1], activation_fn=None, bn=False, weight_decay=wd, scope='fold2/conv5')

    pred = tf.reshape(pred, [batch_size, -1, 1])

    return pred


def get_sdf_basic2_binary(src_pc, globalfeats, is_training, batch_size, num_point, bn, bn_decay, wd=None):

    net2 = tf_util.conv2d(tf.expand_dims(src_pc,2), 64, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
                            weight_decay=wd, scope='fold1/conv1')
    net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold1/conv2')
    net2 = tf_util.conv2d(net2, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold1/conv3')

    globalfeats = tf.reshape(globalfeats, [batch_size, 1, 1, -1])
    globalfeats_expand = tf.tile(globalfeats, [1, src_pc.get_shape()[1], 1, 1])
    print( 'net2', net2.shape)
    print( 'globalfeats_expand', globalfeats_expand.shape)
    concat = tf.concat(axis=3, values=[net2, globalfeats_expand])

    net2 = tf_util.conv2d(concat, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold2/conv1')
    net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold2/conv2')
    pred = tf_util.conv2d(net2, 2, [1,1], padding='VALID', stride=[1,1], activation_fn=None, bn=False, weight_decay=wd, scope='fold2/conv5_bi')

    pred = tf.reshape(pred, [batch_size, -1, 2])

    return pred


def get_sdf_basic2_imgfeat_twostream(src_pc, point_feat, is_training, batch_size, num_point, bn, bn_decay, wd=None):

    net2 = tf_util.conv2d(tf.expand_dims(src_pc,2), 64, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
                            weight_decay=wd, scope='fold1/conv1')
    net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold1/conv2')
    net2 = tf_util.conv2d(net2, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold1/conv3')

    concat = tf.concat(axis=3, values=[net2, point_feat])

    net2 = tf_util.conv2d(concat, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold2/conv1')
    net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold2/conv2')
    pred = tf_util.conv2d(net2, 1, [1,1], padding='VALID', stride=[1,1], activation_fn=None, bn=False, weight_decay=wd, scope='fold2/conv5')

    pred = tf.reshape(pred, [batch_size, -1, 1])

    return pred

def get_sdf_basic2_imgfeat_twostream_binary(src_pc, point_feat, is_training, batch_size, num_point, bn, bn_decay, wd=None):

    net2 = tf_util.conv2d(tf.expand_dims(src_pc,2), 64, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
                            weight_decay=wd, scope='fold1/conv1')
    net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold1/conv2')
    net2 = tf_util.conv2d(net2, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold1/conv3')

    concat = tf.concat(axis=3, values=[net2, point_feat])

    net2 = tf_util.conv2d(concat, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold2/conv1')
    net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold2/conv2')
    pred = tf_util.conv2d(net2, 2, [1,1], padding='VALID', stride=[1,1], activation_fn=None, bn=False, weight_decay=wd, scope='fold2/conv5_bi')

    pred = tf.reshape(pred, [batch_size, -1, 2])

    return pred
####


### local skeleton feature extraction module
def get_patch_feats(vox64_patch4, vox128_patch4, vox256_patch4, is_training, batch_size, bn, bn_decay, wd=None, FLAGS=None):
    vox64_patch4 = tf.expand_dims(vox64_patch4, axis=4)
    vox128_patch4 = tf.expand_dims(vox128_patch4, axis=4)
    vox256_patch4 = tf.expand_dims(vox256_patch4, axis=4)
    print('vox64_patch4', vox64_patch4.get_shape())
    print('vox128_patch4', vox128_patch4.get_shape())
    print('vox256_patch4', vox256_patch4.get_shape())
    net1 = tf_util.conv3d(vox64_patch4, 32, [3, 3, 3], stride=[1, 1, 1], \
        bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, padding='VALID', scope='3dconv1_1') # 2

    net1 = tf_util.conv3d(net1, 64, [2, 2, 2], stride=[2, 2, 2], \
        bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, padding='VALID', scope='3dconv1_2') # 1

    net2 = tf_util.conv3d(vox128_patch4, 32, [3, 3, 3], stride=[1, 1, 1], \
        bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, padding='VALID', scope='3dconv2_1') # 2

    net2 = tf_util.conv3d(net2, 128, [2, 2, 2], stride=[2, 2, 2], \
        bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, padding='VALID', scope='3dconv2_2') # 1

    net3 = tf_util.conv3d(vox256_patch4, 64, [3, 3, 3], stride=[1, 1, 1], \
        bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, padding='VALID', scope='3dconv3_1') # 2

    net3 = tf_util.conv3d(net3, 256, [2, 2, 2], stride=[2, 2, 2], \
        bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, padding='VALID', scope='3dconv3_2') # 1

    net = tf.concat([net1, net2, net3], axis=4)
    net = tf.squeeze(net, [1, 2, 3])
    return net


##### DISN -- SDF / Occ prediction based on local image features
def get_sdf_basic2_imgfeat_localstream(src_pc, point_feat, is_training, batch_size, num_point, bn, bn_decay, wd=None):

    net2 = tf_util.conv2d(tf.expand_dims(src_pc,2), 64, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
                            weight_decay=wd, scope='fold1/conv1')
    net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold1/conv2')
    src_pc_feat = tf_util.conv2d(net2, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold1/conv3')

    concat = tf.concat(axis=3, values=[src_pc_feat, point_feat])

    net2 = tf_util.conv2d(concat, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold2/conv1')
    net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold2/conv2')
    localstream_lastfeat = net2
    pred = tf_util.conv2d(net2, 1, [1,1], padding='VALID', stride=[1,1], activation_fn=None, bn=False, weight_decay=wd, scope='fold2/conv5')

    pred = tf.reshape(pred, [batch_size, -1, 1])

    return src_pc_feat, localstream_lastfeat, pred


def get_sdf_basic2_imgfeat_localstream_binary(src_pc, point_feat, is_training, batch_size, num_point, bn, bn_decay, wd=None):

    net2 = tf_util.conv2d(tf.expand_dims(src_pc,2), 64, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
                            weight_decay=wd, scope='fold1/conv1')
    net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold1/conv2')
    src_pc_feat = tf_util.conv2d(net2, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold1/conv3')

    concat = tf.concat(axis=3, values=[src_pc_feat, point_feat])

    net2 = tf_util.conv2d(concat, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold2/conv1')
    net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold2/conv2')
    localstream_lastfeat = net2
    pred = tf_util.conv2d(net2, 2, [1,1], padding='VALID', stride=[1,1], activation_fn=None, bn=False, weight_decay=wd, scope='fold2/conv5_bi')

    pred = tf.reshape(pred, [batch_size, -1, 2])

    return src_pc_feat, localstream_lastfeat, pred
#####


#######  SkeDISN -- SDF / Occ prediction based on  global/local image feature, local skeleton features with shared point features.
def get_sdf_basic2_share(src_pc, globalfeats, is_training, batch_size, num_point, bn, bn_decay, wd=None):

    net2 = tf_util.conv2d(tf.expand_dims(src_pc,2), 64, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
                            weight_decay=wd, scope='fold1/conv1')
    net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold1/conv2')
    pointfeat = tf_util.conv2d(net2, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold1/conv3')

    globalfeats = tf.reshape(globalfeats, [batch_size, 1, 1, -1])
    globalfeats_expand = tf.tile(globalfeats, [1, src_pc.get_shape()[1], 1, 1])
    print( 'pointfeat', pointfeat.shape)
    print( 'globalfeats_expand', globalfeats_expand.shape)
    concat = tf.concat(axis=3, values=[pointfeat, globalfeats_expand])

    net2 = tf_util.conv2d(concat, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold2/conv1')
    net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold2/conv2')
    pred = tf_util.conv2d(net2, 1, [1,1], padding='VALID', stride=[1,1], activation_fn=None, bn=False, weight_decay=wd, scope='fold2/conv5')

    pred = tf.reshape(pred, [batch_size, -1, 1])
    return pointfeat, pred

def get_sdf_basic2_share_binary(src_pc, globalfeats, is_training, batch_size, num_point, bn, bn_decay, wd=None):

    net2 = tf_util.conv2d(tf.expand_dims(src_pc,2), 64, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
                            weight_decay=wd, scope='fold1/conv1')
    net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold1/conv2')
    pointfeat = tf_util.conv2d(net2, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold1/conv3')

    globalfeats = tf.reshape(globalfeats, [batch_size, 1, 1, -1])
    globalfeats_expand = tf.tile(globalfeats, [1, src_pc.get_shape()[1], 1, 1])
    print( 'pointfeat', pointfeat.shape)
    print( 'globalfeats_expand', globalfeats_expand.shape)
    concat = tf.concat(axis=3, values=[pointfeat, globalfeats_expand])

    net2 = tf_util.conv2d(concat, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold2/conv1')
    net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold2/conv2')
    pred = tf_util.conv2d(net2, 2, [1,1], padding='VALID', stride=[1,1], activation_fn=None, bn=False, weight_decay=wd, scope='fold2/conv5_bi')

    pred = tf.reshape(pred, [batch_size, -1, 2])
    return pointfeat, pred

def get_sdf_basic2_imgfeat_twostream_share(src_pc, point_feat, is_training, batch_size, num_point, bn, bn_decay, wd=None):
    concat = tf.concat(axis=3, values=[src_pc, point_feat])

    net2 = tf_util.conv2d(concat, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='conv1')
    net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='conv2')
    pred = tf_util.conv2d(net2, 1, [1,1], padding='VALID', stride=[1,1], activation_fn=None, bn=False, weight_decay=wd, scope='conv5')

    pred = tf.reshape(pred, [batch_size, -1, 1])

    return pred

def get_sdf_basic2_imgfeat_twostream_share_binary(src_pc, point_feat, is_training, batch_size, num_point, bn, bn_decay, wd=None):
    concat = tf.concat(axis=3, values=[src_pc, point_feat])

    net2 = tf_util.conv2d(concat, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='conv1')
    net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='conv2')
    pred = tf_util.conv2d(net2, 2, [1,1], padding='VALID', stride=[1,1], activation_fn=None, bn=False, weight_decay=wd, scope='conv5_bi')

    pred = tf.reshape(pred, [batch_size, -1, 2])

    return pred
#######