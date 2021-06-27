import tensorflow as tf
from deformation.utils import *
from deformation.models import GCN
from deformation.fetcher import *
from deformation.config import ROOT_LABEL, ROOT_IMG
import openmesh

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('basemesh_root', './demo/demo_allcats_basemesh', 'basemesh root')
flags.DEFINE_bool('load_model', False, 'If load preTrained model.')
flags.DEFINE_string('checkpoint', 'all_mesh_refine', 'save model and load the preTrained model.')
###
flags.DEFINE_float('learning_rate', 0, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1, 'Number of epochs to train.')
flags.DEFINE_integer('hidden', 192, 'Number of units in hidden layer. ')
flags.DEFINE_integer('feat_dim', 963, 'Number of units in feature layer.')
flags.DEFINE_integer('coord_dim', 3, 'Number of units in output layer.')
flags.DEFINE_float('weight_decay', 5e-6, 'Weight for L2 loss.')
flags.DEFINE_float('weight_edge', 0, 'Weight for edge loss')
flags.DEFINE_float('weight_normal', 0, 'Weight for normal loss.')
flags.DEFINE_integer('num_samples', 0, 'Number of samples for CD loss.')
flags.DEFINE_bool('vertex_chamfer', True, 'If compute the cd (vertexes to surface).')
###
flags.DEFINE_string('catname', '03001627', 'the catname of demo')
flags.DEFINE_string('modname', '2c03bcb2a133ce28bb6caad47eee6580', 'the modname of demo')
flags.DEFINE_integer('index', '0', 'the selecte index of input image')

# Define placeholders(dict) and model
num_blocks = 2
num_supports = 2
placeholders = {
    'img_inp': tf.placeholder(tf.float32, shape=(224, 224, 3)), #224*224*3
    'trans_mat': tf.placeholder(tf.float32, shape=(4, 3)), #4*3 camera_right
    'labels': tf.placeholder(tf.float32, shape=(None, 6)),
    'features': tf.placeholder(tf.float32, shape=(None, 3)),
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'edges': tf.placeholder(tf.int32, shape=(None, 2)),
    'faces': tf.placeholder(tf.int32, shape=(None, 3)),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32) 
}
model = GCN(placeholders, logging=True)

config=tf.ConfigProto()
config.gpu_options.allow_growth=True
config.allow_soft_placement=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
dirname = FLAGS.checkpoint
model.load(sess, dirname)

def fetch_data(cat, mod, seq):
    img_root = ROOT_IMG
    label_root = ROOT_LABEL
    basemesh_root = FLAGS.basemesh_root
    # load image file
    img_path = os.path.join(img_root, cat, mod, 'rendering', '%02d.png'%seq)
    print(img_path, os.path.exists(img_path))
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (224,224))
    img_inp = img.astype('float32')/255.0
    # load camera information
    metadata_path = os.path.join(img_root, cat, mod, 'rendering', "rendering_metadata.txt")
    params = np.loadtxt(metadata_path)
    param = params[seq, ...].astype(np.float32)
    az, el, distance_ratio = param[0], param[1], param[3]
    K, RT = getBlenderProj(az, el, distance_ratio, img_w=224, img_h=224)
    trans_mat = np.linalg.multi_dot([RT, rot_mat])
    trans_mat_right = np.transpose(trans_mat)
    # load information file
    basemesh_path = os.path.join(basemesh_root, cat, mod+'_%02d.obj'%seq)
    info = {}
    trimesh = openmesh.read_trimesh(basemesh_path)
    vertices = trimesh.points()
    info['vertices'] = vertices.astype('float32')
    halfedges = trimesh.halfedge_vertex_indices()
    info['edges'] = halfedges.astype('int32')
    faces = trimesh.face_vertex_indices()
    info['faces'] = faces.astype('int32')

    #load label file
    label = np.zeros((10000, 6),dtype='float32')
    return img_inp[:,:,:3], trans_mat_right, label, info, cat+'_'+mod+'_%02d'%seq

img_inp, trans_mat_right, label, basemesh, cat_mod = fetch_data(FLAGS.catname, FLAGS.modname, FLAGS.index)
feed_dict = construct_feed_dict(img_inp, trans_mat_right, basemesh, label, placeholders)
out1, out2, = sess.run([model.output1,model.output2], feed_dict=feed_dict)
export_img_mesh(img_inp, label, basemesh, cat_mod, out1, out2, dirname)
