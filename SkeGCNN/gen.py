import tensorflow as tf
from deformation.utils import *
from deformation.models import GCN
from deformation.fetcher import *

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('idx', 0, 'index of input image')
flags.DEFINE_string('cats', 'all', 'category')
flags.DEFINE_string('list_root', '../sharedata/filelists', 'filelist root')
flags.DEFINE_string('basemesh_root', './data/allcats_basemesh256', 'basemesh root')
flags.DEFINE_bool('load_model', False, 'If load preTrained model.')
flags.DEFINE_string('checkpoint', 'all_e2000_n1', 'save model and load the preTrained model.')
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
    #'lape_idx': tf.placeholder(tf.float32, shape=(None,20)), #for laplace term
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}
model = GCN(placeholders, logging=True)

# Load data, initialize session
data = DataFetcher(filelist_root=FLAGS.list_root, basemesh_root=FLAGS.basemesh_root, train=False, cat_list=FLAGS.cats, idx=FLAGS.idx)
data.setDaemon(True) ####
data.start()
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
config.allow_soft_placement=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
dirname = FLAGS.checkpoint
model.load(sess, dirname)

# Test graph model
test_loss = open(dirname+'/test_loss_record.txt', 'a')
test_loss.write('Start testing Chamfer distance (between mesh vertexes and surface points)')

test_number = data.number
all_loss = np.zeros(test_number,dtype='float32')
for iters in range(test_number):
    # Fetch training data
    img_inp, trans_mat_right, label, basemesh, cat_mod = data.fetch()
    feed_dict = construct_feed_dict(img_inp, trans_mat_right, basemesh, label, placeholders)

    # Testing step
    cd, = sess.run([model.chamfer], feed_dict=feed_dict)
    all_loss[iters] = cd
    mean_loss = np.mean(all_loss[np.where(all_loss)])
    print('Iteration %d, Mean loss = %f, iter loss = %f, %d'%(iters+1, mean_loss, cd, data.queue.qsize()))
    test_loss.write('loss %f\n' % mean_loss)
    test_loss.flush()

    # Save for visualization
    out1, out2, = sess.run([model.output1,model.output2], feed_dict=feed_dict)
    export_mesh(img_inp, label, basemesh, cat_mod, out1, out2, dirname)
    #export_groundtruth(label, cat_mod, dirname)
    #export_img_mesh(img_inp, label, basemesh, cat_mod, out1, out2, dirname)
data.shutdown()
print('CNN-GCN Test Finished!')
