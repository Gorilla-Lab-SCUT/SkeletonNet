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
flags.DEFINE_string('cats', 'all', 'category')
flags.DEFINE_string('list_root', '../sharedata/filelists', 'filelist root')
flags.DEFINE_string('basemesh_root', './data/allcats_basemesh256', 'basemesh root')
flags.DEFINE_bool('load_model', False, 'If load preTrained model.')
flags.DEFINE_string('checkpoint', 'all_e2000_n1', 'save model and load the preTrained model.')
###
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 40, 'Number of epochs to train.')
flags.DEFINE_integer('hidden', 192, 'Number of units in hidden layer. ')
flags.DEFINE_integer('feat_dim', 963, 'Number of units in feature layer.')
flags.DEFINE_integer('coord_dim', 3, 'Number of units in output layer.')
flags.DEFINE_float('weight_decay', 5e-6, 'Weight for L2 loss.')
flags.DEFINE_float('weight_edge', 2000, 'Weight for edge loss')
flags.DEFINE_float('weight_normal', 1, 'Weight for normal loss.')
flags.DEFINE_integer('num_samples', 0, 'Number of samples for CD loss.')
flags.DEFINE_bool('vertex_chamfer', False, 'If compute the cd (vertexes to surface).')

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
    'num_features_nonzero': tf.placeholder(tf.int32) 
}
model = GCN(placeholders, logging=True)

# Load data, initialize session
data = DataFetcher(filelist_root=FLAGS.list_root, basemesh_root=FLAGS.basemesh_root, train=True, cat_list=FLAGS.cats)
data.setDaemon(True)
data.start()
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
config.allow_soft_placement=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
dirname = FLAGS.checkpoint
if FLAGS.load_model:
    model.load(sess, dirname)

# Train graph model
train_loss = open(dirname + '/train_loss_record.txt', 'a')
train_loss.write('Start training, lr =  %f\n'%(FLAGS.learning_rate))
train_number = data.number
print(train_number)
for epoch in range(FLAGS.epochs):
    all_loss = np.zeros(train_number,dtype='float32')
    if epoch == 20:
        model.set_lr(1e-5)
        train_loss.write('Adjust learning rate, lr =  %f\n'%(1e-5))
    if epoch == 30:
        model.set_lr(1e-6)
        train_loss.write('Adjust learning rate, lr =  %f\n'%(1e-6))
    for iters in range(train_number):
        # Fetch training data
        img_inp, trans_mat_right, label, basemesh, cat_mod= data.fetch()
        feed_dict = construct_feed_dict(img_inp, trans_mat_right, basemesh, label, placeholders)

        # Training step
        _, dists = sess.run([model.opt_op, model.loss], feed_dict=feed_dict)
        all_loss[iters] = dists
        mean_loss = np.mean(all_loss[np.where(all_loss)])
        print(cat_mod)
        print('Epoch %d, Iteration %d/%d'%(epoch + 1,iters + 1, train_number))
        print('Mean loss = %f, iter loss = %f, %d'%(mean_loss,dists,data.queue.qsize()))
        if (iters+1)%10000==0:
            model.save(sess, dirname)
    # Save model
    model.save(sess, dirname)
    train_loss.write('Epoch %d, loss %f\n'%(epoch+1, mean_loss))
    train_loss.flush()

data.shutdown()
print 'CNN-GCN Optimization Finished!'
