from __future__ import division
import tflearn
from layers import *
from losses import *

flags = tf.app.flags
FLAGS = flags.FLAGS

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.output1 = None
        self.output2 = None
        self.output3 = None

        self.chamfer = None
        self.loss = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        #with tf.device('/gpu:0'):
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential resnet model
        eltwise = [3,5, 10,12]
        concat = [7]
        self.activations.append(self.inputs)
        for idx,layer in enumerate(self.layers):
            hidden = layer(self.activations[-1])
            if idx in eltwise:
                hidden = tf.add(hidden, self.activations[-2]) * 0.5
            if idx in concat:
                hidden = tf.concat([hidden, self.activations[-2]], 1)
            self.activations.append(hidden)

        self.output1 = self.activations[7]
        self.output2 = self.activations[14] # -1
        #self.output3 = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError
    
    def save(self, sess=None, dir=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        if not dir:
            raise AttributeError("save dirname not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "%s/%s.ckpt" % (dir,self.name))
        print("Model saved in file: %s" % save_path)
    
    def load(self, sess=None, dir=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        if not dir:
            raise AttributeError("load dirname not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "%s/%s.ckpt" % (dir,self.name)
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)

class GCN(Model):
    def __init__(self, placeholders, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        self.loss += mesh_loss(self.output1, self.placeholders, 3000, FLAGS.weight_edge, FLAGS.weight_normal, FLAGS.num_samples)
        self.loss += mesh_loss(self.output2, self.placeholders, 3000, FLAGS.weight_edge, FLAGS.weight_normal, FLAGS.num_samples)
        #self.loss += mesh_loss(self.output3, self.placeholders, 3000, FLAGS.weight_edge, FLAGS.weight_normal, FLAGS.num_samples)
        if FLAGS.vertex_chamfer:
            cd, dist1, idx1, dist2, idx2 = cd_loss(self.output2, self.placeholders)
            self.chamfer = cd
            
        # Weight decay loss
        conv_layers = range(1,7) + range(8,14) #+ range(15,21)
        for layer_id in conv_layers:
            for var in self.layers[layer_id].vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

    def _build(self):
        self.build_cnn18() #update image feature
    # first project block
        self.layers.append(GraphProjection(placeholders=self.placeholders))
        self.layers.append(GraphConvolution(input_dim=FLAGS.feat_dim,
                                            output_dim=FLAGS.hidden,
                                            gcn_block_id=1,
                                            placeholders=self.placeholders, logging=self.logging))
        for _ in range(4):
            self.layers.append(GraphConvolution(input_dim=FLAGS.hidden,
                                                output_dim=FLAGS.hidden,
                                                gcn_block_id=1,
                                                placeholders=self.placeholders, logging=self.logging))
        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden,
                                            output_dim=FLAGS.coord_dim,
                                            act=lambda x: x,
                                            gcn_block_id=1,
                                            placeholders=self.placeholders, logging=self.logging))
    # second project block
        self.layers.append(GraphProjection_Rotate(placeholders=self.placeholders))
        self.layers.append(GraphConvolution(input_dim=FLAGS.feat_dim+FLAGS.hidden,
                                            output_dim=FLAGS.hidden,
                                            gcn_block_id=2,
                                            placeholders=self.placeholders, logging=self.logging))
        for _ in range(4):
            self.layers.append(GraphConvolution(input_dim=FLAGS.hidden,
                                                output_dim=FLAGS.hidden,
                                                gcn_block_id=2,
                                                placeholders=self.placeholders, logging=self.logging))
        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden,
                                            output_dim=FLAGS.coord_dim,
                                            act=lambda x: x,
                                            gcn_block_id=2,
                                            placeholders=self.placeholders, logging=self.logging))
    def set_lr(self, learning_rate):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    def build_cnn18(self):
        x=self.placeholders['img_inp']
        x=tf.expand_dims(x, 0)
#224 224
        x=tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x0=x
        x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#112 112
        x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x1=x
        x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#56 56
        x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x2=x
        x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#28 28
        x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x3=x
        x=tflearn.layers.conv.conv_2d(x,256,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#14 14
        x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x4=x
        x=tflearn.layers.conv.conv_2d(x,512,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#7 7
        x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x5=x
#updata image feature
        self.placeholders.update({'img_feat': [tf.squeeze(x2), tf.squeeze(x3), tf.squeeze(x4), tf.squeeze(x5)]})
        self.loss += tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) * 0.3
