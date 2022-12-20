
from numpy import nonzero
import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.layers import Conv2D,AveragePooling2D,UpSampling2D

class CAComm(Model):
    def __init__(self, n_leaf_ca_channels, n_leaf_ca_schannels, signal_factor,\
                   use_all_ch_in_signal_src, use_all_ch_in_signal_dst,\
                    n_sig_creation_layers ):

        super(CAComm, self).__init__()

        self.n_leaf_ca_channels = n_leaf_ca_channels
        self.n_leaf_ca_schannels = n_leaf_ca_schannels
        self.signal_factor = signal_factor
        self.use_all_ch_in_signal_src = use_all_ch_in_signal_src
        self.use_all_ch_in_signal_dst = use_all_ch_in_signal_dst
        self.n_sig_creation_layers = n_sig_creation_layers

        n_filters = (n_leaf_ca_channels + n_leaf_ca_schannels) \
                                if use_all_ch_in_signal_dst else n_leaf_ca_schannels

        self.signal_creator =  [Conv2D(filters=n_filters,\
                                        kernel_size=1,\
                                            bias_initializer='glorot_uniform',\
                                                kernel_initializer=tf.keras.initializers.Zeros()) \
                                                  for _ in tf.range( n_sig_creation_layers ) ]

        self.feedback = AveragePooling2D(pool_size=(signal_factor,signal_factor), strides=signal_factor )
        self.upscale_signal = UpSampling2D(size=(signal_factor,signal_factor))
        initializer = tf.random_uniform_initializer(minval=-1.0 , maxval=1.0  )

        n_features =  n_leaf_ca_schannels \
            if not use_all_ch_in_signal_dst else (n_leaf_ca_channels + n_leaf_ca_schannels)
        self.signal_lr = tf.Variable( initializer(shape=[1,n_features], dtype=tf.float32),  trainable=True)


    def _create_signal(self, x):

      s = x
      if not self.use_all_ch_in_signal_src:
        split_sizes = [3, -1] # remove latent channels from RGB channels 
        _, s  = tf.split(x, split_sizes, axis=-1 )

      for i in tf.range(self.n_sig_creation_layers):      
        s = self.signal_creator[i](s)

      s = self.upscale_signal(s)

      return s

    def _mix_signal( self, leaf_x, s):

      orig_signal = leaf_x
      if not self.use_all_ch_in_signal_dst:
        orig_ch , orig_signal = tf.split(leaf_x, [self.n_leaf_ca_channels, -1], axis=-1 )

      # add signal from parent CA to signal channels of leaf CA
      new_signal_ch = orig_signal + self.signal_lr*s

      out = (orig_ch,new_signal_ch)  if not self.use_all_ch_in_signal_dst else (new_signal_ch, None)

      return out



    def call(self, parent_x, leaf_x, comm_type='actuator' ):

      if comm_type =='actuator':
        s = self._create_signal(parent_x)
        out = self._mix_signal(leaf_x, s)
      elif comm_type == 'sensor':
        out = self.feedback(leaf_x)
      else:
        print("Invalid comm_type:",comm_type)
        out = None 

      return out



