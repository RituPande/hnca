
import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.layers import Conv2D,AveragePooling2D,UpSampling2D

class CAComm(Model):
    def __init__(self, n_leaf_ca_schannels, signal_factor  ):
        super(CAComm, self).__init__()
        self.signal_creator =  Conv2D(filters=n_leaf_ca_schannels,\
                                        kernel_size=1,\
                                            bias_initializer='glorot_uniform',\
                                                kernel_initializer=tf.keras.initializers.Zeros()) # use linear activation
                                                                                                  # should we use_bias?
        self.feedback = AveragePooling2D(pool_size=(self.signaling_factor,self.signaling_factor), strides= self.signaling_factor )
        self.upscale_signal = UpSampling2D(size=(self.signaling_factor,self.signaling_factor))

        initializer = tf.random_uniform_initializer(minval=-1.0 , maxval=1.0,  )
        self.signal_lr = tf.Variable( initializer(shape=[1], dtype=tf.float32),  trainable=True)

    def call(self, parent_x, leaf_x, use_rgb_in_signal_src=False, use_rgb_in_signal_dst=False ):

      s = parent_x
      if not use_rgb_in_signal_src:
        split_sizes = [3, -1] # remove latent channels from RGB channels 
        _, s  = tf.split(parent_x, split_sizes, axis=-1 )
            
      s = self.signal_creator(s)
      s = self.upscale_signal(s)

      orig_signal = leaf_x
      if not use_rgb_in_signal_dst:
        _ , orig_signal = tf.split(leaf_x, [self.leaf_ca_model.n_channels, -1], axis=3 )

      # add signal from parent CA to signal channels of leaf CA
      new_leaf_x = orig_signal + self.signal_lr*s

      return new_leaf_x



