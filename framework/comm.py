
from numpy import nonzero
import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.layers import Conv2D,AveragePooling2D,UpSampling2D, Dense

class CAComm(Model):
    def __init__(self, n_leaf_ca_channels, n_leaf_ca_schannels, signal_factor,\
                   sensor_all_ch_src, sensor_all_ch_dst, actuator_all_ch_src, actuator_all_ch_dst,\
                    n_sig_creation_layers ):

        super(CAComm, self).__init__()

        self.n_leaf_ca_channels = n_leaf_ca_channels
        self.n_leaf_ca_schannels = n_leaf_ca_schannels
        self.signal_factor = signal_factor
        self.sensor_all_ch_src = sensor_all_ch_src
        self.sensor_all_ch_dst = sensor_all_ch_dst
        self.actuator_all_ch_src = actuator_all_ch_src
        self.actuator_all_ch_dst = actuator_all_ch_dst
        self.n_sig_creation_layers = n_sig_creation_layers

        n_filters = (n_leaf_ca_channels + n_leaf_ca_schannels) \
                                if actuator_all_ch_dst else n_leaf_ca_schannels

        self.signal_creator =  [Conv2D(filters=n_filters,\
                                        kernel_size=1,\
                                            bias_initializer='glorot_uniform',\
                                                kernel_initializer=tf.keras.initializers.Zeros()) \
                                                  for _ in tf.range( n_sig_creation_layers ) ]

        self.feedback = AveragePooling2D(pool_size=(signal_factor,signal_factor), strides=signal_factor )
        self.upscale_signal = UpSampling2D(size=(signal_factor,signal_factor))
        initializer = tf.random_uniform_initializer(minval=-1.0 , maxval=1.0  )

        #n_features =  n_leaf_ca_schannels \
        #    if not use_all_ch_in_signal_dst else (n_leaf_ca_channels + n_leaf_ca_schannels)
        #self.signal_lr = tf.Variable( initializer(shape=[1,n_features], dtype=tf.float32),  trainable=True)
        self.d = 16

        self.Q_sensor = Dense(units=self.d, use_bias=False) 
        self.K_sensor = Dense(units=self.d, use_bias=False) 
        self.V_sensor = Dense(units=1, use_bias=False) 

        self.Q_actuator = Dense(units=self.d, use_bias=False) 
        self.K_actuator = Dense(units=self.d, use_bias=False) 
        self.V_actuator = Dense(units=1, use_bias=False) 


    def _create_signal(self, x, comm_type):

      if comm_type == 'actuator':
        s = x
        if not self.actuator_all_ch_src:
          split_sizes = [3, -1] # remove latent channels from RGB channels 
          _, s  = tf.split(x, split_sizes, axis=-1 )

        for i in tf.range(self.n_sig_creation_layers):      
          s = self.signal_creator[i](s)

        s = self.upscale_signal(s)
      
      elif comm_type =='sensor':
        s = x
        if not self.sensor_all_ch_src:
          split_sizes = [3, -1] # remove latent channels from RGB channels
          _, s  = tf.split(x, split_sizes, axis=-1 )

        s = self.feedback(x)
      else :
        s = None
        print("Invalid comm_type")

      return s

    def _mix_signal( self, x, s, comm_type):

      x_orig_signal_ch = x
      concat_ch = False

      if comm_type == 'actuator':
        
        if not self.actuator_all_ch_dst:
          x_orig_ch , x_orig_signal_ch = tf.split(x, [self.n_leaf_ca_channels, -1], axis=-1 )
          concat_ch = True

      elif comm_type == 'sensor':

        if not self.sensor_all_ch_dst:
          x_orig_ch , x_orig_signal_ch = tf.split(x, [ 3 , -1], axis=-1 )
          concat_ch = True

       
      b_x, h_x, w_x, c_x =   x_orig_signal_ch.shape

      x_reshaped = tf.reshape(x_orig_signal_ch, (b_x*w_x*h_x,c_x,1) )

      b_s, w_s, h_s, c_s =   s.shape

      s_reshaped = tf.reshape(s, ( b_s*w_s*h_s,c_s, 1) )

      Q = self.Q_actuator(x_reshaped)
      K = self.K_actuator(s_reshaped)
      V = self.V_actuator(s_reshaped)

      K_T = tf.transpose(K, perm=[0,2,1])

      scaling_f = tf.math.sqrt(self.d)
      ALPHA = tf.nn.softmax( (Q @ K_T)/scaling_f, axis = -1 )

      x_new_reshaped = tf.reduce_sum(ALPHA*V, axis=-1)

      x_new_signal_ch = tf.reshape(x_new_reshaped, (b_x,h_x, w_x, c_x ))

      out = (x_orig_ch,x_new_signal_ch)  if concat_ch else (x_new_signal_ch, None)

      
      return out



    def call(self, parent_x, leaf_x, comm_type='actuator' ):

      if comm_type =='actuator':
        s = self._create_signal(parent_x, comm_type)
        out = self._mix_signal(leaf_x, s, comm_type)
      elif comm_type == 'sensor':
        s = self._create_signal(leaf_x, comm_type)
        out = self._mix_signal(parent_x, s, comm_type) if parent_x is not None else (s,None)
      else:
        print("Invalid comm_type:",comm_type)
        out = None 

      return out



