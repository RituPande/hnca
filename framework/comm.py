import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.layers import Conv2D,AveragePooling2D,UpSampling2D, Dense
from hnca.framework.multiplexer import SimpleMultiplexer,CrossAttMultiplexer 

class Sensor(Model):

  def __init__( self, n_leaf_ca_channels, n_parent_ca_channels,\
                   signal_factor,\
                       n_sig_creation_layers,\
                        all_ch_src,\
                          all_ch_dst,\
                            multiplex_type='simple',\
                            ):

    super(Sensor,self).__init__()
    self.n_leaf_ca_channels = n_leaf_ca_channels
    self.all_ch_src = all_ch_src
    self.all_ch_dst = all_ch_dst # not used currently. Always set to True
    self.signal_factor = signal_factor
    self.n_sig_creation_layers = n_sig_creation_layers

    self.feedback = AveragePooling2D(pool_size=(self.signal_factor,self.signal_factor), strides=self.signal_factor )
    self.signal_creator =  [Conv2D(filters=n_parent_ca_channels,\
                                        kernel_size=1,\
                                            bias_initializer='glorot_uniform',\
                                                kernel_initializer=tf.keras.initializers.Zeros()) \
                                                  for _ in tf.range( self.n_sig_creation_layers ) ]

    if multiplex_type is None:
      self.multiplexer = None
    elif multiplex_type == 'simple':
      self.multiplexer = SimpleMultiplexer( n_features=n_parent_ca_channels)
    elif multiplex_type == 'cross_attention':
      self.multiplexer = CrossAttMultiplexer(d=16)
    else:
      print('Invalid multiplex_type')
      


  def call( self, x_src, x_dst ):

    x_src = x_src if self.all_ch_src else tf.split(x_src, [self.n_leaf_ca_channels, -1 ], axis=-1)[1]
    s = self.feedback(x_src)
    for i in tf.range(self.n_sig_creation_layers):      
        s = self.signal_creator[i](s)

    # x_dst is None in case of first feedback from leaf ca or if we don't want any contribution
    # of parent CA in leaf CA 
    out = s  if x_dst is None or self.multiplexer==None else self.multiplexer(s, x_dst ) 
    return out
    
    

class Actuator(Model):

  def __init__( self, n_leaf_ca_channels, n_leaf_ca_schannels, 
                        signal_factor,\
                          n_sig_creation_layers,\
                            all_ch_src, all_ch_dst,\
                              multiplex_type='simple' ):

    super(Actuator,self ).__init__()

    self.all_ch_src = all_ch_src
    self.all_ch_dst = all_ch_dst
    self.signal_factor = signal_factor
    self.n_sig_creation_layers = n_sig_creation_layers

    
    n_actuation_filters = (n_leaf_ca_channels + n_leaf_ca_schannels) \
                                if self.all_ch_dst else n_leaf_ca_schannels

    self.signal_creator =  [Conv2D(filters=n_actuation_filters,\
                                        kernel_size=1,\
                                            bias_initializer='glorot_uniform',\
                                                kernel_initializer=tf.keras.initializers.Zeros()) \
                                                  for _ in tf.range( self.n_sig_creation_layers ) ]


    self.upscale_signal = UpSampling2D(size=(self.signal_factor, self.signal_factor))
    
    if multiplex_type is None:
      self.multiplexer = None
    elif multiplex_type == 'simple':
      n = (n_leaf_ca_channels + n_leaf_ca_schannels) \
                                if self.all_ch_dst else n_leaf_ca_schannels
      self.multiplexer = SimpleMultiplexer( n_features=n)
    elif multiplex_type == 'cross_attention':
      self.multiplexer = CrossAttMultiplexer(d=16)
    else:
      print('Invalid multiplex type')

     
  def call(self, x_src, x_dst ):
    
    # create signal from source
    s = x_src if self.all_ch_src else tf.split(x_src, [3,-1], axis=-1 )[1] # remove rgb channels
    for i in tf.range(self.n_sig_creation_layers):      
        s = self.signal_creator[i](s)
    s = self.upscale_signal(s)

    # multiplex to destination
    if self.all_ch_dst: 
        x_dst_sig_ch = x_dst
    else:
        x_dst_feat_ch, x_dst_sig_ch = tf.split(x_src, [self.n_leaf_ca_channels,-1], axis=-1 )
        
    mixed_s_ch = s  if x_dst_sig_ch is None or self.multiplexer==None else self.multiplexer(s, x_dst_sig_ch) 
   
    out = (mixed_s_ch, None) if self.all_ch_dst else (x_dst_feat_ch, mixed_s_ch )
    return out

"""
class CAComm(Model):
    def __init__(self, n_leaf_ca_channels, n_leaf_ca_schannels, n_parent_ca_channels, config_params ):

        super(CAComm, self).__init__()

        self.n_leaf_ca_channels = n_leaf_ca_channels
        self.n_leaf_ca_schannels = n_leaf_ca_schannels
        self.n_parent_ca_channels = n_parent_ca_channels

        self.signal_factor = config_params['signal_factor']
        self.sensor_all_ch_src = config_params['sensor_all_ch_src'] 
        self.sensor_all_ch_dst = config_params['sensor_all_ch_dst'] 
        self.actuator_all_ch_src = config_params['actuator_all_ch_src']
        self.actuator_all_ch_dst = config_params['actuator_all_ch_dst']
        self.n_actuator_sig_creation_layers = config_params['n_actuator_sig_creation_layers']
        self.n_sensor_sig_creation_layers = config_params['n_sensor_sig_creation_layers']

        n_actuation_filters = (n_leaf_ca_channels + n_leaf_ca_schannels) \
                                if self.actuator_all_ch_dst else n_leaf_ca_schannels

        self.actuator_signal_creator =  [Conv2D(filters=n_actuation_filters,\
                                        kernel_size=1,\
                                            bias_initializer='glorot_uniform',\
                                                kernel_initializer=tf.keras.initializers.Zeros()) \
                                                  for _ in tf.range( self.n_actuator_sig_creation_layers ) ]
        
        n_sensor_filters = n_parent_ca_channels
        
        self.sensor_signal_creator =  [Conv2D(filters=n_sensor_filters,\
                                        kernel_size=1,\
                                            bias_initializer='glorot_uniform',\
                                                kernel_initializer=tf.keras.initializers.Zeros()) \
                                                  for _ in tf.range( self.n_sensor_sig_creation_layers ) ]

        self.feedback = AveragePooling2D(pool_size=(self.signal_factor,self.signal_factor), strides=self.signal_factor )
        self.upscale_signal = UpSampling2D(size=(self.signal_factor, self.signal_factor))
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

        for i in tf.range(self.n_actuator_sig_creation_layers):      
          s = self.actuator_signal_creator[i](s)

        s = self.upscale_signal(s)
      
      elif comm_type =='sensor':
        s = x
        if not self.sensor_all_ch_src:
          split_sizes = [3, -1] # remove latent channels from RGB channels
          _, s  = tf.split(x, split_sizes, axis=-1 )

        s = self.feedback(x)
        for i in tf.range(self.n_sensor_sig_creation_layers):      
          s = self.sensor_signal_creator[i](s)

      else :
        s = None
        print("Invalid comm_type")

      return s

    def _mix_signal( self, x, s, comm_type):

      x_orig_signal_ch = x
      concat_ch = False

      if comm_type == 'actuator':
        Q = self.Q_actuator
        K = self.K_actuator
        V = self.V_actuator

        if not self.actuator_all_ch_dst:
          x_orig_ch , x_orig_signal_ch = tf.split(x, [self.n_leaf_ca_channels, -1], axis=-1 )
          concat_ch = True
        
      elif comm_type == 'sensor':
        Q = self.Q_sensor
        K = self.K_sensor
        V = self.V_sensor

        if not self.sensor_all_ch_dst:
          x_orig_ch , x_orig_signal_ch = tf.split(x, [ 3 , -1], axis=-1 )
          concat_ch = True

       
      b_x, h_x, w_x, c_x =   x_orig_signal_ch.shape

      x_reshaped = tf.reshape(x_orig_signal_ch, (b_x*w_x*h_x,c_x,1) )

      b_s, w_s, h_s, c_s =   s.shape

      s_reshaped = tf.reshape(s, ( b_s*w_s*h_s,c_s, 1) )

      q = Q(x_reshaped)
      k = K(s_reshaped)
      v = V(s_reshaped)

      k_T = tf.transpose(k, perm=[0,2,1])

      scaling_f = tf.math.sqrt(tf.cast(self.d, dtype=tf.float32))
      alpha = tf.nn.softmax( (q @ k_T)/scaling_f, axis = -1 )

      x_new_reshaped = tf.reduce_sum(alpha*v, axis=-1)

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
"""


