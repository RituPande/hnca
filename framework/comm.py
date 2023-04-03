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
    elif multiplex_type == 'adder':
      self.multiplexer = SimpleMultiplexer( add_lr=False)
    elif multiplex_type == 'simple':
      self.multiplexer = SimpleMultiplexer( add_lr=True, n_features=n_parent_ca_channels)
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
    self.n_leaf_ca_channels = n_leaf_ca_channels
       
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
    elif multiplex_type == 'adder':
      self.multiplexer = SimpleMultiplexer()
    elif multiplex_type == 'simple':
      n = (n_leaf_ca_channels + n_leaf_ca_schannels) \
                                if self.all_ch_dst else n_leaf_ca_schannels
      self.multiplexer = SimpleMultiplexer( add_lr=True,  n_features=n)
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
        x_dst_feat_ch, x_dst_sig_ch = tf.split(x_dst, [self.n_leaf_ca_channels,-1], axis=-1 )
        
    mixed_s_ch = s  if x_dst_sig_ch is None or self.multiplexer==None else self.multiplexer(s, x_dst_sig_ch) 
   
    out = (mixed_s_ch, None) if self.all_ch_dst else (x_dst_feat_ch, mixed_s_ch )
    return out

