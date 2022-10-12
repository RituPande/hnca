
import tensorflow as tf
from tensorflow import keras
from  keras.layers import  Conv2D, DepthwiseConv2D,Input,Concatenate
from keras.models import Model
import numpy as np



class ImgCA(Model):

    """
    This is a derived class to create a leaf CA that learns rules to create a target image 
    
    Attributes
    ----------
    
    Methods
    -------

    """

    class PerceptionKernelInitializer(tf.keras.initializers.Initializer):

        def __init__( self, outer):
            super().__init__()
            self.outer = outer

        def __call__(self, shape, dtype=None):
            ident = tf.constant([[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]])
            sobel_x = tf.constant([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]])
            sobel_y = tf.constant([[-1.0,-2.0,-1.0],[0.0,0.0,0.0],[1.0,2.0,1.0]])
            lap = tf.constant([[1.0,2.0,1.0],[2.0,-12,2.0],[1.0,2.0,1.0]])

            kernel = tf.stack([ident, sobel_x, sobel_y, lap], axis=-1)[:,:,None,:]
            kernel = tf.repeat(kernel, self.outer.n_channels + self.outer.n_schannels , axis=2)
            return kernel

    def __init__(self, n_channels, n_schannels ):

        super().__init__()

        self._init_ca_class_members(n_channels, n_schannels )
        self._init_ca_layers()
        
   
    def _init_ca_class_members(self, n_channels, n_schannels):
        
        self.n_channels = n_channels
        self.n_schannels = n_schannels
        self.n_features = (n_channels + n_schannels)*4

    def _init_ca_layers( self  ):
        if self.n_schannels > 0:
            self.combined_inp = Concatenate(axis=-1)

        self.perception = DepthwiseConv2D(\
            kernel_size=3,\
                padding='valid',\
                    depth_multiplier= 4,\
                            depthwise_initializer= ImgCA.PerceptionKernelInitializer(self))

        self.perception.trainable = False

        self.features =  Conv2D(filters=self.n_features,\
                kernel_size=1,\
                bias_initializer='glorot_uniform',\
                activation = 'relu') 

        self.new_state = Conv2D(filters=self.n_channels + self.n_schannels,\
            kernel_size=1,\
                use_bias=False,\
                kernel_initializer=tf.keras.initializers.Zeros())



    def _circular_pad( self, x, pad  ):
        x = tf.concat([x[:, -pad:], x, x[:, :pad]], axis=1)
        x = tf.concat([x[:, :, -pad:], x, x[:, :, :pad]], axis=2)
        return x


    def _mask_signal_channels( self, x, split_sizes ):
        b, h, w, c = x.shape
        signal_mask = tf.zeros((b,h,w,ImgCA.n_schannels)) 
        features, _  = tf.split(x, split_sizes, axis=-1 )
        x=tf.concat([features, signal_mask], axis=-1)
        return x

    def call( self, x, s=None, update_rate=0.5, training_type='leaf'  ):
        
        b,h,w,c = x.shape
        udpate_mask = tf.floor(tf.random.uniform(shape=(b,h,w,1) )+update_rate)
        
        if s is not None:
            x = self.combined_inp([x,s])

        x_pad = self._circular_pad(x,1)
        y = self.perception(x_pad)
        y = self.features(y)
        y = self.new_state(y)
        
        y = y*udpate_mask + x
                 
        return y

    def step( self, x_initial=None, s=None, n_steps = 50, update_rate=0.5, training_type='leaf' ):

        x = x_initial
        if x is None:
            x = self.leaf_ca_model.make_seed(self.leaf_img_target_size, n=1)
        # In the first step the feature x and signal s are sent as independent inputs
        x  = self(x, s, update_rate,training_type )
        for _ in tf.range(n_steps-1):
             #Remaining steps sets the signal s as None and the output state that has both the feature and signal state
             # that is taken as input
            x  = self(x, None, update_rate,training_type )
           
    
        return x
    
    # TODO: review this.
    def make_seed(self, size, n=1):
        x = np.ones([n, size, size, self.n_channels ], np.float32)
        if self.n_schannels > 0:
            s = np.zeros([n, size, size, self.n_schannels ], np.float32)  
            ca_seed = np.concatenate([x,s], axis=-1)
        else :
            ca_seed = x

        return ca_seed

    
  