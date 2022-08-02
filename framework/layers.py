from hnca.framework.idefs import ICellularAutomata

import tensorflow as tf
from tensorflow import keras
from  keras.layers import Layer, Conv2D, DepthwiseConv2D, Activation, Concatenate, Lambda

from  numpy import random


import numpy as np

"""
    This is a derived class to create a leaf CA that learns rules to create a target image 
    
    Attributes
    ----------
    
    Methods
    -------

 """


class LeafImgCA(Layer, ICellularAutomata):

    class PerceptionKernelInitializer(tf.keras.initializers.Initializer):

        def __call__(self, shape, dtype=None):
            ident = tf.constant([[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]])
            sobel_x = tf.constant([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]])
            sobel_y = tf.constant([[-1.0,-2.0,-1.0],[0.0,0.0,0.0],[1.0,2.0,1.0]])
            lap = tf.constant([[1.0,2.0,1.0],[2.0,-12,2.0],[1.0,2.0,1.0]])

            kernel = tf.stack([ident, sobel_x, sobel_y, lap], axis=-1)[:,:,None,:]
            kernel = tf.repeat(kernel, LeafImgCA.n_channels, axis=2)
            return kernel

    def __init__(self ):

        super().__init__()

        LeafImgCA._init_static_vars()

        self.perception = DepthwiseConv2D(\
            kernel_size=3,\
                padding='same',\
                    depth_multiplier= 4,\
                         depthwise_initializer= LeafImgCA.PerceptionKernelInitializer())

        self.perception.trainable = False

        self.features =  Conv2D(filters=LeafImgCA.n_features,\
             kernel_size=1,\
                 padding='same',\
                     activation = 'relu') 

        self.new_state = Conv2D(filters=LeafImgCA.n_channels,\
            kernel_size=1,\
                 padding='same',\
                    kernel_initializer=tf.keras.initializers.Zeros())
       
        self.split_rgb_latent = Lambda( lambda x : tf.split(x, [3, x.shape[-1]-3], axis=-1 ) )
        self.rgb_rescale = Activation('sigmoid')
        self.out = Concatenate()
       
           
    @staticmethod
    def _init_static_vars():
        if not hasattr(LeafImgCA, "n_channels"):
            LeafImgCA.n_channels = 16 # exact channel count TBD

        if not hasattr(LeafImgCA, "n_schannels"):
            LeafImgCA.n_schannels = 8 # exact channel count TBD

        if not hasattr(LeafImgCA, "n_features"):
            LeafImgCA.n_features = 128 # exact channel count TBD

  
    def call( self, x, training=None ):
        
        y = self.perception(x)
        y = tf.clip_by_value(y, -2.0, 2.0 )
        y = self.features(y)
        y = tf.clip_by_value(y, 0, 2.0 )
        y = self.new_state(y)
        y = tf.clip_by_value(y, -2.0, 2.0 )
        out = y + x
        y = tf.clip_by_value(y, -2.0, 2.0 )
        #rgb, latent = self.split_rgb_latent(out)
        #rgb = self.rgb_rescale(rgb)
        #out = self.out([rgb, latent])
        return out
        
    @staticmethod
    def make_seed(size, n=1):
        x = np.zeros([n, size, size, LeafImgCA.n_channels ], np.float32)
        return x

    def alive_masking(self, x):
        pass


    
    #TODO
    #Implement the abstract methods of the base class
    

"""
    This is a derived class to create a leaf CA with manually configured weights/rules 
    
    Attributes
    ----------
    
    Methods
    -------

 """

class LeafPreconfCA(ICellularAutomata):
    def __init__(self ):
        super().__init__(self)
    
    def call( self, training=None ):
        pass

   
    #TODO
    #Implement the abstract methods of the base class



"""
    This is a derived class to create higher order CA  
    
    Attributes
    ----------
    
    Methods
    -------

 """    
class HCA(ICellularAutomata):
    def __init__(self):
       super().__init__(self )
       if not hasattr(HCA, "n_channels"):
            HCA.n_channels = 10 # exact channel count TBD

    def call( self ):
        # TODO
        pass

    
    #TODO
    #Implement the abstract methods of the base class

    