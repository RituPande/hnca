from hnca.framework.idefs import ICellularAutomata
import tensorflow as tf
from tensorflow import keras
from  keras.layers import Conv2D, DepthwiseConv2D, Reshape

import numpy as np

"""
    This is a derived class to create a leaf CA that learns rules to create a target image 
    
    Attributes
    ----------
    
    Methods
    -------

 """

class LeafImgCA(ICellularAutomata):

    def __init__(self ):
        super().__init__()

        LeafImgCA._init_static_vars()
        self.perception = DepthwiseConv2D(kernel_size=(3,3),\
         padding='same',\
         depth_multiplier=3,\
         depthwise_initializer=LeafImgCA.getPerceptionKernel() )

        self.features = Conv2D(filters=128, kernel_size=1, padding='same') 
        self.new_state = Conv2D(filters=16, kernel_size=1, padding='same')

   
    @staticmethod
    def _init_static_vars():
        if not hasattr(LeafImgCA, "n_channels"):
            LeafImgCA.n_channels = 16 # exact channel count TBD

        if not hasattr(LeafImgCA, "n_schannels"):
            LeafImgCA.n_schannels = 8 # exact channel count TBD

        if not hasattr(LeafImgCA, "perception_kernel"):
            LeafImgCA.perception_kernel = LeafImgCA.getPerceptionKernel()

    @staticmethod
    def _getPerceptionKernel():
      ident = tf.constant([[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]])
      sobel_x = tf.constant([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]])
      sobel_y = tf.constant([[-1.0,-2.0,-1.0],[0.0,0.0,0.0],[1.0,2.0,1.0]])
      lap = tf.constant([[1.0,2.0,1.0],[2.0,-12,2.0],[1.0,2.0,1.0]])
      kernel = tf.stack([ ident, sobel_x, sobel_y, lap ])
      return kernel

    def call( self, x, training=None ):

        x = self.perception(x)
        x = self.features(x)
        x= self.new_state(x)
        
  
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

    