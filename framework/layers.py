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
        self.perception = DepthwiseConv2D(kernel_size=3,padding='same',depth_multiplier=3)
        self.features = Conv2D(filters=128, kernel_size=1, padding='same') 
        self.new_state = Conv2D(filters=16, kernel_size=1, padding='same')

   
    @staticmethod
    def _init_static_vars():
        if not hasattr(LeafImgCA, "c_channels"):
            LeafImgCA.c_channels = 16 # exact channel count TBD

        if not hasattr(LeafImgCA, "c_schannels"):
            LeafImgCA.c_channels = 8 # exact channel count TBD

   
    def call( self, x, training=None ):

        orig_shape = x.shape 
        x = self.perception(x)
        x = Reshape(orig_shape)(x)
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

    