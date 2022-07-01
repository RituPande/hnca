from hnca.framework.idefs import ICellularAutomata
from tensorflow import keras
from  keras.layers import Conv2D, Dense, Flatten, Input

import numpy as np

"""
    This is a derived class to create a leaf CA that learns rules to create an input image 
    
    Attributes
    ----------
    
    Methods
    -------

 """

class LeafImgCA(ICellularAutomata):

    def __init__(self, img ):
        super().__init__(self)

        LeafImgCA._init_static_vars()

        # TODO: 
        # 1. Create conv2d, flatten and dense layers to create perception vector
        # 2. Update self._cell_states
        
    @staticmethod
    def _init_static_vars():
        if not hasattr(LeafImgCA, "c_channels"):
            LeafImgCA.c_channels = 10 # exact channel count TBD
        if not hasattr(LeafImgCA, "c_filters"):
                LeafImgCA.Perception.c_filters = 3
        if not hasattr(LeafImgCA, "c_filter_size"):
                LeafImgCA.c_filter_size = 3

    
    def cell_states( self, states ):
        pass

    def parent(self,p ):
        self._parent = p
        #TODO
        # Add Empty signal channels equal to the level of parent to _cell_states
    
    
    def call( self, training=None ):
        # Note: Input image is provided in the constructor and not as part of call method
        # TODO
        pass 

        
    def alive_masking(self):
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

    def cell_states( self, states ):
        pass

    def parent(self,p ):
        self._parent = p
        #TODO
        # Add Empty signal channels equal to the level of parent to _cell_states
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

    def cell_states( self, states ):
        pass

    def parent(self,p ):
        self._parent = p
        #TODO
        # Add Empty signal channels equal to the level of parent to _cell_states

    #TODO
    #Implement the abstract methods of the base class

    