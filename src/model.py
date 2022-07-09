from hnca.framework.layers import LeafImgCA, HCA
from hnca.framework.utils import load_image
from keras import Model

"""
This class uses the HNCA framework to create a hierachical CA model
   
    
Attributes
----------
    
Methods
-------

"""

class HNCAModel(Model):

    def __init__( self, image_path ):
        super(HNCAModel,self).__init__()
        img = load_image( image_path )
        self.ca_leaf = LeafImgCA( img )
        
    def call(self ):
        pass
        
    def train( self ):
        pass

    def predict( self ):
        pass
    
    

    