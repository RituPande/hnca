
from hnca.framework.idefs import ICellularAutomata

import tensorflow as tf
from tensorflow import keras
from  keras.layers import Layer, Conv2D, DepthwiseConv2D
import numpy as np
from  hnca.framework.types import CellDetector, Graph
from spektral.models import GeneralGNN
from spektral.utils import sp_matrix_to_sp_tensor



class LeafImgCA(Layer, ICellularAutomata):

    """
    This is a derived class to create a leaf CA that learns rules to create a target image 
    
    Attributes
    ----------
    
    Methods
    -------

    """


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

        self._init_ca_class_members()

        self.perception = DepthwiseConv2D(\
            kernel_size=3,\
                padding='same',\
                    depth_multiplier= 4,\
                         depthwise_initializer= LeafImgCA.PerceptionKernelInitializer())

        self.perception.trainable = False

        self.features =  Conv2D(filters=LeafImgCA.n_features,\
             kernel_size=1,\
              bias_initializer='glorot_uniform',\
                activation = 'relu') 

        self.new_state = Conv2D(filters=LeafImgCA.n_channels,\
            kernel_size=1,\
              use_bias=False,\
                kernel_initializer=tf.keras.initializers.Zeros())
   

    @staticmethod
    def _init_static_vars():
        if not hasattr(LeafImgCA, "n_channels"):
            LeafImgCA.n_channels = 12 # exact channel count TBD

        if not hasattr(LeafImgCA, "n_schannels"):
            LeafImgCA.n_schannels = 8 # exact channel count TBD

        if not hasattr(LeafImgCA, "n_features"):
            LeafImgCA.n_features = LeafImgCA.n_channels*4 # exact channel count TBD

    def _init_ca_class_members(self):
        self._parent =  None
        self._child  = None
        self._level = 0
        self.signal ={}


    def _circular_pad( self, x  ):
      x = x.numpy()
      x = np.pad(x,((0,0),(1,1),(1,1),(0,0)),mode='wrap')
      return tf.convert_to_tensor(x)

    def call( self, x, update_rate=0.5  ):
        
        b,h,w,c = x.shape
        udpate_mask = tf.floor(tf.random.uniform(shape=(b,h,w,1) )+update_rate)
        y = self.perception(x)
        y = self.features(y)
        y = self.new_state(y)
        y = y*udpate_mask + x
        
        return y

    def step( self, x, n_steps, update_rate=0.5):

        for _ in range(n_steps):
            x = self(x, update_rate )
        return x
    
    def process_signal(self, level, signal, parent_cell_id ):
        pass
        
    @staticmethod
    def make_seed(size, n=1):
        x = np.zeros([n, size, size, LeafImgCA.n_channels ], np.float32)
        return x

    
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
        super().__init__()
    
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
class HCA(Layer,ICellularAutomata):
    def __init__(self, x ):
        super().__init__()
        self._init_ca_class_members()

        self.detector = CellDetector()
        self.graph = None
        self.gnca = None
        self.node_members = None
    
    def _init_ca_class_members(self):
        self._parent =  None
        self._child  = None
        self._level = 0
        self.signal ={}
       
    def build(self):

        # TODO: check if default values for other parameters are ok for our needs.
        input_shape =self.graph.node_features.shape

        self.gnca = GeneralGNN(\
            input_shape[0][-1],\
            activation='relu',\
            message_passing=1,\
            pool=None,\
            batch_norm=False )
            
    def update_ca(self, x, make_recursive=False):

        _, node_details = self.detector(x)
        node_features = []
        for cell in node_details:
            node_features.append(cell['center'])

        self.graph =  Graph(node_features)
        self.node_members = node_details['members']
        
    
    def call( self ):
        x = self.graph.node_features
        a = sp_matrix_to_sp_tensor(self.graph.spA)
        x = self.gnca([x,a])
        self.child.update_signal(x,self.node_members )
        return x

    def step( self, x, n_steps, update_rate=0.5):
        pass
    
    #TODO
    #Implement the abstract methods of the base class

    