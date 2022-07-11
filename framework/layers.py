from hnca.framework.idefs import ICellularAutomata
import tensorflow as tf
from tensorflow import keras
from  keras.layers import Conv2D, DepthwiseConv2D, Reshape
from keras.applications.vgg16 import VGG16

import numpy as np

"""
    This is a derived class to create a leaf CA that learns rules to create a target image 
    
    Attributes
    ----------
    
    Methods
    -------

 """

class LeafImgCA(ICellularAutomata):

    class DepthwiseInitializer(tf.keras.initializers.Initializer):

        def __init__(self, mean, stddev):
          ident = tf.constant([[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]])
          sobel_x = tf.constant([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]])
          sobel_y = tf.constant([[-1.0,-2.0,-1.0],[0.0,0.0,0.0],[1.0,2.0,1.0]])
          lap = tf.constant([[1.0,2.0,1.0],[2.0,-12,2.0],[1.0,2.0,1.0]])
          self.kernel = tf.stack([ ident, sobel_x, sobel_y, lap ])

        def __call__(self, shape, dtype=None, **kwargs):
          return self.kernel

    def __init__(self ):
        super().__init__()

        LeafImgCA._init_static_vars()
        self.perception = DepthwiseConv2D(\
        kernel_size=(3,3),\
          padding='same',\
            depth_multiplier=3,\
              depthwise_initializer=LeafImgCA.DepthwiseInitializer()  )

        self.perception.trainable = False

        self.features = Conv2D(\
        filters=128,\
         kernel_size=1,\
          padding='same',\
           activation='relu')

        self.new_state = Conv2D(\
        filters=LeafImgCA.n_channels,\
         kernel_size=1,\
          padding='same')

   
    @staticmethod
    def _init_static_vars():
        if not hasattr(LeafImgCA, "n_channels"):
            LeafImgCA.n_channels = 16 # exact channel count TBD

        if not hasattr(LeafImgCA, "n_schannels"):
            LeafImgCA.n_schannels = 8 # exact channel count TBD

        
    def call( self, x, training=None ):

        x = self.perception(x)
        x = self.features(x)
        x= self.new_state(x)
        
  
    def alive_masking(self, x):
        pass
    
    @staticmethod
    def calc_styles_vgg(img):
      vgg16 =VGG16()
      style_layers = [1, 6, 11, 18, 25]  
      mean = tf.constant([0.485, 0.456, 0.406])[:,None,None]
      std = tf.constant([0.229, 0.224, 0.225])[:,None,None]
      x = (img-mean) / std
      b, h, w,c = x.shape
      features = [x.reshape(b, h*w, c)]
      for i, layer in enumerate(vgg16[:max(style_layers)+1]):
        x = layer(x)
        if i in style_layers:
          b, h, w,c = x.shape
          features.append(x.reshape(b,h*w,c))
      return features

    @staticmethod
    def gram_loss( x, y):
      pass

    @staticmethod
    def create_vgg_loss(target_img):
      target_style = LeafImgCA.calc_styles_vgg(target_img)
      def loss_f(img):
        img_style = LeafImgCA.calc_styles_vgg(img)
        sum(LeafImgCA.gram_loss(x, y) for x, y in zip(target_style, img_style))

      return loss_f

                
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

    