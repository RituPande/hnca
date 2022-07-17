from sympy import Le
from hnca.framework.idefs import ICellularAutomata
from hnca.framework.utils import to_rgb
import tensorflow as tf
from tensorflow import keras
from  keras.layers import Layer, Conv2D, DepthwiseConv2D
from keras.applications.vgg16 import VGG16, preprocess_input


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
            kernel = tf.repeat(kernel, LeafImgCA.n_channels, 2)
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

        self.features =  Conv2D(filters=LeafImgCA.n_features, kernel_size=1, padding='same') 
        self.new_state = Conv2D(filters=LeafImgCA.n_channels, kernel_size=1, padding='same')
        print("Constructor ended")

   
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
        y = self.features(y)
        y = self.new_state(y) + x 
        return y
        
    @staticmethod
    def make_seed(size, n=1):
        x = np.zeros([n, size, size, LeafImgCA.n_channels ], np.float32)
        return x

    def alive_masking(self, x):
        pass
    
    @staticmethod
    def _calc_styles_vgg(img):
        x = preprocess_input(img)

        print("x.shape=",x.shape)
        #mean = tf.constant([0.485, 0.456, 0.406])[None,None,:]
        #std = tf.constant([0.229, 0.224, 0.225])[None,None,:]
        #x = (x-mean) / std

        mean = tf.math.reduce_mean(x,axis=(0,1,2)) [None,None,:]
        std = tf.math.reduce_std (x,axis=(0,1,2) )[None,None,:]
        x = (x-mean) / std
        
        s = (img.shape[1:]) # remove batch dimension of input image shape
        vgg16 = VGG16(weights="imagenet", include_top=False, input_shape=s )
        vgg16.trainable = False ## Not trainable weights
        vgg16.summary()
        style_layers = [1, 4, 7,  11, 15]  
                
        b, h, w, c  = x.shape
        features = [tf.reshape(x, (b, h*w, c) )]

        for i in range(max(style_layers)+1):
            x = vgg16.layers[i](x)
            if i in style_layers:
                b, h, w,c = x.shape
                features.append(tf.reshape(x, (b, h*w,c) ) )
            
        return features


    @staticmethod
    def _gram_loss(y_true,y_pred):
        y_true = tf.squeeze(y_true)
        y_pred = tf.squeeze(y_pred)

        G_true = tf.matmul(tf.transpose(y_true),y_true)
        G_pred = tf.matmul(tf.transpose(y_pred),y_pred)

        loss = tf.reduce_mean(tf.square(G_true - G_pred))

        print("loss=",loss)
        loss =  tf.cast(loss, dtype=tf.float32)
        print("tf.loss=",loss)
        return loss

    @staticmethod
    def _ot_loss(y_true,y_pred):
        pass
    
    @staticmethod
    def create_vgg_loss_fn(target_img, loss_type='gram'):
        
        target_style = LeafImgCA._calc_styles_vgg(target_img)
        
        def loss_f(img):
            img = to_rgb(img)
            loss = np.inf
            if loss_type in ['gram','ot']:
                img_style = LeafImgCA._calc_styles_vgg(img)
                if loss_type=='gram':
                    loss = [LeafImgCA._gram_loss(y_true,y_pred) for y_true, y_pred in zip(target_style, img_style)]
                elif loss_type=='ot':   
                    loss = [LeafImgCA._gram_ot(y_true,y_pred) for y_true, y_pred in zip(target_style, img_style)]
            
            return tf.reduce_sum(loss)
           
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

    