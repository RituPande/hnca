import tensorflow as tf
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from hnca.framework.utils import to_rgb


class StyleLoss:
    def __init__(self, target_img, loss_type):
        self.target_img = target_img
        s = (target_img.shape[1:]) # remove batch dimension of input image shape
        self.vgg16 = VGG16(weights="imagenet", include_top=False, input_shape=s )
        self.vgg16.trainable = False ## Not trainable weights
        self.style_layers = [1, 4, 7,  11, 15]  
        self.target_style = self._calc_styles_vgg(target_img)
        self.loss_type = loss_type
       

    def __call__(self, target, x  ):
        #target parameter is not used. It is present only for consistency of signature with mse loss
        img = to_rgb(x)*255.0
        loss = np.inf
       
        if self.loss_type=='style_ot':   
            img_style = self._calc_styles_vgg(img)
            loss = [self._ot_loss(y_true,y_pred) for y_true, y_pred in zip(self.target_style, img_style)]
        else:
            print("Unsupported loss type")
        return tf.reduce_sum(loss)

    def _calc_styles_vgg(self, img):
        x = img[..., ::-1] - np.float32([103.939, 116.779, 123.68])
        b, h, w, c  = x.shape
        features = [tf.reshape(x, (b, h*w, c) )]        
       
        for i in range(max(self.style_layers)+1):
            x = self.vgg16.layers[i](x)
            if i in self.style_layers:
                b, h, w,c = x.shape
                features.append(tf.reshape(x, (b, h*w,c) ) )
         
        return features

    def _ot_loss(self, y_true, y_pred, n_directions =  32 ):
        
        b, size, c = y_true.shape
        p_vecs,_ = tf.linalg.normalize(tf.random.normal( shape=(c, n_directions ) ), axis = 0 )  # create  n_directions unit vectors with c dimensions

        proj_true = tf.einsum('bnc,cp->bpn', y_true, p_vecs)
        proj_true = tf.sort(proj_true) # sort on axis = -1
        proj_pred = tf.einsum('bnc,cp->bpn', y_pred, p_vecs)
        proj_pred = tf.sort(proj_pred)
        loss = tf.reduce_mean(tf.square(proj_true - proj_pred )) # loss for each pixel in each direction and take their mean
        loss = tf.cast(loss, dtype=tf.float32 ) # take mean of loss across all directions 
        return loss


class MSELoss:
            
    def __call__( self, target, x, is_image=True ):
        pred = to_rgb(x)*255.0 if is_image else x
        loss =  tf.reduce_mean(tf.square(target - pred))
        return loss


class OTLoss:
    def __init__(self, n_directions=32):
        self.n_directions =  n_directions
    
    def __call__( self, y_true, y_pred, is_image=True ):
        y_pred = to_rgb(y_pred)*255.0 if is_image else y_pred

        b, h, w , c = y_true.shape
        y_true = tf.reshape( y_true, (b, h*w, c) )

        b, h, w , c = y_pred.shape
        y_pred = tf.reshape( y_pred, (b, h*w, c) )
        
        p_vecs,_ = tf.linalg.normalize(tf.random.normal( shape=(c, self.n_directions ) ), axis = 0 )  # create  n_directions unit vectors with c dimensions
        proj_true = tf.einsum('bnc,cp->bnp', y_true, p_vecs)
        proj_true = tf.sort(proj_true) # sort on axis = -1
        proj_pred = tf.einsum('bnc,cp->bnp', y_pred, p_vecs)
        proj_pred = tf.sort(proj_pred)
        loss = tf.reduce_mean(tf.square(proj_true - proj_pred )) # loss for each pixel in each direction and take their mean
        loss = tf.cast(loss, dtype=tf.float32 ) # take mean of loss across all directions 
        return loss

    def _resize_1d( self, a, n_pred):
        n_true =  a.shape[0]
        idx = tf.cast( tf.linspace(0, n_true-1, n_pred ) + 0.5, dtype=tf.int32 )
        return tf.gather(a, idx)



    
    