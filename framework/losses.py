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
       

    def __call__(self, x  ):
        #img = tf.clip_by_value(to_rgb(x)*255.0, 0, 255.0)
        img = to_rgb(x)*255.0
        loss = np.inf
        if self.loss_type in ['gram','ot']:
            img_style = self._calc_styles_vgg(img)
            if self.loss_type=='gram':
                loss = [self._gram_loss(y_true,y_pred) for y_true, y_pred in zip(self.target_style, img_style)]
            elif self.loss_type=='ot':   
                loss = [self._ot_loss(y_true,y_pred) for y_true, y_pred in zip(self.target_style, img_style)]
                                
            else:
                print("Unsupported loss type")
        return tf.reduce_sum(loss)

    def _calc_styles_vgg(self, img):
        #x = preprocess_input(img)
        x = img[..., ::-1] - np.float32([103.939, 116.779, 123.68])
        #[print(self.vgg16.layers[i].name) for i in self.style_layers]
      
        b, h, w, c  = x.shape
        features = [tf.reshape(x, (b, h*w, c) )]        
        #features = []
        for i in range(max(self.style_layers)+1):
            x = self.vgg16.layers[i](x)
            if i in self.style_layers:
                b, h, w,c = x.shape
                features.append(tf.reshape(x, (b, h*w,c) ) )
         
        return features

    def _gram_loss(self, y_true, y_pred):
        
        b, size, c = y_true.shape
        G_true = tf.matmul(tf.transpose(y_true,perm=[0,2,1]),y_true)
        G_pred = tf.matmul(tf.transpose(y_pred, perm = [0,2,1]),y_pred)

        loss = tf.reduce_sum(tf.square(G_true - G_pred))/(4.0 * (c ** 2) * ( size** 2))

        loss =  tf.cast(loss, dtype=tf.float32)
        return loss

    def _ot_loss(self, y_true, y_pred, n_directions =  32 ):
        
        b, size, c = y_true.shape
        p_vecs,_ = tf.linalg.normalize(tf.random.normal( shape=(c, n_directions ) ), axis = 0 )  # create  n_directions unit vectors with c dimensions

        proj_true = tf.einsum('bnc,cp->bpn', y_true, p_vecs)
        proj_true = tf.sort(proj_true) # sort on axis = -1
        proj_pred = tf.einsum('bnc,cp->bpn', y_pred, p_vecs)
        proj_pred = tf.sort(proj_pred)
        loss = tf.reduce_sum(tf.square(proj_true - proj_pred )) # loss for each pixel in each direction and take their mean
        loss = tf.cast(loss, dtype=tf.float32 ) # take mean of loss across all directions 
        return loss


class MSELoss:
    def __init__( self, target_img):
        self.target_img = tf.convert_to_tensor(target_img, dtype=tf.float32) 

    def __call__( self, x ):
        img = to_rgb(x)*255.0
        
        loss =  tf.reduce_mean(tf.square(self.target_img - img))

        return loss