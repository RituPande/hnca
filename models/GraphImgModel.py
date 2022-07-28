from hnca.framework.layers import LeafImgCA, HCA
from hnca.framework.losses import StyleLoss
from hnca.framework.utils import load_image, show_image, plot_loss
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import Model
from tqdm import tqdm



"""
This class uses the HNCA framework to create a hierachical CA model
   
    
Attributes
----------
    
Methods
-------

"""

class GraphImgModel(Model):

    def __init__( self, num_steps, leaf_ca_target ):
        super(GraphImgModel,self).__init__()
        
        self.ca_leaf = LeafImgCA( )
        self.num_steps = num_steps
        self.target_size = 224
        self.target_img = load_image(leaf_ca_target)[None,:,:,:3]
        self.loss = StyleLoss( self.target_img )
       
        
    def call(self, x, training=None ):
        
        x = self.ca_leaf(x)
        return x

    def _train_step( self, optimizer ):

        x = LeafImgCA.make_seed(self.target_size)
        with tf.GradientTape() as t:
            for i in range(self.num_steps):
                x = self(x)
                
            loss = self.loss(x, 'gram')
            print('loss=',loss)
        #variables = t.watched_variables()
        variables = self.trainable_variables
               
        grads = t.gradient(loss, variables)
        #grads = [g/(tf.norm(g)+1e-8) for g in grads]
        optimizer.apply_gradients(zip(grads, variables))
        return loss

    def train( self, lr=1e-6, num_epochs= 5 ):

        
        optimizer = tf.keras.optimizers.Adam(lr)
        loss_log = []
        for e in tqdm(range(num_epochs)):
            
            loss = self._train_step(optimizer)
            loss_log.append(loss.numpy())
        
        return loss_log


    def predict( self ):
        x = LeafImgCA.make_seed(self.target_size, n=1)
        for _ in range(self.num_steps):
            x = self(x)
        return x
    

    

    