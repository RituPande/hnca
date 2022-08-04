from hnca.framework.layers import LeafImgCA, HCA
from hnca.framework.losses import StyleLoss, MSELoss
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

    def __init__( self, num_steps, leaf_ca_target, leaf_ca_loss_type='gram' ):
        super(GraphImgModel,self).__init__()
        
        self.ca_leaf = LeafImgCA( )
        self.num_steps = num_steps
        self.target_size = 128
        self.target_img = load_image(leaf_ca_target)[None,:,:,:3]
        if leaf_ca_loss_type in ['gram','ot']:
            self.leaf_ca_loss = StyleLoss( np.copy(self.target_img), leaf_ca_loss_type )
        elif leaf_ca_loss_type == 'mse':
            self.leaf_ca_loss = MSELoss(np.copy(self.target_img) )
        else :
            print("Loss type not supported")

        self.replay_buffer = self.ca_leaf.make_seed(size=128, n=256)
              
          

    def call(self, x, training=None ):
        
        x = self.ca_leaf(x)
        return x

    
    def _train_step( self, optimizer, use_pool, batch_size=2, use_seed=True):

        if use_pool:
            idx = np.random.choice(len(self.replay_buffer), batch_size ) # select random index's from the replay_buffer
            x = self.replay_buffer[idx]
            if use_seed:
                x[0] = np.squeeze(LeafImgCA.make_seed(self.target_size))
        else:
            x = LeafImgCA.make_seed(self.target_size)

        with tf.GradientTape() as t:
            for i in range(self.num_steps):
                x = self(x)

            loss = self.leaf_ca_loss(tf.identity(x))
            if use_pool:
                self.replay_buffer[idx] = x.numpy()
        
        variables = self.trainable_variables
               
        grads = t.gradient(loss, variables)
        #grads = [g/(tf.norm(g)+1e-8) for g in grads]
        optimizer.apply_gradients(zip(grads, variables))
        return loss

    def train( self, lr=1e-6, num_epochs= 5000, use_pool=False, batch_size=2, seeding_epoch_multiple=2 ):

        optimizer = tf.keras.optimizers.Adam(lr)
        loss_log = []
        for e in tqdm(range(num_epochs)):
            use_seed = True if e % seeding_epoch_multiple == 0 else False
            loss = self._train_step(optimizer, use_pool, batch_size, use_seed)
            loss_log.append(loss.numpy())
        
        return loss_log


    def create( self, num_steps=50 ):
        x = LeafImgCA.make_seed(self.target_size, n=1)
        for _ in range(num_steps):
            x = self(x)
        return x
    

    

    