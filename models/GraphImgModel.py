from hnca.framework.layers import LeafImgCA, HCA
from hnca.framework.losses import StyleLoss, MSELoss
from hnca.framework.utils import load_image, show_image, plot_loss, ReplayBuffer
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

    def __init__( self,  leaf_ca_target, leaf_ca_min_steps=32, leaf_ca_max_steps=96, leaf_ca_loss_type='ot' ):
        super(GraphImgModel,self).__init__()
        
        self.ca_leaf = LeafImgCA( )
        self.leaf_ca_min_steps = leaf_ca_min_steps
        self.leaf_ca_max_steps = leaf_ca_max_steps
        self.target_size = 128
        self.target_img = load_image(leaf_ca_target)[None,:,:,:3]
        if leaf_ca_loss_type in ['gram','ot']:
            self.leaf_ca_loss = StyleLoss( np.copy(self.target_img), leaf_ca_loss_type )
        elif leaf_ca_loss_type == 'mse':
            self.leaf_ca_loss = MSELoss(np.copy(self.target_img) )
        else :
            print("Loss type not supported")

        self.replay_buffer = ReplayBuffer()
        self.replay_buffer.add( self.ca_leaf.make_seed(size=128, n=256))
              
          

    def call(self, x, training=None ):
        
        x = self.ca_leaf(x)
        return x

   # dont enable use_pool option as its implementation is not stable yet 
    def _train_step( self, optimizer, curr_epoch, use_pool, batch_size=1):

        if use_pool:
            x = self.replay_buffer.sample_batch(batch_size)
            #if curr_epoch % 8 == 0:
            x[:1]= LeafImgCA.make_seed(self.target_size)
        else:
            x = LeafImgCA.make_seed(self.target_size)

        step_n = np.random.randint(self.leaf_ca_min_steps, self.leaf_ca_max_steps)
        with tf.GradientTape() as t:
            for i in range(step_n):
                x = self(x)
            loss = self.leaf_ca_loss(tf.identity(x))

        # dont store outputs of initial few epochs as their outputs might not be stable.
        if use_pool :
            self.replay_buffer.add(x.numpy())
                       
        variables = self.trainable_variables
        grads = t.gradient(loss, variables)
        grads = [g/(tf.norm(g)+1e-8) for g in grads]
        optimizer.apply_gradients(zip(grads, variables))
        return loss

    def train( self, lr=1e-3, num_epochs= 5000, use_pool=True, batch_size=1):

        lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000,2000], [lr, lr*0.3, lr*0.3*0.3])
        optimizer = tf.keras.optimizers.Adam(lr_sched)
        loss_log = []
        for e in tqdm(range(num_epochs)):
            
            loss = self._train_step(optimizer, e, use_pool, batch_size)
            loss_log.append(loss.numpy())
        
        return loss_log


    def create( self, num_steps=50 ):
        x = LeafImgCA.make_seed(self.target_size, n=1)
        for _ in range(num_steps):
            x = self(x)
        return x
    

    

    