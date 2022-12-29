from hnca.framework.ca import ImgCA
from hnca.framework.losses import StyleLoss, MSELoss
from hnca.framework.utils import load_image, show_image, plot_loss, create_parent_seed 
from hnca.framework.types import ReplayBuffer
from hnca.framework.comm import Sensor, Actuator
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import Model
from tqdm import tqdm
import keras.backend as K
import os, glob



"""
This class uses the HNCA framework to create a hierachical CA model
   
    
Attributes
----------
    
Methods
-------

"""

class HCAImgModel(Model):

    def __init__( self,leaf_ca_target,parent_ca_target,\
                          comm_cfg_params,\
                            leaf_img_target_size=128,parent_img_target_size=32,\
                                leaf_ca_min_steps=32, leaf_ca_max_steps=96,\
                                    parent_ca_min_steps=32, parent_ca_max_steps=96,\
                                        hca_min_steps=32, hca_max_steps=96,\
                                            leaf_ca_loss_type='ot', parent_ca_loss_type='mse',\
                                             n_leaf_ca_channels=3, n_leaf_ca_schannels=9,\
                                                n_parent_ca_channels=12 ):

        super(HCAImgModel,self).__init__()

        self.leaf_img_target_size = leaf_img_target_size
        self.parent_img_target_size = parent_img_target_size
        # n_features for leaf_ca = (n_channels + n_schannels)*4
        self.leaf_ca_model =  ImgCA(n_channels=n_leaf_ca_channels,\
                                        n_schannels=n_leaf_ca_schannels,\
                                          target_size=self.leaf_img_target_size,\
                                             n_features = 64 )
                                             
        self.parent_ca_model =  ImgCA(n_channels=n_parent_ca_channels,\
                                        n_schannels=0,\
                                          target_size=self.parent_img_target_size, \
                                             n_features=64 )

        self.sensor = Sensor( n_leaf_ca_channels, n_parent_ca_channels,\
                                signal_factor= comm_cfg_params['signal_factor'],\
                                 n_sig_creation_layers=comm_cfg_params['n_sensor_sig_creation_layers'],\
                                  all_ch_src=comm_cfg_params['sensor_all_ch_src'],\
                                    all_ch_dst=comm_cfg_params['sensor_all_ch_dst'],\
                                      multiplex_type=comm_cfg_params['sensor_multiplex_type'])

        self.actuator = Actuator(n_leaf_ca_channels, n_leaf_ca_schannels,\
                                  signal_factor= comm_cfg_params['signal_factor'],\
                                    n_sig_creation_layers=comm_cfg_params['n_actuator_sig_creation_layers'],\
                                      all_ch_src=comm_cfg_params['actuator_all_ch_src'],\
                                        all_ch_dst=comm_cfg_params['actuator_all_ch_dst'],\
                                          multiplex_type=comm_cfg_params['actuator_multiplex_type'])

        self.leaf_ca_min_steps = leaf_ca_min_steps
        self.leaf_ca_max_steps = leaf_ca_max_steps
        self.parent_ca_min_steps = parent_ca_min_steps
        self.parent_ca_max_steps = parent_ca_max_steps
        self.hca_min_steps = hca_min_steps
        self.hca_max_steps = hca_max_steps
        
        
        self.leaf_ca_target_img = load_image(leaf_ca_target)[None,:,:,:3]
        self.parent_ca_target_img = load_image(parent_ca_target)[None,:,:,:3]

        self.leaf_replay_buffer = ReplayBuffer()
        self.parent_replay_buffer = ReplayBuffer(max_len=20000)
        self.leaf_replay_buffer.add( self.leaf_ca_model.make_seed(size=self.leaf_img_target_size, n=256))

        self._init_loss_objects(leaf_ca_loss_type, parent_ca_loss_type )

        
    def _init_loss_objects(self, leaf_ca_loss_type, parent_ca_loss_type ):

        if leaf_ca_loss_type is None:
          self.leaf_ca_loss= None
        elif leaf_ca_loss_type in ['gram','ot']:
            self.leaf_ca_loss = StyleLoss( np.copy(self.leaf_ca_target_img), leaf_ca_loss_type )
        elif leaf_ca_loss_type == 'mse':
            self.leaf_ca_loss = MSELoss()
        else :
            print("Leaf CA Loss type not supported")
        
        if parent_ca_loss_type is None:
            self.parent_ca_loss = None 
        elif parent_ca_loss_type in ['gram','ot']:
            self.parent_ca_loss = StyleLoss( np.copy(self.parent_ca_target_img), parent_ca_loss_type )
        elif parent_ca_loss_type == 'mse':
            self.parent_ca_loss = MSELoss()
        else :
            print("Parent CA Loss type not supported")


    def call( self, leaf_x, parent_x):
        # create leaf_x and parent_x for the first time
        if parent_x is None:
          step_n = np.random.randint(self.leaf_ca_min_steps, self.leaf_ca_max_steps)
          leaf_x = tf.stop_gradient(self.leaf_ca_model.step(leaf_x,s=None,n_steps=step_n, training_type='hca'))
          parent_x = self.sensor(tf.identity(leaf_x) , None )
          parent_x = self.parent_ca_model.step(parent_x, s=None, n_steps=1, update_rate=1.0, training_type='hca')
        else:
          # leaf_ca  is actuated from current parent ca state
          leaf_channels, leaf_schannels = self.actuator( tf.identity(parent_x), tf.identity(leaf_x) )
          #parent ca detects feedback from current leaf ca state
          parent_x = self.sensor(tf.identity(leaf_x) , tf.identity(parent_x) )
          #take 1 step of leaf and parent ca with new signals in each direction
          leaf_x = self.leaf_ca_model.step(leaf_channels, s=leaf_schannels, n_steps=1, update_rate=1.0, training_type='hca')
          parent_x = self.parent_ca_model.step(parent_x, s=None, n_steps=1, update_rate=1.0, training_type='hca')

        return leaf_x, parent_x 

    """
    def call(self, leaf_x, parent_x=None ):
        
        step_n = np.random.randint(self.leaf_ca_min_steps, self.leaf_ca_max_steps)
        if parent_x is None:
            leaf_x, s = tf.split(leaf_x, [self.leaf_ca_model.n_channels, -1], axis=-1)
            leaf_x = tf.stop_gradient(self.leaf_ca_model.step(leaf_x,s,n_steps=step_n, training_type='hca'))
            parent_x = self.sensor(leaf_x , None )
            parent_x = self.parent_ca_model.step(parent_x,s=None,n_steps=1, update_rate=1.0, training_type='hca')

        #Get signal from the parent CA and mix with leaf CA 
        leaf_channels, leaf_schannels = self.actuator( parent_x, leaf_x )
                                                      
        leaf_x = self.leaf_ca_model.step(leaf_channels, s=leaf_schannels, n_steps=1, update_rate=1.0, training_type='hca')

        # Report pooled leaf CA signal back to the 
        parent_x = self.sensor(leaf_x, None  )

        parent_x = self.parent_ca_model.step(parent_x, s=None, n_steps=1, update_rate=1.0, training_type='hca')

        return leaf_x, parent_x 
    """
         
    def pretrain_leaf_ca( self, lr=1e-3, num_epochs= 5000, use_pool=True, batch_size=4):

        lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000,2000], [lr, lr*0.3, lr*0.3*0.3])
        optimizer = tf.keras.optimizers.Adam(lr_sched, epsilon=1e-08)
        history = []
        for e in tqdm(range(num_epochs)):
            loss, tape = self._loss_step_leaf_ca(e, use_pool, batch_size)
            variables = self.leaf_ca_model.trainable_variables
            grads = tape.gradient(loss, variables)
            grads = [g/(tf.norm(g)+1e-8) for g in grads]
            optimizer.apply_gradients(zip(grads, variables))
            history.append(loss.numpy())
        
        return history

    def pretrain_parent_ca(self, seed_args, start_epoch=0, lr=1e-3, num_epochs= 5000,\
                                   use_pool=True, batch_size=4,\
                                     es_patience_cfg=500, lr_patience_cfg=250,\
                                      num_batches_per_epoch=8, min_loss=np.inf ):
      
      optimizer = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-08, )
      history = []
      seed = None
      
      seeds = seed_args['seed']
      repeat_count = self.parent_replay_buffer.maxlen//len(seeds)
      repeated_seeds = np.repeat(seeds, repeat_count, axis=0)
      self.parent_replay_buffer.add(repeated_seeds)
      add_count = self.parent_replay_buffer.maxlen % len(seeds)
      if add_count: self.parent_replay_buffer.add(seeds[-add_count:,...])
                
      self.parent_replay_buffer.add(seeds)
          
      es_patience = es_patience_cfg
      lr_patience = lr_patience_cfg
      best_model_weights = self.get_weights()
      
      for e in tqdm(tf.range(start=start_epoch, limit=num_epochs)):
        batch_loss = 0
        for b in tf.range(num_batches_per_epoch):
          loss, tape = self._loss_step_parent_ca(e, b, use_pool, batch_size, seed_args)
          batch_loss += loss
          variables = self.parent_ca_model.trainable_variables
          grads = tape.gradient(loss, variables)
          grads = [g/(tf.norm(g)+1e-8) for g in grads]
          optimizer.apply_gradients(zip(grads, variables))
          
        batch_loss /= num_batches_per_epoch 
        history.append(batch_loss.numpy())

        if batch_loss + 1e-6 < min_loss:
          min_loss = batch_loss
          print("min_loss:",min_loss.numpy() )
          es_patience = es_patience_cfg
          lr_patience = lr_patience_cfg
          best_model_weights = self.get_weights()
          for filename in glob.glob("gdrive/MyDrive/chkpt/parent_ca_wghts*"):os.remove(filename)
     
          self.parent_ca_model.save_weights(f"gdrive/MyDrive/chkpt/parent_ca_wghts_{e}_{optimizer.lr.numpy():0.2e}_{min_loss.numpy():0.2f}_chkpt.h5" )
          #best_opt_weights = optimizer.get_weights()
        else:
          es_patience -= 1
          lr_patience -= 1
          print("loss:",batch_loss.numpy()," es_patience:",es_patience," lr_patience:",lr_patience )
          if es_patience == 0:
            self.set_weights(best_model_weights)
            break
            
        if lr_patience == 0:
          #optimizer = self._reload_optimizer(optimizer,\
          #                       self.parent_ca_model.trainable_variables,\
          #                          best_opt_weights)
          K.set_value(optimizer.lr, optimizer.lr * 0.1)
          print("New lr:",optimizer.lr)
          lr_patience = lr_patience_cfg
          self.set_weights(best_model_weights)
                        
      return history

    def _loss_step_leaf_ca(self, curr_epoch, use_pool, batch_size):

        if use_pool:
            x = self.leaf_replay_buffer.sample_batch(batch_size)
            if curr_epoch % 8 == 0:
              x[:1]= self.leaf_ca_model.make_seed(self.leaf_img_target_size)
        else:
            x = self.leaf_ca_model.make_seed(self.leaf_img_target_size)

        x,s = tf.split(x,[self.leaf_ca_model.n_channels,-1], axis=-1)
        step_n = np.random.randint(self.leaf_ca_min_steps, self.leaf_ca_max_steps)
        with tf.GradientTape() as t:
            x = self.leaf_ca_model.step(x, s, n_steps=step_n, training_type='leaf')
            #overflow loss forces the model to output values within -1.0 and 1.0
            overflow_loss = tf.reduce_sum(tf.abs(x - tf.clip_by_value(x, -1.0, 1.0)))
            loss = self.leaf_ca_loss(tf.identity(x)) # + overflow_loss*1e2
        if use_pool :
          self.leaf_replay_buffer.add(x.numpy())

        return loss, t

    def _loss_step_parent_ca(self, curr_epoch, batch_no,  use_pool, batch_size, seed_args ):

        
        x = self.parent_replay_buffer.sample_batch(batch_size)
        if seed_args['reseed_freq'] != 0 and \
            curr_epoch % seed_args['reseed_freq'] == 0:

          k = np.random.randint(0, len(seed_args['seed']))
          x[0] = seed_args['seed'][k]
          
        
        step_n = np.random.randint(self.parent_ca_min_steps, self.parent_ca_max_steps)
        with tf.GradientTape() as t:
            x = self.parent_ca_model.step(x, None , n_steps=step_n, update_rate=1.0,training_type='parent')
            #overflow loss forces the model to output values within -1.0 and 1.0
            overflow_loss = tf.reduce_sum(tf.abs(x - tf.clip_by_value(x, -1.0, 1.0)))
            loss = self.parent_ca_loss( self.parent_ca_target_img, x, is_image=True ) # + overflow_loss*1e2
       
        self.parent_replay_buffer.add(x.numpy())

        return loss, t

    def _loss_step_hca(self, curr_epoch, use_pool, batch_size, loss_weightage, reseed_freq):
        
        if use_pool:
            leaf_x = self.leaf_replay_buffer.sample_batch(batch_size)
            if reseed_freq !=0 and curr_epoch % reseed_freq == 0:
              leaf_x[:1]= self.leaf_ca_model.make_seed(self.leaf_img_target_size)
        else:
            leaf_x = self.leaf_ca_model.make_seed(self.leaf_img_target_size)

        step_n = np.random.randint(self.hca_min_steps, self.hca_max_steps)
        
        with tf.GradientTape() as t:
            leaf_x, parent_x = self(leaf_x, None )
            for _ in tf.range(step_n-1):
                leaf_x, parent_x = self(leaf_x, parent_x)
            loss_parent = self.parent_ca_loss(self.parent_ca_target_img, parent_x, is_image=True )
            loss_leaf = self.leaf_ca_loss(tf.identity(leaf_x) ) if loss_weightage[1] else tf.constant(0.0, dtype=tf.float32)
            loss_hca = loss_parent*loss_weightage[0] + loss_leaf*loss_weightage[1]

        if use_pool :
          self.leaf_replay_buffer.add(leaf_x.numpy())

        return loss_leaf, loss_parent, loss_hca, t


    def train_hca( self, lr=1e-3, num_epochs= 2000, use_pool=True,\
                      batch_size=4, es_patience_cfg=500, lr_patience_cfg=250,\
                        loss_weightage=[10,1], leaf_training_freq=4, reseed_freq=1 ):

        optimizer_leaf_ca = tf.keras.optimizers.Adam(learning_rate=lr)
        optimizer_hca = tf.keras.optimizers.Adam(learning_rate=lr)
        hca_history = []
        leaf_ca_history = []
        parent_ca_history = []
        early_stopping_patience = es_patience_cfg
        lr_patience = lr_patience_cfg
        min_loss = np.inf
        for e in tqdm(tf.range(num_epochs)):
            loss_leaf, loss_parent, loss_hca, hca_tape = self._loss_step_hca( e, use_pool, batch_size, loss_weightage , reseed_freq)
            hca_history.append(loss_hca.numpy())
            parent_ca_history.append(loss_parent.numpy())
            leaf_ca_history.append(loss_leaf.numpy())
            training_vars = self.parent_ca_model.trainable_variables + self.sensor.trainable_variables + self.actuator.trainable_variables
            gradients_hca = hca_tape.gradient(loss_hca,  training_vars, unconnected_gradients=tf.UnconnectedGradients.ZERO ) 
            grads = [g/(tf.norm(g)+1e-8) for g in gradients_hca]
            optimizer_hca.apply_gradients(zip(grads, training_vars))

            if leaf_training_freq!=0 and  e % leaf_training_freq == 0:
              loss_leaf_ca,leaf_ca_tape = self._loss_step_leaf_ca( e,use_pool, batch_size )
              gradients_leaf_ca = leaf_ca_tape.gradient(loss_leaf_ca, self.leaf_ca_model.trainable_variables)
              grads = [g/(tf.norm(g)+1e-8) for g in gradients_leaf_ca]
              optimizer_leaf_ca.apply_gradients(zip(grads, self.leaf_ca_model.trainable_variables))
            
            if loss_hca + 1e-6 < min_loss:
              min_loss = loss_hca
              print("min_loss:",min_loss.numpy() )
              early_stopping_patience = es_patience_cfg
              lr_patience = lr_patience_cfg
              best_model_weights = self.get_weights()
            else:
              early_stopping_patience -= 1
              lr_patience -= 1
              print("loss:",loss_hca.numpy(),"early_stopping_patience:",early_stopping_patience," lr_patience:",lr_patience )
              if early_stopping_patience == 0:
                self.set_weights(best_model_weights)
                break
            if lr_patience == 0:
                K.set_value(optimizer_hca.lr, optimizer_hca.lr * 0.1)
                K.set_value(optimizer_leaf_ca.lr, optimizer_leaf_ca.lr * 0.1)
                lr_patience = lr_patience_cfg
                self.set_weights(best_model_weights)
                print("lr:",optimizer_hca.lr )

           
        return leaf_ca_history, parent_ca_history, hca_history


    def step( self, x_initial = None, num_steps=50 ):
        leaf_x = x_initial
        
        if leaf_x is None:
          leaf_x = self.leaf_ca_model.make_seed(self.leaf_img_target_size, n=1)
         
        leaf_x, parent_x = self(leaf_x,None)
        for _ in tf.range(num_steps-1):
            leaf_x, parent_x = self(leaf_x, parent_x)

        return leaf_x
    

    

    
