from hnca.framework.ca import ImgCA
from hnca.framework.losses import StyleLoss, MSELoss
from hnca.framework.utils import load_image, show_image, plot_loss, create_parent_seed 
from hnca.framework.types import ReplayBuffer
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import Model
from keras.layers import UpSampling2D,Conv2D,AveragePooling2D
from tqdm import tqdm
import keras.backend as K



"""
This class uses the HNCA framework to create a hierachical CA model
   
    
Attributes
----------
    
Methods
-------

"""

class HCAImgModel(Model):

    def __init__( self,\
                 leaf_ca_target,parent_ca_target,\
                      leaf_ca_min_steps=32, leaf_ca_max_steps=96,  parent_ca_min_steps=32, parent_ca_max_steps=96,\
                            leaf_ca_loss_type='ot', parent_ca_loss_type='mse' ):

        super(HCAImgModel,self).__init__()

        self._init_ca_class_members(leaf_ca_target,parent_ca_target,\
                      leaf_ca_min_steps, leaf_ca_max_steps,  parent_ca_min_steps, parent_ca_max_steps)

        self._init_loss_objects(leaf_ca_loss_type, parent_ca_loss_type )

        self._init_fs_layers()
        
        

                      
    def _init_ca_class_members(self, leaf_ca_target,parent_ca_target,\
                      leaf_ca_min_steps, leaf_ca_max_steps, parent_ca_min_steps, parent_ca_max_steps):
                            
        self.leaf_img_target_size = 128
        self.parent_img_target_size = 32
        # n_features for leaf_ca = (n_channels + n_schannels)*4
        self.leaf_ca_model =  ImgCA(n_channels=12,n_schannels=4,target_size=self.leaf_img_target_size, n_features = 64 )
        self.parent_ca_model =  ImgCA(n_channels=16,n_schannels=0, target_size=self.parent_img_target_size, n_features=64 )
        self.leaf_ca_min_steps = leaf_ca_min_steps
        self.leaf_ca_max_steps = leaf_ca_max_steps
        self.parent_ca_min_steps = parent_ca_min_steps
        self.parent_ca_max_steps = parent_ca_max_steps
        self.signaling_factor = 4

        self.leaf_ca_target_img = load_image(leaf_ca_target)[None,:,:,:3]
        self.parent_ca_target_img = load_image(parent_ca_target)[None,:,:,:3]

        self.leaf_replay_buffer = ReplayBuffer()
        self.parent_replay_buffer = ReplayBuffer()
        self.leaf_replay_buffer.add( self.leaf_ca_model.make_seed(size=self.leaf_img_target_size, n=256))

    def _init_loss_objects(self, leaf_ca_loss_type, parent_ca_loss_type ):

        if leaf_ca_loss_type in ['gram','ot']:
            self.leaf_ca_loss = StyleLoss( np.copy(self.leaf_ca_target_img), leaf_ca_loss_type )
        elif leaf_ca_loss_type == 'mse':
            self.leaf_ca_loss = MSELoss()
        else :
            print("Leaf CA Loss type not supported")
        
        if parent_ca_loss_type in ['gram','ot']:
            self.parent_ca_loss = StyleLoss( np.copy(self.parent_ca_target_img), parent_ca_loss_type )
        elif parent_ca_loss_type == 'mse':
            self.parent_ca_loss = MSELoss()
        else :
            print("Parent CA Loss type not supported")


    def _init_fs_layers( self  ):

        self.signal_creator =  Conv2D(filters=4,\
                                        kernel_size=1,\
                                            bias_initializer='glorot_uniform',\
                                                kernel_initializer=tf.keras.initializers.Zeros()) # use linear activation
                                                                                                  # should we use_bias?
        self.feedback = AveragePooling2D(pool_size=(self.signaling_factor,self.signaling_factor), strides= self.signaling_factor )
        self.upscale_signal = UpSampling2D(size=(self.signaling_factor,self.signaling_factor))

        initializer = tf.random_uniform_initializer(minval=-1.0 , maxval=1.0,  )
        self.signal_lr = tf.Variable( initializer(shape=[1], dtype=tf.float32),  trainable=True)


    def _get_signal(self, x  ):
            
        split_sizes = [3, -1] # remove latent channels from RGB channels 
        _, s  = tf.split(x, split_sizes, axis=-1 )
        s = self.signal_creator(s)
        s = self.upscale_signal(s)
        return s


    def call(self, leaf_x, parent_x=None ):

        step_n = np.random.randint(self.leaf_ca_min_steps, self.leaf_ca_max_steps)
        if parent_x is None:
            leaf_x, s = tf.split(leaf_x, [self.leaf_ca_model.n_channels, -1], axis=-1)
            leaf_x = tf.stop_gradient(self.leaf_ca_model.step(leaf_x,s,n_steps=step_n, training_type='hca'))
            parent_x = self.feedback(leaf_x)
            parent_x = self.parent_ca_model.step(parent_x,s=None,n_steps=1,training_type='hca')

        #Get signal from the parent CA
        s = self._get_signal(parent_x)

        #  get leaf CA features and signals
        features, orig_signal = tf.split(leaf_x, [self.leaf_ca_model.n_channels, -1], axis=3 )

        # add signal from parent CA to signal channels of leaf CA
        s += orig_signal* self.signal_lr

        # iterate through all n steps of leaf CA
        leaf_x = self.leaf_ca_model.step(features, s=s, n_steps=1, training_type='hca')

        # report pooled leaf CA channels back to the parent
        parent_x = self.feedback(leaf_x)
        
        parent_x = self.parent_ca_model.step(parent_x, s=None,n_steps=1,training_type='hca')

        return leaf_x, parent_x 

         
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

    def pretrain_parent_ca(self, seed_args, lr=1e-3, num_epochs= 5000,\
                                   use_pool=True, batch_size=4,\
                                     es_patience_cfg=500, lr_patience_cfg=250,\
                                      num_batches_per_epoch=8, ):
      
      optimizer = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-08, )
      history = []
      seed = None
      if seed_args['seed'] is not None:
          seed = seed_args['seed']
          seeds = seed[None,...]
          seeds = np.repeat(seeds, self.parent_replay_buffer.maxlen, axis=0)
          self.parent_replay_buffer.add(seeds)
      else:
          seeds = np.zeros((self.parent_replay_buffer.maxlen,self.parent_img_target_size,self.parent_img_target_size , self.parent_ca_model.n_channels) )
          for i in tf.range(len(seeds)):
                seeds[i] = create_parent_seed(self.leaf_img_target_size,\
                                      self.leaf_img_target_size,\
                                        seed_args['colors'],\
                                           seed_args['bg'],\
                                            self.signaling_factor,\
                                             seed_args['num_circles'],\
                                              seed_args['min_radius'],\
                                                seed_args['max_radius'],\
                                                  n_channels = self.parent_ca_model.n_channels)
          
          self.parent_replay_buffer.add(seeds)
          
      
      es_patience = es_patience_cfg
      lr_patience = lr_patience_cfg
      min_loss = np.inf
    
      for e in tqdm(tf.range(num_epochs)):
        batch_loss = 0
        for _ in tf.range(num_batches_per_epoch):
          loss, tape = self._loss_step_parent_ca(e, use_pool, batch_size, seed_args)
          batch_loss += loss
          variables = self.parent_ca_model.trainable_variables
          grads = tape.gradient(loss, variables)
          grads = [g/(tf.norm(g)+1e-8) for g in grads]
          optimizer.apply_gradients(zip(grads, variables))
          
        batch_loss /= num_batches_per_epoch 
        history.append(batch_loss.numpy())

        if batch_loss + 1e-6 < min_loss:
          min_loss = batch_loss
          print("min_loss:",min_loss )
          early_stopping_patience = es_patience_cfg
          lr_patience = lr_patience_cfg
          best_model_weights = self.get_weights()
        else:
          early_stopping_patience -= 1
          lr_patience -= 1
          print("early_stopping_patience:",early_stopping_patience," lr_patience:",lr_patience )
          if early_stopping_patience == 0:
            self.set_weights(best_model_weights)
            break
            
        if lr_patience == 0:
          K.set_value(optimizer.lr, optimizer.lr * 0.1)
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

    def _loss_step_parent_ca(self, curr_epoch, use_pool, batch_size, seed_args ):

        if use_pool:
            x = self.parent_replay_buffer.sample_batch(batch_size)
            if curr_epoch % 8 == 0:
              if seed_args['seed'] is not None :
                x[:1] = seed_args['seed']
              else:
                x[:1] = create_parent_seed(self.leaf_img_target_size,\
                                      self.leaf_img_target_size,\
                                        seed_args['colors'],\
                                           seed_args['bg'],\
                                            self.signaling_factor,\
                                             seed_args['num_circles'],\
                                              seed_args['min_radius'],\
                                                seed_args['max_radius'],\
                                                  n_channels = self.parent_ca_model.n_channels)
        else:
            if seed_args['seed'] is not None :
              x[:1] = seed_args['seed']
            else:
              x[:1] = create_parent_seed(self.leaf_img_target_size,\
                                      self.leaf_img_target_size,\
                                        seed_args['colors'],\
                                           seed_args['bg'],\
                                            self.signaling_factor,\
                                             seed_args['num_circles'],\
                                              seed_args['min_radius'],\
                                                seed_args['max_radius'],\
                                                  n_channels = self.parent_ca_model.n_channels)

        step_n = np.random.randint(self.parent_ca_min_steps, self.parent_ca_max_steps)
        with tf.GradientTape() as t:
            x = self.parent_ca_model.step(x, None , n_steps=step_n, training_type='parent')
            #overflow loss forces the model to output values within -1.0 and 1.0
            overflow_loss = tf.reduce_sum(tf.abs(x - tf.clip_by_value(x, -1.0, 1.0)))
            loss = self.parent_ca_loss( self.parent_ca_target_img, x, is_image=True ) # + overflow_loss*1e2
        if use_pool :
          self.parent_replay_buffer.add(x.numpy())

        return loss, t

    def _loss_step_hca(self, curr_epoch, use_pool, batch_size, loss_weightage):
        
        if use_pool:
            leaf_x = self.leaf_replay_buffer.sample_batch(batch_size)
            if curr_epoch % 8 == 0:
              leaf_x[:1]= self.leaf_ca_model.make_seed(self.leaf_img_target_size)
        else:
            leaf_x = self.leaf_ca_model.make_seed(self.leaf_img_target_size)

        step_n = np.random.randint(self.parent_ca_min_steps, self.parent_ca_max_steps)
        with tf.GradientTape() as t:
            leaf_x, parent_x = self(leaf_x, None )
            for _ in tf.range(step_n-1):
                leaf_x, parent_x = self(leaf_x, parent_x)
            loss_parent = self.parent_ca_loss(self.parent_ca_target_img, parent_x, is_image=True )
            loss_leaf = self.leaf_ca_loss(tf.identity(leaf_x) )
            loss_hca = loss_parent*loss_weightage[0] + loss_leaf*loss_weightage[1]
            
        if use_pool :
          self.leaf_replay_buffer.add(leaf_x.numpy())

        return loss_leaf, loss_parent, loss_hca, t


    def train_hca( self, lr=1e-3, num_epochs= 2000, use_pool=True,\
                      batch_size=4, es_patience_cfg=500, lr_patience_cfg=250,\
                        loss_weightage=[10,1] ):

        optimizer_leaf_ca = tf.keras.optimizers.Adam(learning_rate=lr)
        optimizer_hca = tf.keras.optimizers.Adam(learning_rate=lr)
        hca_history = []
        leaf_ca_history = []
        parent_ca_history = []
        early_stopping_patience = es_patience_cfg
        lr_patience = lr_patience_cfg
        min_loss = np.inf
        for e in tqdm(tf.range(num_epochs)):
            loss_leaf, loss_parent, loss_hca, hca_tape = self._loss_step_hca( e, use_pool, batch_size, loss_weightage )
            hca_history.append(loss_hca.numpy())
            parent_ca_history.append(loss_parent.numpy())
            leaf_ca_history.append(loss_leaf.numpy())
            
            gradients_hca = hca_tape.gradient(loss_hca, self.trainable_variables) 
            optimizer_hca.apply_gradients(zip(gradients_hca, self.trainable_variables))

            if e % 4 == 0:
              loss_leaf_ca,leaf_ca_tape = self._loss_step_leaf_ca( e,use_pool, batch_size )
              gradients_leaf_ca = leaf_ca_tape.gradient(loss_leaf_ca, self.leaf_ca_model.trainable_variables)
              optimizer_leaf_ca.apply_gradients(zip(gradients_leaf_ca, self.leaf_ca_model.trainable_variables))
            
            if loss_hca + 1e-6 < min_loss:
              min_loss = loss_hca
              print("min_loss:",min_loss )
              early_stopping_patience = es_patience_cfg
              lr_patience = lr_patience_cfg
              best_model_weights = self.get_weights()
            else:
              early_stopping_patience -= 1
              lr_patience -= 1
              print("early_stopping_patience:",early_stopping_patience," lr_patience:",lr_patience )
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
    

    

    
