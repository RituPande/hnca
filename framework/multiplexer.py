import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.layers import Dense

class SimpleMultiplexer(Model):

  def __init__(self, n_features ):
    super(SimpleMultiplexer,self).__init__()
    self.n_features = n_features
    initializer = tf.random_uniform_initializer(minval=-1.0 , maxval=1.0  )
    self.lr = tf.Variable( initializer(shape=[1,n_features], dtype=tf.float32),  trainable=True)
    
  def call(self, x_src, x_dst ):
    out = x_dst + self.lr*x_src
    return out

class CrossAttMultiplexer(Model):

  def __init__(self, d ):
    super(CrossAttMultiplexer,self).__init__()

    self.d = d
    self.Q = Dense(units=self.d, use_bias=False) 
    self.K = Dense(units=self.d, use_bias=False) 
    self.V = Dense(units=1, use_bias=False) 

    
  def call(self, x_src, x_dst ):
   
      b_x, h_x, w_x, c_x =   x_dst.shape

      x_reshaped = tf.reshape(x_dst, (b_x*w_x*h_x,c_x,1) )

      b_s, w_s, h_s, c_s = x_src.shape

      s_reshaped = tf.reshape(x_src, ( b_s*w_s*h_s,c_s, 1) )

      q = self.Q(x_reshaped)
      k = self.K(s_reshaped)
      v = self.V(s_reshaped)

      k_T = tf.transpose(k, perm=[0,2,1])

      scaling_f = tf.math.sqrt(tf.cast(self.d, dtype=tf.float32))
      alpha = tf.nn.softmax( (q @ k_T)/scaling_f, axis = -1 )

      x_new_reshaped = tf.reduce_sum(alpha*v, axis=-1)

      out = tf.reshape(x_new_reshaped, (b_x,h_x, w_x, c_x ))

      return out
