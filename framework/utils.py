import tensorflow as tf  
import matplotlib.pyplot as plt
import numpy as np
import PIL
from collections import deque
import random 
from collections import deque
import random 

class ReplayBuffer:
  def __init__(self, max_len=256 ):
    self.buffer = deque(maxlen=max_len) 

  def add(self, x ) :
    self.buffer.extendleft(x)

  def sample_batch(self, batch_size):
    mini_batch = random.sample( self.buffer, min(len(self.buffer), batch_size))
    mini_batch = np.asarray(mini_batch)
    return mini_batch



def load_image( image_path, max_size = 128):  
  img =  PIL.Image.open(image_path)
  img.thumbnail((max_size, max_size), PIL.Image.ANTIALIAS)
  img = np.float32(img)
  return img

def show_image( img ):
  img = PIL.Image.fromarray(np.uint8(img) , mode='RGB') 
  display(img)
  #img.show()

def to_rgba(x):
  return x[..., :4]

#def to_alpha(x):
#  return tf.clip_by_value(x[..., 3:4], 0.0, 1.0)

def to_rgb(x):
  return x[..., :3]+ 0.5

  # assume rgb premultiplied by alpha
  #rgb, a = x[..., :3], to_alpha(x)
  #return 1.0-a+rgb
def plot_loss(loss_log):
  plt.figure(figsize=(10, 4))
  plt.title('Loss history ')
  plt.plot(loss_log,"o" , alpha=0.8)
  plt.show()

