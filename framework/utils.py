import tensorflow as tf  
import matplotlib.pyplot as plt
import numpy as np
import PIL

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


def plot_loss(loss_log):
  plt.figure(figsize=(10, 4))
  plt.title('Loss history ')
  plt.ylim(np.min(loss_log), loss_log[0])
  plt.plot(loss_log,"o" , alpha=0.5)
  plt.show()


    
    
