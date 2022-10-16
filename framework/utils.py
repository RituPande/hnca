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


def plot_loss(history1, history2=None, y_label1='Leaf CA history', y_label2=None):
  fig = plt.figure(figsize=(10, 4))
  plt.title('log10(Loss) history')

  ax1 = fig.add_subplot(111)
  ax1.plot(np.log10(history1), 'bo',alpha=0.5)
  ax1.set_ylabel(y_label1,color='b')
  #ax1.set_ylim(np.min(history1), history1[0])
  for tl in ax1.get_yticklabels():
    tl.set_color('b')
  

  if history2 is not None:
    ax2 = ax1.twinx()
    ax2.plot(np.log10(history2), 'ro', alpha=0.5)
    ax2.set_ylabel(y_label2,color='r')
    #ax2.set_ylim(np.min(history2), history2[0])

    for tl in ax2.get_yticklabels():
      tl.set_color('r')
  
  plt.show()


    
    
