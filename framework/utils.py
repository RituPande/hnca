import tensorflow as tf  
import matplotlib.pyplot as plt
import numpy as np
import PIL
from collections import deque
import random 
from collections import deque
import random 
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray

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
  plt.ylim(np.min(loss_log), loss_log[0])
  plt.plot(loss_log,"o" , alpha=0.5)
  plt.show()

def find_cells( img ):
  img_gray = rgb2gray(img)
    
  # Mask out background and extract connected objects
  thresh_val = threshold_otsu(img_gray)
  mask = np.where(img_gray > thresh_val, 1, 0)
  
  labels, nlabels = ndimage.label(mask)

  # find connected objects.
  for label_ind, label_coords in enumerate(ndimage.find_objects(labels)):
    cell = img_gray[label_coords]
    # Check if the label size is too small
    if np.product(cell.shape) < 10: 
      print('Label {} is too small! Setting to 0.'.format(label_ind))
      mask = np.where(labels==label_ind+1, 0, mask)

  # create mask for each component 
  cell_masks = []
  for label_num in range(1, nlabels+1):
    mask = np.where(labels == label_num, 1, 0)
    cell_masks.append(mask)

  # Regenerate the labels
  labels, nlabels = ndimage.label(mask)
  print('There are now {} separate components / objects detected.'.format(nlabels))
    
