from collections import deque
import random 
from collections import deque
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
import cv2
import numpy as np
import scipy
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph

class ReplayBuffer(deque):
  def __init__(self, max_len=256):
    super().__init__(maxlen=max_len) 

  def add(self, x ) :
    self.extendleft(x)

  def sample_batch(self, batch_size):
    mini_batch = random.sample( self, min(len(self), batch_size))
    mini_batch = np.asarray(mini_batch)
    return mini_batch
