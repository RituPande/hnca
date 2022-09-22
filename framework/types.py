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

class CellDetector:
  """
    This class detects cells, their member pixels and their centers in an RGB image.
    To detect the cells, it converts RGB image to gray_scale, performs a guassian blur on it
    and uses a cell segmentation algorithm 'otsu' or 'adaptive' to segment the image.
    It returns, the masks of the detected cells, the pixel values of the cells and their centers.
    
    Attributes
    ----------
    min_pixels : int
        Minimum number of pixels constituting a cell
    max_pixels: int
        Minimum number of pixels constituting a cell
    threshold_type : string
        Algorithm to be used for cell segmentation. It can take value of otsu or adaptive.

      
  """

  def __init__(self, min_pixels=20, max_pixels=128*128,threshold_type='adaptive' ):
    assert threshold_type=='adaptive' or threshold_type=='otsu','threshold_type not supported'
    self.min_pixels = min_pixels
    self.max_pixels = max_pixels
    self.threshold_type = threshold_type

  def __call__(self, img ):
    img_mask = self._preprocess_img( img )
    cell_masks, cell_members = self._extract_cells( img_mask )
    return cell_masks, cell_members


  def _preprocess_img(self, img ):
    
    img_gray = rgb2gray(img)
    img_gray = cv2.GaussianBlur(img_gray,(5,5),cv2.BORDER_DEFAULT)
    if self.threshold_type=='otsu':
      thresh_val = threshold_otsu(img_gray)
    else:
      img_gray = np.uint8(img_gray)
      thresh_val = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,41 , -5)

    img_mask = np.where(img_gray > thresh_val, 1, 0)

    # background is larger than foreground
    if np.sum(img_mask==0) < np.sum(img_mask==1):
      img_mask = np.where(img_mask, 0, 1)

    labels, _ = ndimage.label(img_mask)
    # find connected objects.
    for label_ind, label_coords in enumerate(ndimage.find_objects(labels)):
      cell = img_gray[label_coords]
      # Check if the label size is too small
      if np.product(cell.shape) < self.min_pixels or np.product(cell.shape) > self.max_pixels: 
        #print('Label {} is too small! Setting to 0.'.format(label_ind))
        img_mask = np.where(labels==label_ind+1, 0, img_mask)

    return img_mask

  def _fill_cell( self, pix):
    cell = {}

    y_min =  min(pix[0])
    y_max = max(pix[0])
    x_min =  min(pix[1])
    x_max = max(pix[1])
    center_x = (x_min + x_max)//2
    center_y = (y_min + y_max)//2
   
    cell['center'] = (center_y, center_x)
    cell['members'] = np.asarray(pix).T
    return cell

  def _extract_cells(self, img_mask ):

    # Generate the labels
    labels, nlabels = ndimage.label(img_mask)
    cell_masks = []
    cell_members = []
    for label_num in range(1, nlabels+1):
      mask = np.where(labels == label_num, 1, 0)
      cell_masks.append(mask)
      pix = np.where(labels == label_num )
      cell = self._fill_cell(pix)
      cell_members.append(cell)

    return cell_masks, cell_members

    
class ReplayBuffer:
  def __init__(self, max_len=256):
    self.buffer = deque(maxlen=max_len) 

  def add(self, x ) :
    self.buffer.extendleft(x)

  def sample_batch(self, batch_size):
    mini_batch = random.sample( self.buffer, min(len(self.buffer), batch_size))
    mini_batch = np.asarray(mini_batch)
    return mini_batch

class Graph:
    def __init__(self, x, mode='number', k=8, radius=5  ):
        assert mode=='number' or mode=='radius'
        if isinstance(x, list):
          x = map(np.array, x)
          x = np.array(list(x))

        
        self.node_features = x
        self.n_nodes = x.shape[0]
        if mode=='number':
          self.spA = kneighbors_graph(\
                          self.node_features,\
                            n_neighbors=k,\
                              mode='connectivity',\
                                include_self=False) 
        elif mode=='radius':
          self.spA = radius_neighbors_graph(\
                      self.node_features,\
                        radius=radius,\
                          mode='connectivity',\
                            include_self=False )
      

   