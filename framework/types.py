from collections import deque
import random 
from collections import deque
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
import cv2
import numpy as np
import scipy

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
      thresh_val = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,25,-10)

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
  def __init__(self, max_len=256 ):
    self.buffer = deque(maxlen=max_len) 

  def add(self, x ) :
    self.buffer.extendleft(x)

  def sample_batch(self, batch_size):
    mini_batch = random.sample( self.buffer, min(len(self.buffer), batch_size))
    mini_batch = np.asarray(mini_batch)
    return mini_batch

class Graph:
    def __init__(self, node_features, k=3, normalize=False):

        x = map(np.array, node_features)
        x = np.array(list(x))
        self.node_features = x
        self.n_nodes = len(node_features)
        self.n_edges = None
        self.A, self.spA = self._prepare_adjacency_matrix( k, normalize )
        
    def _prepare_adjacency_matrix(self, k, normalize):
        dist = np.zeros((self.n_nodes,self.n_nodes))
        A = np.zeros((self.n_nodes,self.n_nodes))

        for i in range(self.n_nodes):
            for j in range( i+1, self.n_nodes):
                dist[i,j] = np.linalg.norm(self.node_features[i]- self.node_features[j] )
                dist[j,i] = dist[i,j]

        nn =  np.argsort(dist)
        for i in range(self.n_nodes):
            for j in range(k+1):
                neighbor = nn[i,j]
                A[i,neighbor] = 1

        # normalize adjacency matrix
        if normalize:
            deg = np.sum(A, axis=-1)
            A = A/deg

        return A, scipy.sparse.csr_matrix(A)