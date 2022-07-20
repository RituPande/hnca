import tensorflow as tf  
import matplotlib.pyplot as plt
import numpy as np
import PIL

def load_image( image_path, max_size = 224):  
  img =  PIL.Image.open(image_path)
  img.thumbnail((max_size, max_size), PIL.Image.ANTIALIAS)
  img = np.float32(img)
  return img

def show_image( img ):
  img = PIL.Image.fromarray(np.uint8(img), 'RGB' ) 
  img.show()

def to_rgba(x):
  return x[..., :4]

def to_rgb(x):
  return x[..., :3]

def plot_loss(loss_log):
  plt.figure(figsize=(10, 4))
  plt.title('Loss history (log10)')
  plt.plot(np.log10(loss_log), '.', alpha=0.8)
  plt.show()

