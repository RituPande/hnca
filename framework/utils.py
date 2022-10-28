import tensorflow as tf  
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import PIL
from keras.layers import AveragePooling2D


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

def zoom(img, scale=4):
  img = np.repeat(img, scale, 0)
  img = np.repeat(img, scale, 1)
  return img


def collides( test_c, circles):

    test_x, test_y, test_r = test_c
    ret_val = False
    for c in circles:
        x, y, r = c
        d = np.sqrt( (test_x - x)**2 + (test_y - y)**2 )
        if d < (test_r + r):
            ret_val = True
            break

    return ret_val

def fillCircles( img, circles, colors):
    num_colors = len(colors)
    color_index = 0
    circle_index = 0
    for c in circles:
        x,y,r = c
        cv2.circle(img, (x,y) , r, colors[color_index], cv2.FILLED)
        color_index += 1
        if color_index % num_colors == 0:
            color_index = 0
        circle_index += 1
    return img


def create_random_circles(image_width, image_height, num_circles, min_radius, max_radius):

    circles = []
    while len(circles) < num_circles:
        r = random.randint(min_radius, max_radius)
        x = random.randint(r, image_width -r )
        y = random.randint(r, image_height-r )
        
        if not collides( (x,y,r), circles):
            circles.append((x,y,r))

    #print(circles)
    return circles

def create_parent_seed(image_height,image_width, colors, bg, scale, num_circles, min_radius, max_radius):
    img = np.full((image_height,image_width, 16),bg ,dtype=np.uint8 )
    circles = create_random_circles(image_width, image_height, num_circles, min_radius, max_radius)
    fillCircles(img[...,:3], circles, colors)
    img = img[None,...]
    img = AveragePooling2D(pool_size=(scale,scale) )(tf.cast(img, dtype=tf.float32)) 
    return img[0]
    
    
