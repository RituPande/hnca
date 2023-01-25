#!/usr/bin/env python

"""
    This script is used to generate target image in 'rgba' format for leaf CA. The background is always white with alpha 
    channel 0 ( fully transparent). The foreground includes  non-overlapping circles with random centers and radii with alpha channel
    set to 255( fully opaque). It saves the image in an output file, if requested.
        
"""

import cv2
import numpy as np
import random
import argparse 
from skimage.measure import block_reduce
import tensorflow as tf
from keras.layers import AveragePooling2D
import PIL
from IPython.display import display


"""
Description:
Checks if a circle provided in 'test_c' paramter overlaps with any circle in parameter'circle'  

Input:
test_c: tuple (x,y,r) where x,y is center of the circle and r radius
circles: list of tuples (x,y,r)

RetVal:
'False' if test_c does not overlap with any circle in 'circles'. 'True' otherwise.

"""

def collides( test_c, circles):

    test_x, test_y, test_r = test_c
    ret_val = False
    for c in circles:
        x, y, r = c
        if np.sqrt((test_x -x)**2 + ( test_y - y)**2) < (test_r + r):
            ret_val = True
            break

    return ret_val

"""
Description:
Generates non-overlapping circles  with random centers and radii.
 
Input:
image_width: Width of the image
image_height: Height of the image
num_circles: number of circles to generate
min_radius, max_radius: A circle radius is randomly selected min_radius and max_radius.
    To have circles of same radii make min_radius equal to max_radius

RetVal:
List containing  'num_circle' tuples (x,y,r)

"""
def gen_circle(img_height, img_width, n, r=None, center_y=None, center_x=None ):


    if center_y is not None and center_x is not None:
      assert img_height and img_width > 20 and n %2 == 0
      center_y = img_height//2
      center_x = img_width//2

    if r is None: r = min(center_y, center_x) - 5

    theta = (2*np.pi)/n

    y = center_y
    x = center_x + r

    points = np.zeros((n,2))
    points[0,0] = y
    points[0,1] = x

    for i in range(1,n):
        y = center_y + r* np.sin(theta*i)
        x = center_x + r* np.cos(theta*i)
        points[i,0]=int(round(y))
        points[i,1]=int(round(x))
    
    return points

def gen_circles_in_quad(img_height, img_width, n ):
    points_1 = gen_circle(img_height/2, img_width/2, n//4 )
    points_2 = gen_circle(img_height/2, img_width/2, n//4 )
    points_2[:,1] += img_width/2
    points_3 = gen_circle(img_height/2, img_width/2, n//4 )
    points_3[:,0] += img_height/2
    points_4 = gen_circle(img_height/2, img_width/2, n//4 )
    points_4[:,1] += img_width/2
    points_4[:,0] += img_height/2

    points =  np.vstack( (points_1, points_2, points_3, points_4))
    return points
    

def  gen_concentric_circle(image_height, image_width, num_points, r=None):

    points_1 = gen_circle(image_height, image_width, num_points//4, 10 )
    points_2 = gen_circle(image_height, image_width, num_points//4, 20 )
    points_3 = gen_circle(image_height, image_width, num_points//4, 40 )
    points_4 = gen_circle(image_height, image_width, num_points//4, 60 )
    points =  np.vstack( (points_1, points_2, points_3,points_4 ))
    return points


def create_random_circles(image_width, image_height, num_circles, min_radius, max_radius):

    #initiatlize an image with white background

    circles = []
    while len(circles) < num_circles:
        r = random.randint(min_radius, max_radius)
        x = random.randint(r, image_width -r )
        y = random.randint(r, image_height-r )
        
        if not collides( (x,y,r), circles):
            circles.append((x,y,r))

    #print(circles)
    return circles

"""
Description:
Generates unique colors to fill the circles.
 
Input:
num_colors: Number of unique colors to generate
RetVal:
List containing  'num_colors' tuples (r,g,b)

"""
def get_unique_colors(num_colors):

    colors = [] 
    while len(colors) < num_colors :
        r,g,b = random.sample (range(0,255), 3)
        if (r,g,b) not in colors:
            colors.append((r,g,b))
    return colors

def fillCircles( img, circles, colors, target_points):
    
    color_index = 0
    circle_index = 0
    num_colors = len(colors)
    for c in circles:
        x,y,r = c
        if target_points is not None:
            y = int(target_points[circle_index][0])
            x = int(target_points[circle_index][1])
        cv2.circle(img, (x,y) , r, colors[color_index], cv2.FILLED)
        color_index += 1
        if color_index % num_colors == 0:
            color_index = 0
        circle_index += 1
    return img


def fillStars(img, circles, colors):

    circle_index = 0
    num_colors = len(colors)
    for c in circles:
        x,y,r = c
        points = gen_circle(img_height=None, img_width=None, n=6, r=r, center_y=y, center_x=x )
        cv2.drawContours(img,[points[0],points[2],points[4] ], 0, colors[circle_index],  -1 )
        cv2.drawContours(img,[points[1],points[3],points[5] ], 0, colors[circle_index],  -1 )

        color_index += 1
        if color_index % num_colors == 0:
            color_index = 0
        circle_index += 1


def create_star_image(image_width = 224 ,image_height= 224, num_circles=10, num_colors = 2, min_radius=5, max_radius=10, bg=52, save_img=True, target_points=None, colors=None):

  img = np.full((image_height,image_width, 3),bg , dtype=np.uint8 )
  noise = np.random.randint(0,5,size=(image_height,image_width, 3), dtype=np.uint8)
    
  circles = create_random_circles(image_width, image_height, num_circles, min_radius, max_radius)
  if colors is None:
    colors = get_unique_colors(num_colors)
  img = fillStars(img, circles, colors)
  img += noise
  if save_img:
        cv2.imwrite('../img/hca_target_img.png',img)



"""
Description:
This is the main function that draws the image and saves it in a file, if requested.
 
Input:
Refer script arguments
RetVal:
None
"""


def create_image(image_width = 224 ,image_height= 224, num_circles=10, num_colors = 2, min_radius=5, max_radius=10, bg=52, save_img=True, target_points=None, colors=None ):

    # generate leaf CA target image
    img = np.full((image_height,image_width, 3),bg , dtype=np.uint8 )
    noise = np.random.randint(0,5,size=(image_height,image_width, 3), dtype=np.uint8)
    
    circles = create_random_circles(image_width, image_height, num_circles, min_radius, max_radius)
    if colors is None:
      colors = get_unique_colors(num_colors)
    img = fillCircles(img, circles, colors, None)
    img += noise
    #cv2.imshow('image', img)
    
    if save_img:
        cv2.imwrite('../img/leaf_ca_target_img.png',img)

    # generate HCA target image
    img = np.full((image_height,image_width, 3),bg , dtype=np.uint8 )
    img = fillCircles(img, circles, colors, target_points)
    img += noise
    #cv2.imshow('image', img)
    
    if save_img:
        cv2.imwrite('../img/hca_target_img.png',img)
    
    # generate leaf CA feedback
    img = img[None,...]
    img = AveragePooling2D(pool_size=(4,4) )(tf.cast(img, dtype=tf.float32)) 
    #img = block_reduce(img,(2,2,1), func=np.mean)
    #img = block_reduce(img,(2,2,1), func=np.mean)
   
    img = np.squeeze(img.numpy())
    #cv2.imshow('image', img)
    
    if save_img:
        cv2.imwrite('../img/hca_pooled_img.png',img)
   
    

def create_pooled_img( img_path ):
  img = cv2.imread(img_path.strip())
  img = img[None,...]
  img = AveragePooling2D(pool_size=(4,4) )(tf.cast(img, dtype=tf.float32))
  img = np.squeeze(img.numpy())
  cv2.imwrite('hnca/img/leaf_ca_pooled_img_1.png',img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--image_width", help = "Enter image width")
    parser.add_argument("-H", "--image_height", help = "Enter image height")
    parser.add_argument("-n", "--num_circles", help = "Enter number of circles")
    parser.add_argument("-c", "--num_colors", help = "Enter number of different circle colors")
    parser.add_argument("-r", "--min_radius", help = "Enter minimum circle radius")
    parser.add_argument("-R", "--max_radius", help = "Enter maximum circle radius")
    parser.add_argument("-b", "--background", help = "Enter background color of image (0/255)")
    parser.add_argument("-s", "--save_image", help = "Enter (y/n)")
    parser.add_argument("-t", "--target", help ="1- Single 2- Concentric, 3- Circles in quadrants")
    parser.add_argument("-p", "--pool", help ="path of image to be pooled")

    args = parser.parse_args()

    image_width= int(args.image_width) if args.image_width else 128
    image_height= int(args.image_height) if args.image_height else 128
    num_circles= int( args.num_circles) if args.num_circles else 10
    num_colors= int(args.num_colors) if args.num_colors else 2
    min_radius= int(args.min_radius) if args.min_radius else 5
    max_radius= int(args.max_radius) if args.max_radius else 10
    background = int(args.background) if args.background else 0 
    target = int(args.target) if args.target else 0 
    save_img= True if args.save_image=='y' else False

    points = None
    if target == 1:
        points = gen_circle(image_height, image_width, num_circles )    
    elif target==2:
        points = gen_concentric_circle( image_height, image_width, num_circles )
    elif target == 3:
        points = gen_circles_in_quad(image_height, image_width, num_circles )
    elif target == 4:
        colors = [(9,230,199), ( 250, 3, 185)]
        create_star_image(image_width,image_height,num_circles,num_colors,min_radius,max_radius,background, save_img, None, colors)

    else:
        print("No target input")
    if target != 0 or target != 4:
      colors = [(9,230,199), ( 250, 3, 185)]
      create_image(image_width,image_height,num_circles,num_colors,min_radius,max_radius,background, save_img, points, colors)
    if args.pool:
      create_pooled_img(args.pool)


