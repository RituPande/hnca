#!/usr/bin/env python

"""
    This script is used to generate target image in 'rgba' format for leaf CA. The background is always white with alpha 
    channel 0 ( fully transparent). The foreground includes  non-overlapping circles with random centers and radii with alpha channel
    set to 255( fully opaque). It saves the image in an output file, if requested.
        
"""

import cv2
import numpy as np
import random
from math import dist
import argparse 

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
        if dist([test_x, test_y], [x,y] ) < (test_r + r):
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

def create_circles(image_width, image_height, num_circles, min_radius, max_radius):

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
List containing  'num_colors' tuples (r,g,b,a) where a is always 255

"""
def get_unique_colors(num_colors):

    colors = [] 
    while len(colors) < num_colors :
        r,g,b = random.sample (range(255), 3)
        if (r,g,b) not in colors:
            #make the alpha channel opaque
            colors.append((r,g,b,255))
    return colors

"""
Description:
This is the main function that draws the image and saves it in a file, if requested.
 
Input:
Refer script arguments
RetVal:
None
"""


def create_image(image_width = 224 ,image_height= 224, num_circles=10, num_colors = 2, min_radius=5, max_radius=10, save_img = False ):

    img = np.full((image_height,image_width, 4),255, dtype=np.uint8 )
    # make alpha channel transparent.
    img[...,3] = 0

    circles = create_circles(image_width, image_height, num_circles, min_radius, max_radius)
    colors = get_unique_colors(num_colors)
    color_index = 0

   
    for c in circles:
        x,y,r = c
        cv2.circle(img, (x,y) , r, colors[color_index], cv2.FILLED)
        color_index += 1
        if color_index % num_colors == 0:
            color_index = 0

    if save_img:
     cv2.imwrite('../img/target_img.png',img)

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--image_width", help = "Enter image width")
    parser.add_argument("-H", "--image_height", help = "Enter image height")
    parser.add_argument("-n", "--num_circles", help = "Enter number of circles")
    parser.add_argument("-c", "--num_colors", help = "Enter number of different circle colors")
    parser.add_argument("-r", "--min_radius", help = "Enter minimum circle radius")
    parser.add_argument("-R", "--max_radius", help = "Enter maximum circle radius")
    parser.add_argument("-s", "--save_image", help = "Enter (y/n)")

    args = parser.parse_args()

    image_width= int(args.image_width) if args.image_width else 224
    image_height= int(args.image_height) if args.image_height else 224
    num_circles= int( args.num_circles) if args.num_circles else 10
    num_colors= int(args.num_colors) if args.num_colors else 2
    min_radius= int(args.min_radius) if args.min_radius else 5
    max_radius= int(args.max_radius) if args.max_radius else 10
    save_image= True if args.save_image == "y" else False

    create_image(image_width,image_height,num_circles,num_colors,min_radius,max_radius,save_image)
    

