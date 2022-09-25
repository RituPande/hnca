#!/usr/bin/env python
import numpy as np
import argparse 

def gen_circle(img_height, img_width, n ):

    assert img_height and img_width > 20 and n %2 == 0

    center_y = img_height//2
    center_x = img_width//2

    r = min(center_y, center_x) - 10

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

    np.save('../hca_targets/circle.npy',points)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--image_width", help = "Enter image width")
    parser.add_argument("-H", "--image_height", help = "Enter image height")
    parser.add_argument("-n", "--num_points", help = "Enter number of points")
    #parser.add_argument("-t", "--target", help = "Enter target shape(c)")

    args = parser.parse_args()

    image_width= int(args.image_width) if args.image_width else 128
    image_height= int(args.image_height) if args.image_height else 128
    num_points= int( args.num_points) if args.num_points else 20

    gen_circle(image_height, image_width, num_points )
    
    

