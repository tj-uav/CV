# The goal of this module is to take one coherent image and to split it into many smaller images
# These smaller images will be rotated in random ways with noise, simulating a real image data set that needs stitching

from PIL import Image 
import os
from sys import argv
from random import random

def image_split(rows, columns, buffer):
    """
    Splits image into specified number of rows and columns \n
    Specify the image via a command line argument: \n
    python Mapsplitter.py maps/{insert img name}

    arguments -> \n
    rows: number of rows \n
    columns: number of columns \n
    buffer: size of overlap in ALL direction (pixels)
    """

    # extract image name from directory
    im = Image.open(argv[1]) 
    filename = os.path.basename(argv[1])
    im_name = os.path.splitext(f"{filename}")[0]

    rect_width = int(im.width / columns)
    rect_height = int(im.height / rows)

    split_images_dir = mkdir(im_name)
        
    im_num = 1


    # generate buffers, or overlap regions
    left_buffers = [buffer for i in range(int(im.width / rect_width) - 1)]
    left_buffers.insert(0, 0)

    top_buffers = [buffer for i in range(int(im.height / rect_height) - 1)]
    top_buffers.insert(0, 0)

    right_buffers = [buffer for i in range(int(im.width / rect_width) - 1)]
    right_buffers.append(0)

    bottom_buffers = [buffer for i in range(int(im.height / rect_height) - 1)]
    bottom_buffers.append(0)

    # starting index
    i = j = 0

    # make the starting x position of the rectangle go from 0 to to the image width - rect width
    # add one to make sure it includes the final rectanlge
    for left in range(0, im.width - rect_width + 1, rect_width):

        # y index is zero upon reaching a new column
        j = 0

        # same logic as iterating through x-pos
        for top in range(0, im.height - rect_height + 1, rect_height):

            right = left + rect_width
            bottom = top + rect_height

            im1 = im.crop((left - left_buffers[i], top - top_buffers[j], right + right_buffers[i], bottom + bottom_buffers[j]))

            # add rotation noise to dataset
            # im1 = im1.rotate(5 * (random() - 0.5), Image.NEAREST, expand = 1)

            im1.save(f"{split_images_dir}/{im_name}_{i:03}_{j:03}.png")
            im_num += 1

            # move to the next row
            j += 1
        
        # move to the next column
        i += 1

def mkdir(im_name):
    i = 1
    if not os.path.exists(f"split_images/{im_name}_v{i}"):
        os.mkdir(f"split_images/{im_name}_v{i}")
    
    else:
        i += 1
        while os.path.exists(f"split_images/{im_name}_v{i}"):
            i += 1

        os.mkdir(f"split_images/{im_name}_v{i}")

    im_dir = f"split_images/{im_name}_v{i}"

    return im_dir

image_split(5, 5, 100)
