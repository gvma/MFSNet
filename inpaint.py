import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default="data/", help='Directory of raw input images.')
parser.add_argument('--destination', type=str, default="inpainted_ims/", help='Directory where the inpainted images will be saved.')
args = parser.parse_args()

root = args.root
destination = args.destination
if not os.path.exists(destination):
    os.makedirs(destination)

for im in os.listdir(root):
    print('Image: {}'.format(root + im))
    src=cv2.imread(root+im)
    # Convert the original image to grayscale
    print('Converting to grayscale:')
    grayScale = cv2.cvtColor( src, cv2.COLOR_RGB2GRAY )
    # cv2_imshow(grayScale)

    # Kernel for the morphological filtering
    print('Kernel step:')
    kernel = cv2.getStructuringElement(1,(17,17))
    
    # Perform the blackHat filtering on the grayscale image to find the 
    # hair countours
    print('Blackhat filtering:')
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

    # intensify the hair countours in preparation for the inpainting 
    # algorithm
    print('Threshold:')
    ret,thresh2 = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)

    print('Inpainting image:')
    # inpaint the original image depending on the mask
    destination = cv2.inpaint(src,thresh2,1,cv2.INPAINT_TELEA)

    print('Writing image {}'.format(im))
    cv2.imwrite('data/inpaint/inpainted/inpainted_' + im, destination, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    print('############################')
