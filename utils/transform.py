import cv2
import numpy as np
import matplotlib.pyplot as plt

def center_crop(img, width,height):
    y, x = img.shape

    x_center = x/2.0
    y_center = y/2.0

    x_min = int(x_center - width/2.0)
    x_max = x_min + width
    if x_min < 0:
        x_min = 0
    if x_max > x:
        x_max = x

    y_min = int(y_center - height/2.0)
    y_max = y_min + height

    if y_min < 0:
        y_min = 0
    if y_max > y:
        y_max = y

    img_cropped = img[y_min:y_max, x_min: x_max]
        
    return img_cropped

def image_minmax(img):
    img_minmax = ((img - np.min(img)) / (np.max(img) - np.min(img))).copy()
    img_minmax = (img_minmax * 255).astype(np.uint8)
        
    return img_minmax
