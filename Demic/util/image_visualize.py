import numpy as np
from PIL import Image


def add_countor(In, Seg, Color=(0, 255, 0)):
    """
    add a segmentation contour to an 2d image
    inputs:
        In: a PIL Image
        Seg: a binary segmentation (PIL Image)
        Color: rgb value for the contour color
    output:
        Out: the input PIL image with contour added
    """
    Out = In.copy()
    [H, W] = In.size
    for i in range(H):
        for j in range(W):
            if(i==0 or i==H-1 or j==0 or j == W-1):
                if(Seg.getpixel((i,j))!=0):
                    Out.putpixel((i,j), Color)
            elif(Seg.getpixel((i,j))!=0 and  \
                 not(Seg.getpixel((i-1,j))!=0 and \
                     Seg.getpixel((i+1,j))!=0 and \
                     Seg.getpixel((i,j-1))!=0 and \
                     Seg.getpixel((i,j+1))!=0)):
                     Out.putpixel((i,j), Color)
    return Out

def add_seeds(In, Seeds):
    """
    add scribbles to a 2d image
    inputs:
        In: a PIL Image
        Seeds: a PIL image of scribbles. 127 for foreground, 255 for background
    output:
        Out: the input PIL image with scribbles added
    """
    Out = In.copy()
    [H, W] = In.size
    for i in range(H):
        for j in range(W):
            if(Seeds.getpixel((i,j))==127):
                Out.putpixel((i,j), (255, 0, 0))
            elif(Seeds.getpixel((i,j))==255):
                Out.putpixel((i,j), (0, 0, 255))
    return Out

