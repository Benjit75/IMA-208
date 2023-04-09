#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 15:55:37 2022

@author: ckervazo
"""

import numpy as np
import cv2
from scipy.interpolate import griddata

#%%
def PSNR(im1,im2):
    """
    Computes the PSNR between im1 and im2. The two images must have the same size.

    Parameters
    ----------
    im1, im2 : nparray
        Two images.

    Returns
    -------
    psnr : float
    """

    MSE = 1 / (im1.size) * np.sum((im1 - im2) ** 2)
    psnr = 10 * np.log10(255 ** 2 / MSE)
    
    return psnr
