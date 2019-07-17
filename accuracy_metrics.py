# Machine Learning CS 6375 Project
# On Deep Dark Light.

# Team members:
# Swapnil Bansal sxb180020
# Harshel Jain hxj170009
# Vyom Gupta fxv180000
# Lipsa Senapati lxs180002
import argparse
import glob
import os.path as path
import skimage.io
import skimage.measure
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt


if __name__ == '__main__':
    output_Canon = './output_Canon/'
    groundTruth_Canon = './groundTruth_Canon/'
    
    gtArr = glob.glob(groundTruth_Canon + '/1*.png')
    outputArr = glob.glob(output_Canon + '/1*.png')
    n = len(gtArr)
    psnr_list = []
    ssim_list = []
        
    for i in range(n):
        gt = gtArr[i]            # ground truth
        ot = outputArr[i]
        gt_img = skimage.io.imread(gt)
        ot_img = skimage.io.imread(output_Canon + gt.split("/")[2])
        
        psnr = skimage.measure.compare_psnr(gt_img, ot_img)
        ssim = skimage.measure.compare_ssim(gt_img, ot_img, multichannel=True)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        log = '[%4d / %4d] %s PSNR: %8.3f SSIM: %8.3f' \
            % (i, n, gt.split("/")[2], psnr, ssim)
        print(log)

    print('mean PSNR: %.3f' % np.mean(psnr_list))
    print('mean SSIM: %.3f' % np.mean(ssim_list))
