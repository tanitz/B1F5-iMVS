import base64
import os
import queue
import sys
import threading
import time
import urllib
from ctypes import *
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def FIND_POINT2(CropImage):
    ed = cv.ximgproc.createEdgeDrawing()
    EDParams = cv.ximgproc_EdgeDrawing_Params()
    EDParams.MinPathLength = 5    
    EDParams.PFmode = False         
    EDParams.MinLineLength = 5    
    EDParams.NFAValidation = True 
    ed.setParams(EDParams)
    gray = cv.cvtColor(CropImage, cv.COLOR_BGR2GRAY)
    gray = cv.bilateralFilter(gray, 9, 75, 75)
    ed.detectEdges(gray)
    ellipses = ed.detectEllipses()
    DetectedCircles = []
    if ellipses is not None: 
        for i in range(len(ellipses)):
            center = (int(ellipses[i][0][0]), int(ellipses[i][0][1]))
            center_x, center_y = (int(ellipses[i][0][0]), int(ellipses[i][0][1]))
            axes = (int(ellipses[i][0][2])+int(ellipses[i][0][3]),int(ellipses[i][0][2])+int(ellipses[i][0][4]))
            angle = ellipses[i][0][5]
            major_axis = axes[0]
            minor_axis = axes[1]
            if abs(major_axis-minor_axis) < 15 :
                diameter = (major_axis + minor_axis) / 2.0
                DetectedCircles.append([center,axes,angle])
    return DetectedCircles

def FIND(image):
    Image_output = cv.imread(image)
    K=1
    ROI_Crop3 = [180,700,200,800]
    Dcircle=[]
    esrc = Image_output.copy()
    crop3 = esrc[ROI_Crop3[0]//K:ROI_Crop3[1]//K, ROI_Crop3[2]//K:ROI_Crop3[3]//K]
    DetectedCircles = FIND_POINT2(crop3)
    if len(DetectedCircles) > 0:
        max_index = max(range(len(DetectedCircles)), key=lambda i: max(DetectedCircles[i][1]))
        print(image,"  max",max_index)
        
        center,axes,angle = DetectedCircles[max_index]
        centerX_circle, centerY_circle = center
        R1,R2 = axes
        D1 = R1+R2
        print(D1)
        if D1 > 20  :
            Dcircle.append(D1)
            cv.ellipse(Image_output, (centerX_circle+ROI_Crop3[2]//2, centerY_circle+ROI_Crop3[0]//2), axes, angle,0, 360, (0,0,255), 2, cv.LINE_AA)
            cv.ellipse(crop3, (centerX_circle, centerY_circle), axes, angle,0, 360, (0,0,255), 2, cv.LINE_AA)
    cv.imshow('Image_output', crop3)
    cv.waitKey(0)

