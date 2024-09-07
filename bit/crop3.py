# -- coding: utf-8 --
"""
Created on 2021.01.03
hik grab images

@author: yanziwei
"""
import base64
import os
import queue
import sys
import termios
import threading
import time
import urllib
from ctypes import *

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

import tqdm
from boxx import *
from PIL import Image
from skimage import io
import socket

sys.path.append("/opt/MVS/Samples/64/Python/MvImport")
from MvCameraControl_class import *
g_bExit = False
image_return = None

class HikMvCamera:
    def __init__(self):

        # ch:创建相机实例 | en:Creat Camera Object
        self.nConnectionNum = self.find_device()
        self.cam = {}
        self.stParam = {}
        for i in range(self.nConnectionNum):
            self.cam[str(i)] = MvCamera()
            self.cam[str(i)] = self.create_cam(self.cam[str(i)], i)
            self.cam[str(i)], self.stParam[str(i)] = self.set_info(self.cam[str(i)])

    def set_info(self, cam):

        # ch:设置触发模式为off | en:Set trigger mode as off
        ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            # print("set trigger mode fail! ret[0x%x]" % ret)
            sys.exit()

        # #设置PixelFormat为RGB-8
        ret = cam.MV_CC_SetEnumValue("PixelFormat", 0x02180014)
        if ret != 0:
            # print("set RBG8 fail! ret[0x%x]" % ret)
            sys.exit()

        # 设置ROI宽高
        # ret = cam.MV_CC_SetIntValue("Width", 5472) #5472
        # if ret != 0:
        #     print("set width fail! ret[0x%x]" % ret)
        #     sys.exit()
        # ret = cam.MV_CC_SetIntValue("Height", 3648) #3648
        # if ret != 0:
        #     print("set height fail! ret[0x%x]" % ret)
        #     sys.exit()
        # # 设置ROI偏移位置
        # ret = cam.MV_CC_SetIntValue("OffsetX", 2096) #2096
        # if ret != 0:
        #     print("set width fail! ret[0x%x]" % ret)
        #     sys.exit()
        # ret = cam.MV_CC_SetIntValue("OffsetY", 1184) #1184
        # if ret != 0:
        #     print("set height fail! ret[0x%x]" % ret)
        #     sys.exit()
        # 设置曝光
        # ret = cam.MV_CC_SetFloatValue("ExposureTime", 50000)
        # if ret != 0:
        #     print("set ExposureTime fail! ret[0x%x]" % ret)
        #     sys.exit()

        # ret = cam.MV_CC_SetIntValue("GevSCPD", 1000)
        # if ret != 0:
        #     # print("set GevSCPD fail! ret[0x%x]" % ret)
        #     sys.exit()

        # 设置SDK图像缓存个数
        # cam.MV_CC_SetImageNodeNum(10)
        # ch:获取数据包大小 | en:Get payload size

        stParam = MVCC_INTVALUE()
        memset(byref(stParam), 0, 2 * sizeof(MVCC_INTVALUE))
        ret = cam.MV_CC_GetIntValue("PayloadSize", stParam)
        if ret != 0:
            # print("get payload size fail! ret[0x%x]" % ret)
            sys.exit()

        return cam, stParam

    def find_device(self):
        # 创建相机
        # 获得版本号
        self.SDKVersion = MvCamera.MV_CC_GetSDKVersion()
        print("SDKVersion[0x%x] :v3.0" % self.SDKVersion)
        self.deviceList = MV_CC_DEVICE_INFO_LIST()
        self.tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
        tlayerType = self.tlayerType
        deviceList = self.deviceList
        # ch:枚举设备 | en:Enum device
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            print("enum devices fail! ret[0x%x]" % ret)
            sys.exit()

        if deviceList.nDeviceNum == 0:
            print("find no device!")
            sys.exit()

        print("Find %d devices!" % deviceList.nDeviceNum)
        for i in range(0, deviceList.nDeviceNum):
            mvcc_dev_info = cast(
                deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)
            ).contents
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                print("\ngige device: [%d]" % i)
                strModeName = ""
                for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                    strModeName = strModeName + chr(per)
                print("device model name: %s" % strModeName)

                nip1 = (
                    mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xFF000000
                ) >> 24
                nip2 = (
                    mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00FF0000
                ) >> 16
                nip3 = (
                    mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000FF00
                ) >> 8
                nip4 = mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000FF
                print("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))

        return deviceList.nDeviceNum

    def create_cam(self, cam, nConnectionNum):

        deviceList = self.deviceList
        # ch:选择设备并创建句柄| en:Select device and create handle
        stDeviceList = cast(
            deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)
        ).contents

        ret = cam.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            print("create handle fail! ret[0x%x]" % ret)
            sys.exit()

        # ch:打开设备 | en:Open device
        ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            print("open device fail! ret[0x%x]" % ret)
            sys.exit()

        # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
        if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
            nPacketSize = cam.MV_CC_GetOptimalPacketSize()
        if int(nPacketSize) > 0:
            ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
            if ret != 0:
                print("Warning: Set Packet Size fail! ret[0x%x]" % ret)
        else:
            print("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

        return cam

    def get_frame(self, img_root=None):
        global image_return
        now_time = time.time()
        local_time = time.localtime(now_time)
        dt = time.strftime("%y%m%d_%H%M%S_", local_time)

        for cam_id, cam in self.cam.items():
            ret = cam.MV_CC_StartGrabbing()
            if ret != 0:
                print("start grabbing fail! ret[0x%x]" % ret)
                sys.exit()
            # print(cam_id)

            nPayloadSize = self.stParam[cam_id].nCurValue
            data_buf = (c_ubyte * nPayloadSize)()
            pData = byref(data_buf)
            stFrameInfo = MV_FRAME_OUT_INFO_EX()
            memset(byref(stFrameInfo), 0, 2*sizeof(stFrameInfo))
            img_save_name = img_root + dt + str(cam_id) + ".jpg"
            ret = cam.MV_CC_GetOneFrameTimeout(pData, nPayloadSize, stFrameInfo, 1000)
            if ret == 0:
    
                image = np.asarray(pData._obj)
                image = image.reshape(
                	(stFrameInfo.nHeight, stFrameInfo.nWidth, 3))
                image= cv.cvtColor(image,cv.COLOR_RGB2BGR)
                image_return = image
            # else:
        
                # print("no data[0x%x]" % ret)

            ret = cam.MV_CC_StopGrabbing()
            if ret != 0:
                print("stop grabbing fail! ret[0x%x]" % ret)
                del data_buf
                sys.exit()

        return image_return

    def close_cam(self):
        for cam in self.cam.values():

            # ch:关闭设备 | Close device
            ret = cam.MV_CC_CloseDevice()
            if ret != 0:
                print("close deivce fail! ret[0x%x]" % ret)
                del data_buf
                sys.exit()

            # ch:销毁句柄 | Destroy handle
            ret = cam.MV_CC_DestroyHandle()
            if ret != 0:
                print("destroy handle fail! ret[0x%x]" % ret)
                del data_buf
                sys.exit()
        print("Stop grab image")
        # del data_buf

    def DericheFilter(img):
        dximg = cv.ximgproc.GradientDericheX(	img, 50/100., 1000/1000.	)
        dyimg = cv.ximgproc.GradientDericheY(	img, 50/100., 1000/1000.	)
        dx2=dximg*dximg
        dy2=dyimg*dyimg
        module = np.sqrt(dx2+dy2)
        cv.normalize(src=module,dst=module,norm_type=cv.NORM_MINMAX)
        module_8bit = cv.convertScaleAbs(module, alpha=255.0)  # แปลงภาพให้เป็น 8-bit
        return module_8bit
    # return module

    def sharpen_edges(img):
        # Define a sharpening kernel
        sharpening_kernel = np.array([[-1, -1, -1],
                                    [-1,  11, -1],
                                    [-1, -1, -1]])
        
        # Apply the kernel to the image
        sharpened_img = cv.filter2D(img, -1, sharpening_kernel)
        return sharpened_img

    def remove_noise(img):
        # Apply a median blur to remove noise
        denoised_img = cv.medianBlur(img, 5)  # You can change the kernel size
        return denoised_img

    def apply_threshold(img, lower_thresh, upper_thresh):
        # Apply binary thresholding
        _, thresh_img = cv.threshold(img, lower_thresh, upper_thresh, cv.THRESH_BINARY)
        return thresh_img

def FIND_POINT1(CropImage):
    ed1 = cv.ximgproc.createEdgeDrawing()
    EDParams = cv.ximgproc_EdgeDrawing_Params()
    EDParams.MinPathLength = 100    # try changing this value between 5 to 1000
    EDParams.PFmode = True         # defaut value try to swich it to True
    EDParams.MinLineLength = 10    # try changing this value between 5 to 100
    EDParams.NFAValidation = True   # defaut value try to swich it to False
    ed1.setParams(EDParams)
    ed1.detectEdges(CropImage)
    lines = ed1.detectLines()
    Detectedline = []
    # print("lines :",len(lines)) 
    min_x = 5000
    min_y = 5000
    min_point = None     
    if lines is not None: 
        lines = np.uint16(np.around(lines))
        # print("lines :",len(lines))
        for i in range(len(lines)):
            DY1 = np.int16(lines[i][0][1])
            DY2 = np.int16(lines[i][0][3])
            DX1 = np.int16(lines[i][0][0])
            DX2 = np.int16(lines[i][0][2])
            # cv.line(CropImage, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 1, cv.LINE_AA)
            if abs(DX1 - DX2) <= 4 :
                # Y1,Y2,X1,X2 = 131,740,1200,1700         #CROP 2
                if (abs(DY1-DY2) >40 ):
                    cv.line(CropImage, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 1, cv.LINE_AA)
                    Detectedline.append(lines[i])
                    if min_x > DX1:
                        min_x = DX1
                    if min_x > DX2:
                        min_x = DX2
                    if min_y > DY1:
                        min_y = DY1
                    if min_y > DY2:
                        min_y = DY2
                    min_point = (min_x,min_y)
                    # print(Detectedline)
    return Detectedline , min_point    

def FIND_POINT2(CropImage):
    ed = cv.ximgproc.createEdgeDrawing()
    EDParams = cv.ximgproc_EdgeDrawing_Params()
    EDParams.MinPathLength = 5    # try changing this value between 5 to 1000
    EDParams.PFmode = False         # defaut value try to swich it to True
    EDParams.MinLineLength = 5    # try changing this value between 5 to 100
    EDParams.NFAValidation = True   # defaut value try to swich it to False
    ed.setParams(EDParams)
    CropImage = cv.cvtColor(CropImage, cv.COLOR_BGR2GRAY)
    # gray = cv.GaussianBlur(gray, (9,9), 0)
    # gray = cv.bilateralFilter(gray,15,55,55)
    ed.detectEdges(CropImage)
    ellipses = ed.detectEllipses()
    # print("ellipses :",len(ellipses))
    DetectedCircles = []
    if ellipses is not None: # Check if circles and ellipses have been found and only then iterate over these and add them to the image
        for i in range(len(ellipses)):
            if len(ellipses) > 2 and i == 1:
                i = 2
            center = (int(ellipses[i][0][0]), int(ellipses[i][0][1]))
            center_x, center_y = (int(ellipses[i][0][0]), int(ellipses[i][0][1]))
            axes = (int(ellipses[i][0][2])+int(ellipses[i][0][3]),int(ellipses[i][0][2])+int(ellipses[i][0][4]))
            angle = ellipses[i][0][5]
            major_axis = axes[0]
            minor_axis = axes[1]
            if abs(major_axis-minor_axis) < 2 :
                # print(major_axis-minor_axis)
                diameter = (major_axis + minor_axis) / 2.0
                # print("diameter :",diameter)
                DetectedCircles.append([center,axes,angle])

    return DetectedCircles

def press_any_key_exit():
    fd = sys.stdin.fileno()
    old_ttyinfo = termios.tcgetattr(fd)
    new_ttyinfo = old_ttyinfo[:]
    new_ttyinfo[3] &= ~termios.ICANON
    new_ttyinfo[3] &= ~termios.ECHO
    # sys.stdout.write(msg)
    # sys.stdout.flush()
    termios.tcsetattr(fd, termios.TCSANOW, new_ttyinfo)
    try:
        os.read(fd, 7)
    except:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, old_ttyinfo)

def grab_img():
    Nameimage = "Stream"
    cap = HikMvCamera()
    WINDOW_NAME = "hik detector"
    Pixel_size = 1.0
    data = True
    cv.namedWindow('Controls', cv.WINDOW_NORMAL)

    cv.createTrackbar('Blur ksize', 'Controls', 0, 10, lambda x: update_image())
    cv.createTrackbar('Blur sigma', 'Controls', 0, 50, lambda x: update_image())
    cv.createTrackbar('Canny thresh1', 'Controls', 127, 255, lambda x: update_image())
    cv.createTrackbar('Canny thresh2', 'Controls', 250, 255, lambda x: update_image())
    while True:

        K = 2
        img_root = "./grab_img/"
        start_time = time.time()
        image = cap.get_frame(img_root)
        src = image.copy()
        c1 = src.copy()
        c2 = src.copy()
        c3 = src.copy()
        color_arrow = (0, 0, 255)
        color_text = (255, 0, 0)
        height, width, channels = src.shape
        src = cv.resize(src, (width//K,height//K), interpolation = cv.INTER_AREA)
        Image_output = src.copy()
        # client_handler = threading.Thread(target=handle_client_connection, args=(client_socket,))
        # client_handler.start()        
            
        data = True
        ROI_Crop2 = [500,700,700,1300] 
        ROI_Crop1 = [1150,1350,700,1300]
        ROI_Crop3 = [1500,2200,700,1300]
        
        i1 = src[ROI_Crop1[0]//K:ROI_Crop1[1]//K, ROI_Crop1[2]//K:ROI_Crop1[3]//K]
        i2 = src[ROI_Crop2[0]//K:ROI_Crop2[1]//K, ROI_Crop2[2]//K:ROI_Crop2[3]//K]
        i3 = src[ROI_Crop3[0]//K:ROI_Crop3[1]//K, ROI_Crop3[2]//K:ROI_Crop3[3]//K]
        
        blur_ksize = cv.getTrackbarPos('Blur ksize', 'Controls') * 2 + 1
        blur_sigma = cv.getTrackbarPos('Blur sigma', 'Controls') / 10.0
        canny_thresh1 = cv.getTrackbarPos('Canny thresh1', 'Controls')
        canny_thresh2 = cv.getTrackbarPos('Canny thresh2', 'Controls')
        # cv.imshow("crop1", i1)

        # cv.imshow("crop2", i2)
        # cv.imshow("crop3", i3)
        # cv.waitKey(0)
        # break
        
        # cv.imshow("c", edges)
# **********************************************************************************************
        Dcircle=[]
        esrc = src.copy()
        # Y1,Y2,X1,X2 = 1050,1600,800,1700          #CROP 3
        crop3 = esrc[ROI_Crop3[0]//K:ROI_Crop3[1]//K, ROI_Crop3[2]//K:ROI_Crop3[3]//K]

        _, binary_image = cv.threshold(crop3, canny_thresh1, canny_thresh2, cv.THRESH_BINARY)
        CropImagebinary_image = cv.cvtColor(binary_image, cv.COLOR_BGR2GRAY)
        DetectedCircles = FIND_POINT2(binary_image)
        # print("DetectedCircles :",len(DetectedCircles))
        for i in range(len(DetectedCircles)):
            center,axes,angle = DetectedCircles[i]
            centerX_circle, centerY_circle = center
            # print("Center Circle :",centerX_circle, centerY_circle)
            R1,R2 = axes
            D1 = R1+R2
            if D1 > 100 :
                Dcircle.append(D1)
                print("Diameter Circle :",D1)
                cv.ellipse(Image_output, (centerX_circle+ROI_Crop3[2]//2, centerY_circle+ROI_Crop3[0]//2), axes, angle,0, 360, (0,0,255), 2, cv.LINE_AA)
                cv.ellipse(crop3, (centerX_circle, centerY_circle), axes, angle,0, 360, (0,0,255), 2, cv.LINE_AA)
            # cv.circle(Image_output, (centerX_circle+ROI_Crop3[2]//2, centerY_circle+ROI_Crop3[0]//2), 2, (0,0,255), 0)
        line_length = 200

        cv.imshow("crop3", crop3)  
        cv.imshow("binary_image", CropImagebinary_image)  

    # **********************************************************************************************
            
            # break
    # **********************************************************************************************
        # except :
        #     print("Error 3")
        #     h1, w1 = Image_output.shape[:2]
        #     cv.imshow('result',cv.resize(Image_output,(w1//2,h1//2)))
            
        if cv.waitKey(1) & 0xFF == ord('q'):
            client_socket.close()
            server_socket1.close()
            break

    cap.stop_cam()

def handle_client_connection(client_socket):
    while True:
        message = client_socket.recv(1024)
        if not message:
            break
        print(f"Received: {message.decode()}")
        if cv.waitKey(1) & 0xFF == ord('q'):
            client_socket.close()
            server_socket1.close()
            break
    client_socket.close()

# server_socket1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server_socket1.bind(('127.0.0.1',5000))
# server_socket1.listen(1)
# print("กำลังรอการเชื่อมต่อ...")
# client_socket  = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# client_socket.connect(('127.0.0.1',5005))
# print("กำลังรอการเชื่อมต่อ...")
# conn1, addr1 = server_socket1.accept()
# print(f"เชื่อมต่อจาก: {addr1}")

if __name__ == "__main__":
    grab_img()

        