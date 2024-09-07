import cv2 as cv
import threading
import socket
import json
import struct

server_socket1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket1.bind(('127.0.0.1', 5001))
server_socket1.listen(1)
print("กำลังรอการเชื่อมต่อ...")
conn1, addr1 = server_socket1.accept()
print(f"เชื่อมต่อจาก: {addr1}")

def FIND_POINT2(CropImage):
    ed = cv.ximgproc.createEdgeDrawing()
    EDParams = cv.ximgproc_EdgeDrawing_Params()
    EDParams.MinPathLength = 10    
    EDParams.PFmode = False         
    EDParams.MinLineLength = 5    
    EDParams.NFAValidation = True 
    ed.setParams(EDParams)
    gray = cv.cvtColor(CropImage, cv.COLOR_BGR2GRAY)
    gray = cv.bilateralFilter(gray, 1, 55, 55)
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
            if abs(major_axis-minor_axis) < 20 :
                diameter = (major_axis + minor_axis) / 2.0
                DetectedCircles.append([center,axes,angle])
    return DetectedCircles

def FIND(Image_output):
    K=1
    ROI_Crop3 = [100,500,200,500]
    Dcircle=[]
    esrc = Image_output.copy()
    crop3 = esrc[ROI_Crop3[0]//K:ROI_Crop3[1]//K, ROI_Crop3[2]//K:ROI_Crop3[3]//K]
    DetectedCircles = FIND_POINT2(crop3)
    if len(DetectedCircles) > 0:
        max_index = min(range(len(DetectedCircles)), key=lambda i: min(DetectedCircles[i][1]))
        center,axes,angle = DetectedCircles[max_index]
        centerX_circle, centerY_circle = center
        R1,R2 = axes
        D1 = R1+R2
        print(D1)
        if D1 > 15  :
            Dcircle.append(D1)
            cv.ellipse(crop3, (centerX_circle, centerY_circle), (30,30), angle,0, 360, (0,255,0), 2, cv.LINE_AA)
    return crop3

def capture_camera(index, window_name):
    cap = cv.VideoCapture(index) 
    if not cap.isOpened():
        print(f"ไม่สามารถเปิดกล้อง {index} ได้")
        return
    while True:
        K=1
        ROI_Crop3 = [100,500,200,500]
        ret, frame = cap.read()
        try:
            crop3 = FIND(frame)
            cv.imshow(window_name, crop3)
            
            # Convert the image to bytes
            _, buffer = cv.imencode('.jpg', crop3)
            image_bytes = buffer.tobytes()

            # Create a dictionary to
