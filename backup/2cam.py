import cv2 as cv
import threading

def FIND_POINT2(CropImage):
    ed = cv.ximgproc.createEdgeDrawing()
    EDParams = cv.ximgproc_EdgeDrawing_Params()
    EDParams.MinPathLength = 100    
    EDParams.PFmode = False         
    EDParams.MinLineLength = 5    
    EDParams.NFAValidation = True 
    ed.setParams(EDParams)
    gray = cv.cvtColor(CropImage, cv.COLOR_BGR2GRAY)
    
    gray = cv.bilateralFilter(gray, 5, 75, 75)
    gray = cv.medianBlur(gray,5)
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
        # FIND(frame)
        try:
            crop3 = FIND(frame)
            cv.imshow(window_name, crop3)
        except:
           
            Dcircle=[]
            esrc = frame.copy()
            crop3 = esrc[ROI_Crop3[0]//K:ROI_Crop3[1]//K, ROI_Crop3[2]//K:ROI_Crop3[3]//K]
            cv.imshow(window_name, crop3)
        if not ret:
            print(f"ไม่สามารถอ่านเฟรมจากกล้อง {index} ได้")
            break
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyWindow(window_name)

thread1 = threading.Thread(target=capture_camera, args=(0, 'Camera 1'))

thread1.start()

thread1.join()

# ปิดหน้าต่างทั้งหมด
cv.destroyAllWindows()
