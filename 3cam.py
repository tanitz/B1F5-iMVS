import cv2 as cv
import threading
import socket

UDP_IP = 'localhost'  # Replace with your UDP server IP
UDP_PORT1 = 4002  # Replace with your UDP port number
UDP_PORT2 = 4003
def send_image_to_udp(sock, image,UDP_PORT):
    try:
        # Encode the image as a JPEG
        _, img_encoded = cv.imencode('.jpg', image)
        
        # Convert to a binary buffer
        img_buffer = img_encoded.tobytes()
        
        # Send the binary buffer via UDP
        sock.sendto(img_buffer, (UDP_IP, UDP_PORT))
    except Exception as e:
        print(f"Error sending image: {e}")

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
    ROI_Crop3 = [180,400,300,510]
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
        # print(D1)
        if D1 > 15  :
            Dcircle.append(D1)
            cv.ellipse(crop3, (centerX_circle, centerY_circle), (30,30), angle,0, 360, (0,255,0), 2, cv.LINE_AA)
    return crop3 , len(DetectedCircles)

def capture_camera(index, window_name):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    cap = cv.VideoCapture(index) 
    
    if not cap.isOpened():
        print(f"ไม่สามารถเปิดกล้อง {index} ได้")
        return
    while True:
        data = "NG"
        K=1
        ROI_Crop3 = [100,500,200,500]
        ret, frame = cap.read()
        # FIND(frame)
        try:
            
            crop3,circle = FIND(frame)
            cv.imshow(window_name, crop3)
            print(circle)
            if circle > 0:
                data = "OK"
            else:
                data = "NG"

        except:
            
            Dcircle=[]
            esrc = frame.copy()
            crop3 = esrc[ROI_Crop3[0]//K:ROI_Crop3[1]//K, ROI_Crop3[2]//K:ROI_Crop3[3]//K]
            cv.imshow(window_name, crop3)
            
        send_image_to_udp(sock, crop3,UDP_PORT1)
        sock.sendto(data.encode(), (UDP_IP, UDP_PORT2))
        if not ret:
            print(f"ไม่สามารถอ่านเฟรมจากกล้อง {index} ได้")
            break
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyWindow(window_name)

thread1 = threading.Thread(target=capture_camera, args=(2, 'Camera 2'))

thread1.start()

thread1.join()

# ปิดหน้าต่างทั้งหมด
cv.destroyAllWindows()
