import cv2 as cv
import numpy as np
import csv
import os

# CSV file to store parameters
PARAMS_CSV = 'params.csv'

def save_params_to_csv(blur_ksize, blur_sigma, canny_thresh1, canny_thresh2):
    with open(PARAMS_CSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['blur_ksize', 'blur_sigma', 'canny_thresh1', 'canny_thresh2'])
        writer.writerow([blur_ksize, blur_sigma, canny_thresh1, canny_thresh2])

def read_params_from_csv():
    if os.path.exists(PARAMS_CSV):
        with open(PARAMS_CSV, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            row = next(reader)
            try:
                blur_ksize = int(row[0])
                blur_sigma = float(row[1])
                canny_thresh1 = int(row[2])
                canny_thresh2 = int(row[3])
                return [blur_ksize, blur_sigma, canny_thresh1, canny_thresh2]
            except ValueError as e:
                print(f"Error reading parameters from CSV: {e}")
                return [3, 1.0, 100, 200]  # Default values
    return [3, 1.0, 100, 200]  # Default values if file doesn't exist

# def update_image():
#     image = cv.imread('1_1.bmp')
#     if image is None:
#         print("Error loading image")
#         return
    
#     src = image.copy()
#     K = 1
    
#     ROI_Crop1 = [1200, 1400, 700, 1300]
#     crop1 = src[ROI_Crop1[0]//K:ROI_Crop1[1]//K, ROI_Crop1[2]//K:ROI_Crop1[3]//K]
    
#     grayL = cv.cvtColor(crop1, cv.COLOR_BGR2GRAY)

#     blur_ksize = cv.getTrackbarPos('Blur ksize', 'Controls') * 2 + 1
#     blur_sigma = cv.getTrackbarPos('Blur sigma', 'Controls') / 10.0
#     canny_thresh1 = cv.getTrackbarPos('Canny thresh1', 'Controls')
#     canny_thresh2 = cv.getTrackbarPos('Canny thresh2', 'Controls')
    
#     grayL = cv.GaussianBlur(grayL, (blur_ksize, blur_ksize), blur_sigma)
#     edges = cv.Canny(grayL, canny_thresh1, canny_thresh2) 
    
#     rect1 = (120, 35, 50, 130)
#     rect2 = (360, 35, 35, 130)
    
#     cv.rectangle(crop1, (rect1[0], rect1[1]), (rect1[0] + rect1[2], rect1[1] + rect1[3]), (0, 255, 0), 5)
#     cv.rectangle(crop1, (rect2[0], rect2[1]), (rect2[0] + rect2[2], rect2[1] + rect2[3]), (0, 255, 0), 5)
    
#     L_detect1 = edges[rect1[1]:rect1[1] + rect1[3], rect1[0]:rect1[0] + rect1[2]]
#     L_detect2 = edges[rect2[1]:rect2[1] + rect2[3], rect2[0]:rect2[0] + rect2[2]]
    
#     contoursFirst, _ = cv.findContours(L_detect1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#     min_x = float('inf')
#     min_x_point = (0, 0)
#     for contour in contoursFirst:
#         for point in contour:
#             if point[0][0] < min_x:
#                 min_x = point[0][0]
#                 min_x_point = (point[0][0], point[0][1])

#     contoursSecond, _ = cv.findContours(L_detect2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#     max_x = float('-inf')
#     max_x_point = (0, 0)
#     for contour in contoursSecond:
#         for point in contour:
#             if point[0][0] > max_x:
#                 max_x = point[0][0]
#                 max_x_point = (point[0][0], point[0][1])
    
#     offset_contoursFirst = [contour + np.array([rect1[0], rect1[1]]) for contour in contoursFirst]
#     offset_contoursSecond = [contour + np.array([rect2[0], rect2[1]]) for contour in contoursSecond]

#     cv.drawContours(crop1, offset_contoursFirst, -1, (0, 255, 0), 1) 
#     cv.drawContours(crop1, offset_contoursSecond, -1, (255, 0, 0), 1)  
#     cv.circle(crop1, (min_x_point[0] + rect1[0], min_x_point[1] + rect1[1]), 5, (0, 0, 255), -1)
#     cv.circle(crop1, (max_x_point[0] + rect2[0], max_x_point[1] + rect2[1]), 5, (0, 0, 255), -1)

#     cv.imshow("Region 1", L_detect1)
#     cv.imshow("Region 2", L_detect2)
#     cv.imshow("edges", edges)
#     cv.imshow("image", crop1)

def update_image():
    image = cv.imread('/home/code/hikrobot/image/Image_test2.bmp')
    if image is None:
        print("Error loading image")
        return
    
    src = image.copy()
    K = 1
    
    ROI_Crop1 = [1200, 1400, 700, 1300]
    crop1 = src[ROI_Crop1[0]//K:ROI_Crop1[1]//K, ROI_Crop1[2]//K:ROI_Crop1[3]//K]
    
    grayL = cv.cvtColor(crop1, cv.COLOR_BGR2GRAY)

    blur_ksize = cv.getTrackbarPos('Blur ksize', 'Controls') * 2 + 1
    blur_sigma = cv.getTrackbarPos('Blur sigma', 'Controls') / 10.0
    canny_thresh1 = cv.getTrackbarPos('Canny thresh1', 'Controls')
    canny_thresh2 = cv.getTrackbarPos('Canny thresh2', 'Controls')
    
    grayL = cv.GaussianBlur(grayL, (blur_ksize, blur_ksize), blur_sigma)
    edges = cv.Canny(grayL, canny_thresh1, canny_thresh2) 
    
    rect1 = (120, 35, 50, 150)
    rect2 = (360, 35, 35, 150)
    
    cv.rectangle(crop1, (rect1[0], rect1[1]), (rect1[0] + rect1[2], rect1[1] + rect1[3]), (0, 255, 0), 5)
    cv.rectangle(crop1, (rect2[0], rect2[1]), (rect2[0] + rect2[2], rect2[1] + rect2[3]), (0, 255, 0), 5)
    
    L_detect1 = edges[rect1[1]:rect1[1] + rect1[3], rect1[0]:rect1[0] + rect1[2]]
    L_detect2 = edges[rect2[1]:rect2[1] + rect2[3], rect2[0]:rect2[0] + rect2[2]]
    
    # Find contours in L_detect1
    contoursFirst, _ = cv.findContours(L_detect1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    min_x = float('inf')
    min_x_point = (0, 0)
    for contour in contoursFirst:
        # No size or area filtering
        # Draw all contours
        x, y, w, h = cv.boundingRect(contour)
        area = cv.contourArea(contour)
        # print(f"Contour area: {area}")
        if len(contour) > 40:  # Check if contour is not empty
            for point in contour:
                if point[0][0] < min_x:
                    min_x = point[0][0]
                    min_x_point = (point[0][0], point[0][1])

    # Find contours in L_detect2
    contoursSecond, _ = cv.findContours(L_detect2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    max_x = float('-inf')
    max_x_point = (0, 0)
    for contour in contoursSecond:
        # No size or area filtering
        # Draw all contours
        x, y, w, h = cv.boundingRect(contour)
        area = cv.contourArea(contour)
        # print(f"Contour area: {area}")
        if len(contour) > 40:  # Check if contour is not empty
            for point in contour:
                if point[0][0] > max_x:
                    max_x = point[0][0]
                    max_x_point = (point[0][0], point[0][1])
    
    offset_contoursFirst = [contour + np.array([rect1[0], rect1[1]]) for contour in contoursFirst]
    offset_contoursSecond = [contour + np.array([rect2[0], rect2[1]]) for contour in contoursSecond]

    cv.drawContours(crop1, offset_contoursFirst, -1, (0, 255, 0), 1) 
    cv.drawContours(crop1, offset_contoursSecond, -1, (255, 0, 0), 1)  
    cv.circle(crop1, (min_x_point[0] + rect1[0], min_x_point[1] + rect1[1]), 2, (0, 255, 0), -1)
    cv.circle(crop1, (max_x_point[0] + rect2[0], max_x_point[1] + rect2[1]), 2, (0, 255, 0), -1)
    print("L_Point=",(min_x_point[0] + rect1[0], min_x_point[1] + rect1[1]))
    print("R_Point=",(max_x_point[0] + rect2[0], max_x_point[1] + rect2[1]))
    cv.imshow("Region 1", L_detect1)
    cv.imshow("Region 2", L_detect2)
    cv.imshow("edges", edges)
    cv.imshow("image", crop1)


initial_params = read_params_from_csv()

cv.namedWindow('Controls', cv.WINDOW_NORMAL)

cv.createTrackbar('Blur ksize', 'Controls', initial_params[0] // 2, 10, lambda x: update_image())
cv.createTrackbar('Blur sigma', 'Controls', int(initial_params[1] * 10), 50, lambda x: update_image())
cv.createTrackbar('Canny thresh1', 'Controls', initial_params[2], 255, lambda x: update_image())
cv.createTrackbar('Canny thresh2', 'Controls', initial_params[3], 255, lambda x: update_image())

update_image()

while True:
    key = cv.waitKey(1) & 0xFF
    if key == 27:  # ESC key to exit
        blur_ksize = cv.getTrackbarPos('Blur ksize', 'Controls') * 2 + 1
        blur_sigma = cv.getTrackbarPos('Blur sigma', 'Controls') / 10.0
        canny_thresh1 = cv.getTrackbarPos('Canny thresh1', 'Controls')
        canny_thresh2 = cv.getTrackbarPos('Canny thresh2', 'Controls')
        
        save_params_to_csv(blur_ksize, blur_sigma, canny_thresh1, canny_thresh2)
        break

cv.destroyAllWindows()
