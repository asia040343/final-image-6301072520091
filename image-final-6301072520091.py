import numpy 
import cv2 
import os

video1 = cv2.VideoCapture('videos\left_output-1.avi') 
img1 = cv2.imread('images\Template-1.png') 
img_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#############################################################################################################

while video1.isOpened() : 
    V,  frame = video1.read()  

    if V :
        
        img_frame = frame
        img_sharp = numpy.array([[0,0,0], [0, 1.5,0],[0,0,0]])  
        img_filter = cv2.filter2D(img_frame, -1, img_sharp) 
########################################################################################################################
       
        img_frame = cv2.cvtColor(img_filter, cv2.COLOR_BGR2GRAY) 
    
        sift = cv2.SIFT_create() #ใช้ function sift ค้นหาจุดที่คล้ายกันมากที่สุด 
        bf = cv2.BFMatcher() # ทำ Brute-Force แมทเชอร์ 

        template_kpts, template_desc = sift.detectAndCompute(img1, None)
        query_kpts, query_desc = sift.detectAndCompute(img_frame, None)
        matches = bf.knnMatch(template_desc, query_desc, k=2)

        good_matches = list() 
        good_matches_list = list()
        for A, B in matches :
            if A.distance < 0.7*B.distance :
                good_matches.append(A)
                good_matches_list.append([A])
        

           ###############################################################################################################


        if len(good_matches) > 14 : 
            matches1 = numpy.float32([ template_kpts[A.queryIdx].pt for A in good_matches ]).reshape(-1,1,2)
            matches2 = numpy.float32([ query_kpts[A.trainIdx].pt for A in good_matches ]).reshape(-1,1,2)

            R, Error_R = cv2.findHomography(matches1, matches2, cv2.RANSAC, 1.2) # RANSAC 
            
            h, w = img1.shape[:2]
            frame_box1 = numpy.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1,1,2) 
            transform = cv2.perspectiveTransform(frame_box1, R)

            detec_img1 = cv2.polylines( frame, [numpy.int32(transform)], True, (0,255,0), 2, cv2.LINE_AA)
            draw_img1 = cv2.drawMatchesKnn(img1, template_kpts, detec_img1, query_kpts, good_matches_list, None, flags=2, matchesMask=Error_R)

          
            cv2.imshow('BOOK_ASIA',detec_img1)
           
        if cv2.waitKey(int(1000/20)) & 0xFF == ord('a') : 
            break
   

