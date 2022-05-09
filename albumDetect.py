#!/usr/bin/python3

# ========================================
# ENPM673 Spring 2022: Perception for Autonomous Robotics
# Project 1: Problem-4 Album Cover Detection 
#
# Authors: John Edward Draganov*, Yoseph Kebede
# ========================================
# Run as 'python3 detection.py'
# Press ESC for exit

import os
import math as m
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2

def maskBW(image):
    """Masking Image"""

    # Mask image to only show AR Tag in grayscale
    image_hsv= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h,w = image_hsv.shape
    # White threshold
    lower_grey_1 = np.array([200])
    upper_grey_1 = np.array([255])
    mask2 = cv2.inRange(image_hsv, lower_grey_1, upper_grey_1)
    pts1 = np.argwhere(mask2)
    print(len(pts1))
    if len(pts1) ==0:
        image_hsv= cv2.equalizeHist(image_hsv)
        #image_hsv= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lower_grey_1 = np.array([200])
        upper_grey_1 = np.array([255])
        mask2 = cv2.inRange(image_hsv, lower_grey_1, upper_grey_1)
        pts2 = np.argwhere(mask2)
        print(len(pts2))
    
    image_hsv = cv2.bitwise_and(image_hsv, image_hsv, mask = mask2)
    image_hsv[np.where(mask2==0)] = 0

    return image_hsv, pts1

def processIm(image):
    """Processes images to isolate faces"""

    imMono, pts= maskBW(image)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image_hsv, pts, imMono

def faceRec(imageHsv, view):
    """Identify Faces in Image"""
    """Reference: 
    https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81"""

    imageHsv = cv2.cvtColor(imageHsv, cv2.COLOR_BGR2GRAY)
    if view =="front":
        vision = 'haarcascade_frontalface_default.xml'
    else:
        vision = 'lbpcascade_profileface.xml'
    faceC = cv2.CascadeClassifier(vision)

    faces = faceC.detectMultiScale(imageHsv, scaleFactor=1.3,
                                    minNeighbors=5, minSize=(30,30),
                                    flags = cv2.CASCADE_SCALE_IMAGE)#cv2.cv.CV_HAAR_SCALE_IMAGE)#
    pts = len(faces)
    # Draw rectangle around faces
    for (x,y,w,h) in faces:
        imageHsv = cv2.rectangle(imageHsv, (x,y), (x+w,y+h), (255,0,0),2)
        #pts.append(x)

    return imageHsv, pts

def featDet(image, secImg, view1="",tre=0,loop=False,H=0):
    """Match Feature between the two image inputs"""

    # Create features - using Flann Based Matcher
    orb = cv2.ORB_create(nfeatures=500)
   
    k2, des2 = orb.detectAndCompute(secImg, None)
    k1, des1 = orb.detectAndCompute(image, None)

    i_p = dict(algorithm=6, table_number=6,
            key_size=12, multi_probe_level=2)

    s_p = {}

    f = cv2.FlannBasedMatcher(i_p, s_p)
    match = f.knnMatch(des1, des2, k=2)

    hPoints = []

    # Use different threshold for ORB feature detection if warp
    # doesn't match image

    # Use 0.93 as threshold for Gorill far pic
    # Use 0.9 for lady red close up
    if tre==0:
        if view1 == "side":
            thresh = 0.89 #0.89, 0.92, 0.93
        else:
            thresh = 0.9   #0.9
    else:
        thresh = tre


    for i,j in match:
        if i.distance < thresh * j.distance:
            hPoints.append(i)
    match = np.asarray(hPoints)

    
    # Store matching points
    mk1 = []
    mk2 = []
    for m in match:
        i1 = m.queryIdx
        i2 = m.trainIdx 

        mk1.append([k1[i1].pt])
        mk2.append([k2[i2].pt])
    if (len(mk1) != 0) or (len(mk2) != 0):
        mk1=np.vstack(mk1)
        mk2 = np.vstack(mk2)

        mk1 = np.float32(mk1)
        mk2 = np.float32(mk2)
    else:
        thresh = 0
        return H, thresh

    F, mask = cv2.findFundamentalMat(mk1, mk2, cv2.FM_LMEDS)
        
    if mask is not None:
        # We select only inlier points
        m1 = mk1[mask.ravel() == 1]
        m2 = mk2[mask.ravel() == 1]

        Hmt1, mask1 = cv2.findHomography(m1,m2,cv2.RANSAC, 5.0)

        if not loop:
            # Method 2: findContours
            tInv = cv2.threshold(secImg,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
            blur = cv2.GaussianBlur(tInv,(1,1),0)
            threshold = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            count= cv2.findContours(threshold,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            mask = np.ones(secImg.shape[:2],dtype="uint8")*255
            noCount = len(count)
            if noCount !=0:
                # Using contours from findContours to detect edges
                for c in count:
                    x,y,w,h = cv2.boundingRect(c)
                    if w*h>700:
                        cv2.rectangle(mask,(x,y),(x+w,y+h),(0,0,255),-1)
                    
                image = cv2.bitwise_and(image,image,mask=mask)
            #Draw matches - just for illustration
            mImg = cv2.drawMatches(image,k1,secImg,k2,match[:20],None)
            # cv2.imshow('Match',mImg)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

    else:
        Hmt1 = H
        thresh = 0

    return  Hmt1, thresh    


def contourEdge(image,method=""):
    """Finding Outer Rectangle Edges of Album Covers"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # HouglLines approach to detect frame edges
    if method== "Hough":
        # Method 1: HoughLinesP
        #oneEdge =  cv2.Canny(image, 30, 200)
        oneEdge =  cv2.Canny(image, 50, 150, None, 3, L2gradient=True)
        lines = cv2.HoughLinesP(oneEdge,1,np.pi/180,threshold = 200,minLineLength=150,maxLineGap=250)
        
        #lines = cv2.HoughLines(oneEdge, 1, np.pi/180,200)
        noLines = len(lines)        
        copy = image.copy()
        if noLines <20:
            for line in lines:
                x1,y1,x2,y2 = line[0]
                cv2.line(copy, (x1,y1),(x2,y2),(255,0,0),3)
            print("Number of Lines detected by HoughLines are: ",noLines)
            """ # Uncomment to hee Hough Lines on image """
            # cv2.imshow('Hough Lines',copy)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
            return image, True

        else:
            return image, False
    else:    
        # Method 2: findContours
        tInv = cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
        blur = cv2.GaussianBlur(tInv,(1,1),0)
        thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        count= cv2.findContours(thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        mask = np.ones(image.shape[:2],dtype="uint8")*255
        noCount = len(count)

        # Using Corner Detection to determine if image corners can be located
        # Modify image data-type to float
        op_Image = np.float32(image)

        bif = cv2.bilateralFilter(op_Image, 5, 75, 75) #75

        # Detect corners of AR Tag with corner Harris method
        cornH = cv2.cornerHarris(bif, 2, 3, 0.09) #0.04  # 2,5,0.06
        cornH = cv2.dilate(cornH,None)
        
        mask2 = np.zeros_like(image)
        inp = 0.3 * cornH.max()
        mask2[cornH > inp] = 25
        ret, cornH = cv2.threshold(cornH,inp,255,0)
        cornH = np.uint8(cornH)

        pts = np.argwhere(mask2)

        noPts = len(pts)
        imCopy = image.copy()

        if noCount !=0 and noPts >10:
           
            # Using contours from findContours to detect edges
            for c in count:
                x,y,w,h = cv2.boundingRect(c)
                if w*h>1000:
                    cv2.rectangle(mask,(x,y),(x+w,y+h),(0,0,255),-1)
            for i in range(len(pts)):
                cv2.circle(imCopy,tuple(reversed(pts[i])),3,(0,0,255),-1)   
            image = cv2.bitwise_and(image,image,mask=mask)
            show = cv2.bitwise_and(imCopy,image,mask=mask)
            print ("Total number of Harris corner Points: ", noPts)
            """ # Uncomment to see image with corner points """
            # cv2.imshow('Masked Contour Detection',show)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            
            return image, True
        else:
            return image, False

def realign(image, secImg, pts, view="",treshold=0,loop=False,H=np.zeros((0,0))):
    """Compare original image with reoriented image"""

    if len(pts) < 100:
        # Histogram Equalization - method
        #secImg= cv2.equalizeHist(secImg)
        #secImg = cv2.GaussianBlur(secImg,(5,5),0)
        #sharpen_kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
        #secImg = cv2.filter2D(secImg,-1,sharpen_kernel)
        
        # Adaptive Equalization - CLAHE method
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        secImg = clahe.apply(secImg)
        
        
   
    h,w = secImg.shape
    #Find homography based on matching points
    if treshold==0:
        H, thresh = featDet(image,secImg, view1=view)
    else:
        H, thresh = featDet(image,secImg,tre=treshold,loop=loop,H=H)
    
    if thresh != 0:
        #Warp the second image to the orientation of the first image
        wrp = cv2.warpPerspective(secImg,H,(w,h),flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR)
        wrp= cv2.cvtColor(wrp, cv2.COLOR_GRAY2BGR)
    else:
        wrp = image
    
    return wrp, thresh, H

def imageSimilarity(image1, image2, treshold):
    """Find similarity of images"""

    orb = cv2.ORB_create(nfeatures=500)
   
    k2, des2 = orb.detectAndCompute(image2, None)
    k1, des1 = orb.detectAndCompute(image1, None)

    # Check if images are similar
    numKeyPts = 0
    if len(k1) <= len(k2):
        numKeyPts = len(k1)
    else:
        numKeyPts = len(k2)

    i_p = dict(algorithm=6, table_number=6,
    key_size=12, multi_probe_level=2)

    s_p = {}

    f = cv2.FlannBasedMatcher(i_p, s_p)
    match = f.knnMatch(des1, des2, k=2)

    hPoints = []
    
    try:      
        for i,j in match:
            if i.distance < treshold * j.distance:
                hPoints.append(i)
                match = np.asarray(hPoints)

        similarity = float(len(match) / numKeyPts) * 100
    except:
        similarity = 0

    return similarity

def findMatchPts(img,fImg,key,thresh, method=""):
    """Detect if matching points were found during feature matching step"""
    condition = False
     # Check that matching points were found
    if thresh != 0:
        # Check to see if rectangular album frame can be detected
        check, flag= contourEdge(img,method)
        
        # Perform similarity check if only edges have been detected
        if flag:
            similarity = imageSimilarity(fImg, check,thresh)
        
            if similarity < key:
                condition = False
            else:
                condition = True
        else:
            condition = False
    
    return condition    

def main():
    """Detect Album Cover Images"""

    # Save the six distinct vertical orientation imges
    for i in range(1,7):
        d = str(i)
        img = cv2.imread("Pics/grp_"+d+"/IMG_0%d.JPG"%i)
        img = cv2.resize(img,(800, 600))
        
        fImg, pts, imMono= processIm(img)

        noFiles = len(next(os.walk("Pics/grp_"+d))[2])
    
        for j in range(2,noFiles+1):
            img1 = cv2.imread("Pics/grp_"+d+"/IMG_%d.jpg"%j)
            img1 = cv2.resize(img1,(800, 600))
            if i ==1 or i==4:
                view = "side"
            else:
                view = "front"
            
            fImg1, pts2, imMono2= processIm(img1)
            
            img2, thresh, H_old = realign(fImg,fImg1,pts2,view)
            key = 35
            condition= findMatchPts(img2,fImg,key,thresh)
            tag = str(i)+str(j)

            # Check that matching points were found
            if not condition:
                # If warped image is gibbrish, loop for different threshold values
                # until correct warp is obtained

                # Creating a list of thresholds to reiterate through
                tre = []
                t = 0.7
                for i in range(1,31):
                    t +=0.01 
                    tre.append(t)
                
                indx = 0
                while (not condition) and (indx < len(tre)):
                    img2, thresh, H_new = realign(fImg,fImg1, pts2,treshold=tre[indx],loop=True,H=H_old)
                    indx += 1
                    key = 20
                    print(thresh)
                    
                    # Check if points were found and similarity met
                    condition = findMatchPts(img2,fImg,key,thresh,method="Contour")
                    
                    # Check if there are HoughLines to verify if image is correctly warped
                    
                    if condition:
                        sharpen_kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
                        img2 = cv2.filter2D(img2,-1,sharpen_kernel)
                        condition = findMatchPts(img2,fImg,key,thresh,method="Hough")
                    H_old = H_new                                                   
            else:
                imgGray = np.hstack((img1,img2))
                
                # Uncomment to see plots
                #cv2.imshow('warp', check)
                #cv2.imshow('Gray-Compare', imgGray)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # cv2.imwrite("picRect%d.jpg" %j,imgGray)
                
                cv2.imwrite("Pics/Adjusted_Pic/adjPic_Easy"+tag+".jpg",imgGray)
            
            # Save adjusted picture in a folder to pass on to face detection step
            cv2.imwrite("Pics/Adjusted_Pic/adjPic_"+tag+".jpg",img2)
            
            print("*******Pictures Saved**********")
    print("*******End of Code**********")
            

if __name__ == '__main__':
    main()
