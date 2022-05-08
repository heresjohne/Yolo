import cv2 
import matplotlib.pyplot as plt
#%matplotlib inline
import glob
# read images
image_list = []
sift_list = []
keyp_list = []
for filename in glob.glob('../yolov5/images/*.jpeg'):
    
    img1 = cv2.imread(filename)  
    image_list.append(img1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
    sift_list.append(descriptors_1)
    keyp_list.append(keypoints_1)

counter = 0
for filename1 in glob.glob('../yolov5/images/*.jpeg'):
    img2 = cv2.imread('test.jpeg')  

    filename1 = filename1.lstrip('../yolov5/images/')

    filename1 = str(filename1).rstrip('.jpeg')
    filename1 = str(filename1).rstrip('.jpg')
    filename1 = filename1[1:]

    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()

    keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)
#feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(sift_list[counter],descriptors_2)
    matches = sorted(matches, key = lambda x:x.distance)
    if len(matches) > 700:
        print(len(matches))
        img3 = cv2.drawMatches(image_list[counter], keyp_list[counter], img2, keypoints_2, matches[:50], img2, flags=2)
        plt.imshow(img3),plt.show()

        print('Album name is:',filename1)
    counter += 1


