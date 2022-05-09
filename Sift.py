# import cv2 
# import matplotlib.pyplot as plt
# #%matplotlib inline
# import glob
# # read images
# image_list = []
# sift_list = []
# keyp_list = []
# import streamlit as st


# def sift(opt):

#     for filename in glob.glob('*.jpg'):
#         st.button('Wow')

#         img1 = cv2.imread(filename)  
#         image_list.append(img1)
#         img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

#         sift = cv2.xfeatures2d.SIFT_create()

#         keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
#         sift_list.append(descriptors_1)
#         keyp_list.append(keypoints_1)

#     counter = 0
#     for filename1 in glob.glob('./images/*.jpeg'):

#         #img2 = cv2.imread('test.jpeg')  
#         img2 = opt
#         filename1 = filename1.lstrip('../yolov5/images/')

#         filename1 = str(filename1).rstrip('.jpeg')
#         filename1 = str(filename1).rstrip('.jpg')
#         filename1 = filename1[1:]

#         img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#         sift = cv2.xfeatures2d.SIFT_create()

#         keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)
#         #feature matching
#         bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

#         matches = bf.match(sift_list[counter],descriptors_2)
#         matches = sorted(matches, key = lambda x:x.distance)
#         if len(matches) > 700:
#             print(len(matches))
#             img3 = cv2.drawMatches(image_list[counter], keyp_list[counter], img2, keypoints_2, matches[:50], img2, flags=2)
#             plt.imshow(img3),plt.show()

#             print('Album name is:',filename1)
#             st.button('Yes')
#         else: 
#              #st.button(f'{get_detection_folder()}')
#              st.button('None')

#         counter += 1

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from tqdm import tqdm
from time import sleep
import time
import pandas as pd
import numpy as np
import streamlit.components.v1 as components

import streamlit as st
import os
import cv2 
import matplotlib.pyplot as plt
#%matplotlib inline
# read images
image_list = []
sift_list = []
keyp_list = []
filename_list = []
cid = '73f3ce006c4d406197e700230991abea'

secret = '7272246311f149818dab149eab062074'
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager
=
client_credentials_manager)



def file_selector(opt,folder_path='./images/'):
    filenames = os.listdir(folder_path)
    for filename in filenames:


        img1 = cv2.imread('./images/'+filename)  
        image_list.append(img1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()

        keypoints_1,descriptors_1= sift.detectAndCompute(img1,None)
        sift_list.append(descriptors_1)
        keyp_list.append(keypoints_1)
        filename_list.append(str(filename))

    found = 0 
    counter = 0
    for filename in filenames:

        img2 = cv2.imread(opt)  
        #img2 = opt
        
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()

        keypoints_2,descriptors_2 = sift.detectAndCompute(img2,None)
        #feature matching
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

        matches = bf.match(sift_list[counter],descriptors_2)
        matches = sorted(matches, key = lambda x:x.distance)
        vari = len(filenames)
        if len(matches) > 590 and found == 0:
            print(len(matches))
            img3 = cv2.drawMatches(image_list[counter], keyp_list[counter], img2, keypoints_2, matches[:50], img2, flags=2)
            #plt.imshow(img3),plt.show()
            st.image(img3)
            filename2 = filename_list[counter]
            filename3 = str(filename2).rstrip('.jpeg')
            filename3 = str(filename2).rstrip('.jpg')
            print('Album name is:',filename_list[counter])
            found = 1
            st.header(f'Album name is: {filename3}')
            break
            
        else: 
            
            print('aaa')
            #found = 0
            
#              st.button('None')

        counter += 1
        
        
        if found == 0 and vari == counter:
            st.header('Oops, we do not have that album')
        #st.header(f'{found}')
#         if found == 1:
#             #track_results = sp.search(q='album:'+ str(filename3), type='album', limit=1)
# #             result = sp.search(filename3,type ="album")
#             result = sp.search(filename3,type ="album",limit = 1)
#             #print(result)

#             #st.header(f'Album name is: {track_results}')
#             album_uris = result['albums']['items'][0]['uri']
#             print(album_uris)
#             album_uris = album_uris.lstrip('spotify:album:')
#             album_uri_link = "https://open.spotify.com/album/" + album_uris
#     #         if type(url) == str:
#             audio2=album_uri_link 
#             components.iframe(album_uri_link , width=600, height=200 )


        
#     selected_filename = st.selectbox('Select a file', filenames)
#     return os.path.join(folder_path, selected_filename)

# filename = file_selector()
# st.write('You selected `%s`' % filename)
