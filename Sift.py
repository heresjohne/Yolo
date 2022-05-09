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
found = 0

cid = '16f1343fb87b4e5b9eea7dcff1123613'
secret = '330ce74bb10e426b841045d8d9ad491b'
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
        if len(matches) > 600:
            print(len(matches))
            img3 = cv2.drawMatches(image_list[counter], keyp_list[counter], img2, keypoints_2, matches[:50], img2, flags=2)
            #plt.imshow(img3),plt.show()
            filename2 = filename_list[counter]
            filename3 = str(filename2).rstrip('.jpeg')
            filename3 = str(filename2).rstrip('.jpg')
            print('Album name is:',filename_list[counter])
            found = 1
            st.header(f'Album name is: {filename3}')
            break
        else: 
            
            print('aaa')
            found = 0
            
#              st.button('None')

        counter += 1
        
        
        if found == 0 and vari == counter:
            st.header('Oops, we do not have that album')
        if found == 1 and vari == counter:
            track_results = sp.search(q='album:'+ filename3, type='album', limit=1)
            result = sp.search(i,type ="album")
            st.header(f'Album name is: {result}')
            album_uris = result['album']['items'][0]['uri']
            df_album_uris = pd.DataFrame(album_uris,columns=['album_uri'])
            df_album_uris.head()
            new = df._album_uris["album_uri"].str.split(":",n = 2,expand = True)
            df_album_uris["album_uri_new"]= new[2]
            df_album_uris.head()

    #         if type(url) == str:
            try:
                audio2=df_album_uris 
                components.iframe(album_uri_link , width=600, height=200 )
            except:
                print('ww')

        
#     selected_filename = st.selectbox('Select a file', filenames)
#     return os.path.join(folder_path, selected_filename)

# filename = file_selector()
# st.write('You selected `%s`' % filename)
