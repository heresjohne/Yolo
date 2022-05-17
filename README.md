#------------ Running Scraping Code ----------------------#
Run scraper.py

#------------ Running BoundBox Code ----------------------#
Put training tags/labels in a folder from parent (.) as '.yolov5/data/FinalProject/labels/train'
Put training images in folder from parent as '.yolov5/data/FinalProject/CrowdHuman/train/'
Rung boundbox.py

#------------ Running Streamlit Code ----------------------#
Take photo of album with your phone from one of 6 available test groups (to add your own set skip these instructions and move to next section).
Upload to website and hit run. Streamlit has current bug that prevents it working on IOS

#------------ Uplading Custom Streamlit Code ----------------------#
Make new folder titled "grp_*n*" under yolov5/Pics where n is the next number in the sequence
Insert a good, recitifed image of the album you want to detect naming it IMG_0*n*.png (see other images for reference)
In yolov5/main on line 133/134 add a next line as:

            elif source_index == n:
                img2 = albumDetect.main((n+1),opt.source)
                
