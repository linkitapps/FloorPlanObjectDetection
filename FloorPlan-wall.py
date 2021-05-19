#!/usr/bin/env python
# coding: utf-8

# # Q1. Wall Measurement
# 
# Given a floor plan: export the length of all the walls in a CSV.  The above given figure is just an example figure, you can 
# choose any floor plan.
# 

# ## In absence of details, assumptions were made. In the given floorplan design, solid lines in black are seen. Length of these solid lines are calculated.
# The solution is developed on Python, using mainly Opencv, Numpy and csv Library.
# To make the measurement, contour methods are used and imported to csv file.
# Using OPENCV ‘ arclength’ function, perimeter is calculated. 
# Length is approximated to be half of the perimeter, though only perimeter is exported to csv, however in the image. Walls are Labelled with Length and areas , as well.
# 

# In[1]:


#Import the necessary libraries
import cv2
import numpy as np
import csv


# # Contour Method:  the walls

# In[2]:


#Wall measurement 
#to get contour for SOLID walls, uncomment ->'img_gray=255-img_gray', to contur the rets of the object, 
#comment-->  img_gray=255-img_gray.
#for SOLID Walls, min area =50~100 and max: 60,000.
#for Rest of the object, min area = 1 and maximum area =7500 (excluding rooms open space detection, which have highest area)
#****Each individual contour is a Numpy array of (x,y) coordinates of boundary points of the object.
#Draw Contour: first argument is source image, second argument is the contours which should be passed as a Python list, 
#third argument is index of contours (useful when drawing individual contour. To draw all contours, pass -1) and 
#remaining arguments are color, thickness etc.
img = cv2.imread('floorplan.jpg') #This image is of size (351,555,3)
img=cv2.resize(img,(900,500))
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #Converted to gray scale
img_gray=255-img_gray   #inverted to highlight solid lines in white
gray = cv2.threshold(img_gray.copy(), 200,255, cv2.THRESH_BINARY_INV)[1] #threshold applied to cleanout noise
contours,hierarchy = cv2.findContours(gray,cv2.RETR_LIST ,cv2.CHAIN_APPROX_SIMPLE ) # retrieval method: cv2.RETR_TREE/LIST/EXTERNAL,  appprox method: CHAIN_APPROX_SIMPLE removes reduntant points (givesonly cornor poits data)

bucket_walls=[]  #
contour_sizes = []
#Select the contour based on area size
i=0
font = cv2.FONT_HERSHEY_SIMPLEX 
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area>100 and area<60000:
        L=cv2.arcLength(cnt, True)
        contour_sizes.append(L)
        text1 = "Wall_" + str(i)
        text2= "Length_" +str(round(L/2)) #arcLength calculates the perimeter of a contour. Length is approx to be half.
        text3 = "Area_" + str(round(area))
        bucket_walls.append(cnt)
        cv2.drawContours(img,[cnt],0,(255,0,0),2) # selecting to draw contour by contour
        x = cnt[0][0][0]
        y=cnt[0][0][1]
        cv2.putText(img, text1, (x+15, y+30),font, 0.5, (255, 0, 0),2)
        cv2.putText(img, text2, (x+15, y+40),font, 0.4, (0, 40,255))
        cv2.putText(img, text3, (x+15, y+60),font, 0.4, (0, 40,255))   
        x1,y1,w1,h1 = cv2.boundingRect(cnt)
        cv2.rectangle(img,(x1,y1),(x1+w1,y1+h1),(0,255,0),2)
        i+=1
#Display the Contour
cv2.imshow('img',img)
cv2.waitKey()

#write the length to 
a = np.asarray(contour_sizes)
a.tofile("wall.csv",sep=',',format='%10.5f')


# In[3]:


print("Total contours detected:", len(contours))
print("There are %s part of Walls in this floor Plan:" %(len(bucket_walls)))


# ## Connected Component: Get the number of Walls

# In[4]:


#Using the connected component stats method to highlight the Walls/Number of object identified. Connected component stats returns, 
#labels with stats coordinates (x,y,w,h,a) and centroid info.

#Color image is loaded so that color outlines can displayed
img_color = cv2.imread('floorplan.jpg')   #canvas.jpg 
img_color=cv2.resize(img_color,(900,500))

#Load the image, resize, change to gray and apply connected components
img = cv2.imread('floorplan.jpg') #This image is of size (351,555,3)
img=cv2.resize(img,(900,500))
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #Converted to gray scale
img_gray=255-img_gray   #inverted to highlight solid lines in white
gray = cv2.threshold(img_gray.copy(), 235,255, cv2.THRESH_BINARY)[1]   #245,255,
# Find the connected components on the floor
nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(gray, connectivity=8)# takes in gray image, connectivity to lookout for corners and output image data type.

#Threshold to filterout noises
gap_threshold = 20

#Create a empty list to collect identified objects
Walls_object=[]

font = cv2.FONT_HERSHEY_SIMPLEX 

unique = np.unique(labels) #labels are in multitude, so we get only unique labels

#Loop through the labels to highlight identified areas and gets stats(coordinates) f
for i in range(0, len(unique)):
    component = labels == unique[i]  # enforce to select only unique label
    stat = stats[i]    # get the stats and centroid for that label
    centroid= centroids[i]
    if img_color[component].sum() == 0 or np.count_nonzero(component) < gap_threshold:   #filter out the undesirable noises
        color = 0
    else:
        Walls_object.append([i, component,stat, centroid])  # collect the labels and stats 
        color = np.random.randint(0, 255, size=3)
    img_color[component] = color    #hightlight label/components on image 
    text="Label_" +str(i)
    x,y = centroid
    cv2.putText(img_color, text, (x.astype(int), y.astype(int)),font, 0.4, (250, 0,0)) #print Labels

#show image
cv2.imshow('img',img_color)   
cv2.waitKey()
print("Stats: ",len(stats))
print("Centroid: ",len(centroids))
print("label: ",len(labels))
print("Unique Labels ", len(unique))


# In[ ]:




