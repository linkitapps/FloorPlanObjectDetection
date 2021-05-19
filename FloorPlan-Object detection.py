#!/usr/bin/env python
# coding: utf-8

# # Q2. Given a floor plan, detect different objects present the floor plan. The object can be door, windows, etc. 

# # ANS: There are several methods to object detection:
#  
# Labeling the objects using  contour/ bounding box: 
# 
# Object Detection using Template Matching which can be done without machine learning, using template image.
# 
# Object detection using deep learning: 
#      There are models available for doing object detection recognition in an image. For example, the most prominent ones are RCNN, fast RCNN, faster RCNN, mask RCNN, Yolo and SSD. However, most of the them are developed for natural images/Real life images. 
#      
#     In this case, we are working on document images ex floor plans, where we need to detect objects like Windows, flowerpots, dining table, Bed, Computer, sofa, sink, etc.
#     
#     There are not many model available for the  documents type images.
#     
#     In order to apply deep learning, we need training image labelled with classes (in our case, icons used by architecture software in a floorplan).
#     
#     I tried to use existing  YOLO model ,and loaded  pretrained model and weight for document type, with darkflow implementation and  however facing import issue with Darflow (https://github.com/thtrieu/darkflow).
#     
#    
# 

# ## In the interest of time, have labeled the objects using contour only. 

# In[4]:


import cv2
import numpy as np


# In[5]:


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
#img_gray=255-img_gray   #inverted to highlight solid lines in white
gray = cv2.threshold(img_gray.copy(), 230,255, cv2.THRESH_BINARY)[1] #threshold applied to cleanout noise
contours,hierarchy = cv2.findContours(gray,cv2.RETR_LIST ,cv2.CHAIN_APPROX_SIMPLE )

#Calculat the boundary box of individual contour and store it in contourBBS
contoursBBS = []
for contour in contours:
    [x, y, w, h] = cv2.boundingRect(contour)
    contoursBBS.append([x, y, w, h, w*h])
contoursBBS = np.array(contoursBBS)

#Filter the boundry box based on width, height and area.
#We can tune the box sized accordingly
contourRects = []
for c in contoursBBS:
    [x, y, w, h, a] = c
    if w < 1 or h <1 or a < 10 or a > 60000:    # 
        continue
    contourRects.append(c)

    #Get the center point of the bounding box
for i, rect in enumerate(contourRects):
    [x, y, w, h, a] = rect
    topCenter, bottomCenter = (int((2*x + w)/2), int(y)), (int((2*x + w)/2), int(y+h))


#draw the bounding box on the detected object
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.circle(img, topCenter, 2, (0, 0, 255), 2)
    cv2.circle(img, bottomCenter, 2, (0, 0, 255), 2)


cv2.imshow('img',img)
cv2.waitKey()


# ## Connected cmponents to get Number of objects

# In[29]:


#Using the connected component stats method to highlight the Walls/Number of object identified. Connected component stats returns, 
#labels with stats coordinates (x,y,w,h,a) and centroid info.

#Color image is loaded so that color outlines can displayed
img_color = cv2.imread('floorplan.jpg')   #canvas.jpg 
img_color=cv2.resize(img_color,(900,500))

#Load the image, resize, change to gray and apply connected components
img = cv2.imread('floorplan.jpg') #This image is of size (351,555,3)
img=cv2.resize(img,(900,500))
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #Converted to gray scale
#img_gray=255-img_gray   #inverted to highlight solid lines in white
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

