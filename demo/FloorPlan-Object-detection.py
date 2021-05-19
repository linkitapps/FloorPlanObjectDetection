#!/usr/bin/env python

import cv2
import numpy as np
    
img_filepath = './images/apartment.jpg'
img = cv2.imread(img_filepath)
img_height, img_width, img_channels = img.shape
img_color = cv2.imread(img_filepath)
img_color = cv2.resize(img_color,(img_width,img_height))
img = cv2.resize(img,(img_width,img_height))
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.threshold(img_gray.copy(), 235,255, cv2.THRESH_BINARY)[1]

# Find the connected components on the floor
nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(gray, connectivity=8)# takes in gray image, connectivity to lookout for corners and output image data type.

#Threshold to filterout noises
gap_threshold = 20

#Create a empty list to collect identified objects
Walls_object=[]

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

#show image
cv2.imshow('img',img_color)   
cv2.waitKey()
