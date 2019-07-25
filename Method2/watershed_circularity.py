'''
Created on Jun 24, 2019
We'll implement the circularity by checking the circularity of each connected component and then applying circularity depending upon the shape, if it is circular 
@author: T01130
'''

# import the necessary packages
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
from cv2 import imread,imshow, imwrite
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt


#########################################################################
def give_watershed_segmentation(t):
    image = t
    
    D = ndimage.distance_transform_edt(image)#Exact euclidean distance transform. DISTANCE MAP
    localMax = peak_local_max(D, indices=False, min_distance=20,labels=image)#Find peaks in an image as coordinate list or boolean mask.
      
    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    
    labels = watershed(-D, markers, mask=image)
#     print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
    
    end_image = np.zeros(image.shape, dtype="uint8")
    
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue
      
        #We want to seperate the cells, so we can do it this way
        mask = np.zeros(image.shape, dtype="uint8")
        mask[labels == label] = 255
        kernel = np.ones((3,3),np.uint8)
        eroded_image = cv2.erode(mask,kernel,iterations=1)
        end_image[eroded_image == 255] = 255
    
    return end_image

##
def get_area(contours):
    area=cv2.contourArea(contours[0])
    return area
def get_perimeter(contours):
    perimeter=cv2.arcLength(contours[0],True)
    return perimeter
def get_circularity(u):
    contours = cv2.findContours(u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    area=get_area(contours)
    perimeter=get_perimeter(contours)
    #C = 4_pi_A/P^2'
    C=(4*np.pi*area)/(perimeter*perimeter+1e-8)
    return C
##
#########################################
#########################################
#path2load='E:/Cell_data/U-Net/Post-processing/Segmented images/'
path2load='E:/Cell_data/U-Net/Post-processing/processed/TYPE2/result/'
# k=2

def give_different_connected_components(k):
    y=imread(path2load+str(k)+'.png',0)
    #Extracting connected components
    ret, labels = cv2.connectedComponents(y)
    z=[]
    for label in np.unique(labels):
            # if the label is zero, we are examining the 'background'
            # so simply ignore it
            if label == 0:
                continue
          
            #We want to seperate the cells, so we can do it this way
            mask = np.zeros(y.shape, dtype="uint8")
            mask[labels == label] = 255
            z.append(mask)
    print('No. of different contours are:',len(z))
    return z

def watershed_circularity(k):
    y=imread(path2load+str(k)+'.png',0)
    z=give_different_connected_components(k)
    new_image= np.zeros(y.shape, dtype="uint8")
    for i in range(len(z)):
        if get_circularity(z[i])>=0.6:
            new_image[z[i] == 255] = 255
        else:
            watershed_image=give_watershed_segmentation(z[i])
            new_image[watershed_image == 255] = 255
    return new_image

# imshow('new_image',watershed_circularity(5))
# cv2.waitKey(0)

# path2save='E:/Cell_data/U-Net/Post-processing/Watershed_circularity/'
# 
# for i in tqdm(range(670)):
#     watershed_image=watershed_circularity(i)
#     imwrite(path2save+str(i)+'.png',watershed_image)



path2save='E:/Cell_data/U-Net/Post-processing/processed/TYPE2/watershed_circularity/'
a=np.load('E:/Cell_data/U-Net/Post-processing/processed/TYPE2/IMAGES/images.npy')
for i in tqdm(a):
    watershed_image=watershed_circularity(i)
    imwrite(path2save+str(i)+'.png',watershed_image)


