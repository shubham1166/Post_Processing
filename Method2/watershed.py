'''
This contains a very simple implimentation of watershed algorithm that we have implimented to get our results
'''

# import the necessary packages
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
from cv2 import imread,imshow, imwrite
import numpy as np
import cv2
from tqdm import tqdm

#These are the paths to the segmented images that we have found using deep learning algorithms 
path2original='E:/Cell_data/U-Net/Post-processing/origanal images/'
path2segmented='E:/Cell_data/U-Net/Post-processing/Segmented images/'

def give_watershed_segmentation(t):
    image = cv2.imread(path2segmented+str(t)+'.png',0)
    
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
    
name1=np.load('E:/Cell_data/U-Net/Post-processing/processed/coloured/name.npy')    
for i in tqdm(name1):
    z= give_watershed_segmentation(i)
    imwrite('E:/Cell_data/U-Net/Post-processing/Watershed_processed/type1/'+str(i)+'.png',z )
    
name2=np.load('E:/Cell_data/U-Net/Post-processing/processed/TYPE2/IMAGES/images.npy')    
for i in tqdm(name2):
    z= give_watershed_segmentation(i)
    imwrite('E:/Cell_data/U-Net/Post-processing/Watershed_processed/type2/'+str(i)+'.png',z )    

name3=np.load('E:/Cell_data/U-Net/Post-processing/processed/TYPE3/images/images.npy')    
for i in tqdm(name3):
    z= give_watershed_segmentation(i)
    imwrite('E:/Cell_data/U-Net/Post-processing/Watershed_processed/type3/'+str(i)+'.png',z ) 
    
