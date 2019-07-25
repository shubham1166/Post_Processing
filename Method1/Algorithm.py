'''
Created on Jun 24, 2019
This is the algorithm to post process the images
@author: T01130
'''

import cv2
import numpy as np
from final_post_processing.functions import give_output_unet,type_3_process,get_average_area,type_2_process,type_1_process,watershed_circularity
from tqdm import tqdm





##################################################################
path2original='E:/Cell_data/U-Net/Post-processing/origanal images/'
#Use only original images

def algorithm(t):
    original_image=cv2.imread(path2original+str(t)+'.png',1)
    segmented_image=give_output_unet(original_image)
    ########################################################
    '''
    For type-1 coloured images
    We'll check the mean of a in l*a*b images
    

    For type-2  and type-3 images(Black images with big and small cells)
    We'll check the mean of l in l*a*b images and then check the avegare areas of the contours
    
        For type-4 images ,that are images with white backgrounds, the mean of l  
    >=200 in l*a*b and mean of a and b is around 128
    '''
    lab_original_image= cv2.cvtColor(original_image,cv2.COLOR_BGR2LAB)
    L=lab_original_image[:,:,0]
    A=lab_original_image[:,:,1]
    B=lab_original_image[:,:,2]
    if np.mean(A)>130:
        print('This is a type 1 image')
        new_image=type_1_process(original_image,segmented_image,L)
        new_segmentation=watershed_circularity(new_image)
        return new_segmentation

    elif np.mean(L)<50:
        #These are type 2 and type 3 images
        if get_average_area(segmented_image)>=2000:
            print('This is a type 2 image')
            new_image=type_2_process(original_image,segmented_image)
#             return new_image
            new_segmentation=watershed_circularity(new_image)
            return new_segmentation
        else:
            print('This is a type 3 image')
            new_image=type_3_process(original_image,segmented_image)
            new_segmentation=watershed_circularity(new_image)
        return new_segmentation
    elif np.mean(L)>=200 and np.mean(A)<=129:
        print('This is a type 4 image')
        new_segmentation=watershed_circularity(segmented_image)
        return new_segmentation
    else:
        new_image=type_1_process(original_image,segmented_image,L)
#       return new_image
        new_segmentation=watershed_circularity(new_image)
        return new_segmentation
        

             
