'''
Created on Jun 24, 2019

@author: T01130
'''
import numpy as np
import UNET_
import keras
import matplotlib
import cv2
from skimage.filters import threshold_otsu
from skimage.morphology import reconstruction
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage



#######################################################################
width=512
height=512
depth=3
classes=2
pathtoweight1='E:/Cell_data/U-Net/'
keras.backend.set_image_data_format('channels_first')
net = U_Net(width, height, depth, classes,weightsPath=pathtoweight1+'Cell.h5_0183-0.9988.h5')

def whitening(im):
    """
    As the images that we have are Channel_first
    """
    im = im.astype("float32")              
    for i in range(np.shape(im)[0]):                                
        im[i,:,:] = (im[i,:,:]- np.mean(im[i,:,:]))#/(np.std(im[:,:,i])+1e-9)            
    return im
def give_output_unet(x):
    '''
      If the image is in channel last, then we'll convert the image in channel first
    '''
    if x.shape[-1]==3:
        x=np.transpose(x,[2,0,1])
    z=[]
    z.append(whitening(x))
    # Finding the probabilities of the outputs
    probs = net.predict(np.array(z))
    prediction= np.argmax(probs[0],axis=1)
    prediction = np.reshape(prediction,(512,512)) 
    prediction=prediction.astype("uint8")
    return prediction*255
###############################################################################


def get_average_area(segmented_image):
    '''
    Input image should be binary
    '''
    contours = cv2.findContours(segmented_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    area=[]
    for cnt in contours:
        area.append(cv2.contourArea(cnt))
    return np.mean(area)



#################################################################################\

def contrast_enhancement(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    return cl1

def reconstruct_image(image,segmented_image):
    try:
        background = reconstruction(segmented_image,image,method='dilation')
    except ValueError:
        try:
            background = reconstruction(segmented_image,image,method='erosion')
        except ValueError:
            return image
    background=background*255
    return background
def type_1_process(original_image,segmented_image,L):
    segmented_image = cv2.erode(segmented_image,np.ones((3,3),dtype="uint8"),iterations = 4)
    lab_original_image= cv2.cvtColor(original_image,cv2.COLOR_BGR2LAB)
    ########################################################################################################
    #Extracting the non zero cordinates to use brightness value using l*a*b
    L_cell=L*segmented_image
    L_cell=contrast_enhancement(L_cell)
    non_zero_cordinates=np.nonzero(segmented_image)
    #OR USE     non_zero_cordinates=np.where(binary_image!=0)
    x=L_cell[non_zero_cordinates]
    x=x.astype('float32')
    #x is the image non-zero cordinates to be thresholded
    val = threshold_otsu(x)
    z= x>val
    z=z.astype(int)
    new_segmentation = np.zeros((512,512),dtype="uint8")
    new_segmentation[non_zero_cordinates]=z*255
    new_image=reconstruct_image(segmented_image,new_segmentation)
    return new_image

def type_2_process(original_image,segmented_image):
    segmented_image = cv2.erode(segmented_image,np.ones((3,3),dtype="uint8"),iterations = 5)
    binary_image=segmented_image/255
    original_image=cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    contrast_image=contrast_enhancement(original_image)
    merged_image=binary_image*contrast_image
#     background = reconstruction(merged_image,segmented_image,method='dilation')
    non_zero_cordinates=np.nonzero(segmented_image)
    x=merged_image[non_zero_cordinates]
    val = threshold_otsu(x)
    z= x>=val
    z=z.astype(int)
    new_segmentation = np.zeros((512,512),dtype="uint8")
    new_segmentation[non_zero_cordinates]=255*z
    new_image=reconstruct_image(segmented_image,new_segmentation)
    return new_image


def type_3_process(img,segmented_image):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    segmented_image = cv2.erode(segmented_image,np.ones((3,3),dtype="uint8"),iterations = 3)
    segmented_image=segmented_image/255.
    bb=gray*segmented_image
    new_image=reconstruct_image(segmented_image,bb)
    return new_image
#############################################################################

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

def give_different_connected_components(image):
    y=image.astype('uint8')
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

def watershed_circularity(image):
    y=image
    z=give_different_connected_components(image)
    new_image= np.zeros(y.shape, dtype="uint8")
    for i in range(len(z)):
        if get_circularity(z[i])>=0.6:
            new_image[z[i] == 255] = 255
        else:
            watershed_image=give_watershed_segmentation(z[i])
            new_image[watershed_image == 255] = 255
    return new_image







    

    


