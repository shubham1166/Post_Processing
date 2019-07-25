# Post-processing on cell data 


## Need for post-processing

In my other [repository](https://github.com/shubham1166/A-Comparision-between-U-Net-and-SegNet) where i did a comparision between SegNet and U-Net segmentation , results of U-Net were better but there were still many images in which the cells were connected that should not have been connected. So there is a need of some post processing.

---
## Dataset
The dataset that I have used is a kaggle dataset for identification and segmentation of nuclei in cells. The dataset consists of 670 images and each of the image is an RGB image with dimension 512×512. Below is the figue of the type of data that we have for segmentation.
**![](https://lh6.googleusercontent.com/Ngzs_qC2dUCs-fRkOOVSumBDYS8R3KI69cVdTWaQA6SxM2Qmlsh6tr39SlN5R_6kn_iV_l3xiAS6B6Lwvl96LL_Yzwj18t3c1H0JSyzHDlt4Q7aRoD2I1qkzjgeXUDnq_HcpO5wR)**
**Fig**: Row1 consists of original images followed by row2 that consists of original labelled images(Source:[Kaggle](https://www.kaggle.com/paultimothymooney/identification-and-segmentation-of-nuclei-in-cells))

In my approach , I have devided the image in 4 different types namely:

1. **Type 1**: Type 1 images are all the colored images like:
**![](https://lh5.googleusercontent.com/WKGNN-4vSsYmQL33qJn0t6htKL1DW7EpQYzqb642zoW94GAdKwq028Iinj2PrYBBX3iif5ccpqAcliTuJsrbUzOBZUahD3XPyN3rVvRZ5jDXxZ2u6h1tl-9iUjWFCDNea9_thCo0)**
2. **Type 2**: Type 2 images are all the black background images with bidd nuclei/cells like:
**![](https://lh6.googleusercontent.com/mij22mgjr932oRW8qxzBZB2h6xMb0Jo-1GMHYyQdRj9CBVFd1lhkorPJSidJT1Lsd4t5y31QE_hq38Jtl0SeeyoE-Lc3hNHGShyU1LrNZclo3C0V_4oavUvuTRyPT0eOoW_DeM-h)**
3. **Type 3**: Type 3 images are all the black background images with small cells like:
**![](https://lh6.googleusercontent.com/OOQ0nTL8R_fOCOZjsJdXS7PlJj52FRJS0lWcqIbLGVBcCf0m99PYggg_TD-1QEBfVyIwOllblsvMSCFaYRFATqb1agF_rxWEbiabv2C0NfwqGT7jsYYwI9Q5VX2vbODl43jC-fO8)**
4. Type 4: Type 4 images are all the images with white background like:
**![](https://lh4.googleusercontent.com/obuu12e6fDvtXgPKJ5ygCzF-pruGoBhwjFRL3A6Eo6cCsk2wcduJ5oPlAAiMYQw0BE80fIpvuJ4iAi8LQMZvJvCsAviCDWanbw6l3zHPtgrYT-y3m-5UV2gLYAlEAyRLRRYBcbzq)**

The reason of devising the post processed image was different behaviors of the images.

---
## Post-Processing
The post-processing has been done by two different methods:

1. **Converting to L*a*b and then thresholding** : The images are converted in L*a*b format and then the segmented image from the U-net output is treated as a mask on the original images and Otsu thresholding has been applied on the non-zero brightness 'L' values. The images are divided in four types as there as in type 1 some image reconstruction was used and in type 2 and 3 contrast enhancement was used and type 4 didnot need any post-processing.
**![](https://lh6.googleusercontent.com/xWEv17u3ytP3RJjWrcnAET-SjooK-NDGdo4tQyn5sY_KirBYuqoA6cZXk95UP6x5P0OLC5W2o4w2r67QpUoJL0QplIfjJusKr4kZOETBfxvrZ-5mHo1Ahzzbuww9kjHsno2PGN0h)**
**Fig**:The figure consists of Original image followed by Predicted output followed by post-processed image

2. **Using Circularity and Watershed together**, where
**![](https://lh3.googleusercontent.com/9bU4uTNYfMK9zq6rfoaF6bFV4Okhs1lt3G83xGqlQE507hJF7KFLtSSJ-_GSQxiasYx_0nTDOFPImsgt3RLRhc_wTTShmzkDyiagJo9iOu92aUsjtMFaQ2u2ZOrmsWVywUpzrDgn)**,where *C* is the circularity, *A* is the area and *P* is the perimeter. In the labeled images, each connected component is checked and its circularity is checked and then watershed algorithm is applied to all the connected components where circularity was less than 0.6.
![](https://www.overleaf.com/project/5d1dcc6dbb53f75109902303/file/5d2c25e4e1deed5875ead6ad)
**Fig**:The figure consists of Original image followed by Original label followed by post-processed image

---
## Results
**![](https://lh3.googleusercontent.com/9JBEYsDaJptrZBYZNULyQJ8N4zvaoxPEUjM3O25NTAoBT2ws75jlmbC_ckVtnQ8uM5b82uWFE9Iyb9OpF1iqznx2bQ1NcMEUnPUvoxON_EjHP7q3gxUCdlnCVWYJD_qwEIUuEtNb)**

The dice loss in case of post-processed images is more than that of U-Net and SegNet  because of the labeled images as in labeled images there are cells that are connected.


 
