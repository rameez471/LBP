import cv2
import numpy as np
import argparse
from PIL import Image
import matplotlib.pyplot as plt

def getLBPimage(image):
    '''
    Input: Image of shape (height,width)
    Output: LBP converted image of same shape
    '''
    #Convert the image into grayscale image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imgLBP = np.zeros_like(gray_image)
    neighbor = 3

    for ih in range(0,image.shape[0] - neighbor):
        for iw in range(0,image.shape[1] - neighbor):
            img = gray_image[ih:ih+neighbor,iw:iw+neighbor]
            center = img[1,1]
            img01 = (img >= center) * 1.0
            img01_vector = img01.flatten()
            img01_vector = np.delete(img01_vector,4)
            where_img01_vector = np.where(img01_vector)[0]
            
            if len(where_img01_vector) >= 1:
                num = np.sum(2**where_img01_vector)
            else:
                num = 0

            imgLBP[ih+1,iw+1] = num
        
    return imgLBP

def blockshaped(arr, nrows, ncols):
    '''
    Input: Image in numpy array form

    Output: Images divided into blocks
    '''
    h,w = arr.shape
    return (arr.reshape(h//nrows, nrows,-1, ncols)
                        .swapaxes(1,2)
                        .reshape(-1, nrows, ncols))

def histogram(imgArray, plot=False):
    '''
    Input: Image divided into grids

    Output: Concentrated Histogram of Input Image
    '''
    hist, bin_edges = np.histogram(imgArray,density=True)

    if plot:
        plt.hist(hist,bins=bin_edges)
        plt.show()

    return hist


def _main():
    
    parser = argparse.ArgumentParser(description='Enter Image: ')
    parser.add_argument('--image',type=str,help='Location of Image')
    parser.add_argument('-y',action='store_true')
    args = parser.parse_args()

    image = Image.open(args.image)
    image = np.array(image)

    lbp_image = getLBPimage(image)
    img = Image.fromarray(lbp_image)
    img.show()
    img.save('../Results/jon_snow.jpg')

    if(args.y):
        vecImgLbp = lbp_image.flatten()
        plt.hist(vecImgLbp,bins=2**8)
        plt.show()


      


_main()



    
