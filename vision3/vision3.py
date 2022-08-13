from PIL import Image
import math
import numpy as np
from queue import Queue


def gauss1d(sigma):
    length = math.ceil(sigma * 6)
    if(length % 2 == 0):
        length += 1
    
    x = np.array(range(-int(length/2), int(length/2)+1))
    
    fun = np.vectorize(lambda x : (math.exp(-(float)(x**2) / (float)(2*(sigma**2)))))
    arr = fun(x)
    
    sum = np.sum(arr)
    arr /= sum
    
    return arr


def gauss2d(sigma):
    arr = np.outer(gauss1d(sigma), gauss1d(sigma))
    
    sum = np.sum(arr)
    arr /= sum
    
    return arr


def convolve2d(array,fliter):
    padSize = int(len(fliter) / 2)
    paddingImg = np.pad(array,((padSize,padSize),(padSize,padSize)),mode='constant',constant_values=0)
    
    fliter = np.flip(fliter)
    
    arr = np.ones((len(array),len(array[0]))).astype(np.float32)
    
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            arr[i][j] = np.sum(paddingImg[i:i+len(fliter), j:j+len(fliter)] * fliter)
    
    return arr


def gaussconvolve2d(array,sigma):
    return convolve2d(array,gauss2d(sigma))


img = Image.open("iguana.bmp").convert('L')
img.show()
#img.save("greyscale.jpg")

imgArr = np.asarray(img).astype(np.float32)
blurArr = gaussconvolve2d(imgArr,1.6)
blur = Image.fromarray(blurArr.astype(np.uint8))
blur.show()
#blur.save("blur.jpg")


def sobel_filters(img):
    #x,y sobel filter
    xfilter = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], np.float32)
    yfilter = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], np.float32)
    
    #convolve
    Ix = convolve2d(img,xfilter)
    Iy = convolve2d(img,yfilter)
    
    G = np.hypot(Ix,Iy) # grandient = root(Ix^2 + Iy^2)
    G = G / G.max() * 255 # mapping value to 0~255
    theta = np.arctan2(Iy,Ix) # theta = arctan(Iy/Ix)
    
    return (G, theta)


G, theta = sobel_filters(blurArr)
grandientImg = Image.fromarray(G.astype(np.uint8))
grandientImg.show() # show gradient applying sobel_filter Suppression
#grandientImg.save("gradientImg.jpg")


def non_max_suppression(G, theta):
    theta = np.rad2deg(theta) # convert radian to degree
    theta[theta<0] += 180
    
    res = np.zeros(G.shape, dtype=np.float32)
    for i in range(1,G.shape[0]-1):
        for j in range(1,G.shape[1]-1):
            p = G[i][j]
            if(theta[i][j]<22.5 or theta[i][j]>=157.5): # degree 0
                q = G[i][j+1]
                r = G[i][j-1]
            elif(22.5<=theta[i][j]<67.5): # degree 45
                q = G[i+1][j-1]
                r = G[i-1][j+1]
            elif(67.5<=theta[i][j]<112.5): # degree 90
                q = G[i+1][j]
                r = G[i-1][j]
            else: #degree 135
                q = G[i+1][j+1]
                r = G[i-1][j-1]
            if(q<=p) and (r<=p):
                res[i][j]=p #local maximum

    return res


NMS = non_max_suppression(G, theta)
NMSImg = Image.fromarray(NMS.astype(np.uint8))
NMSImg.show() # show non_maximum_suppression
#NMSImg.save("NMS.jpg")


def double_thresholding(img):
    diff = img.max() - img.min()
    #threshold values
    highThreshold = img.min() + diff * 0.15
    lowThreshold = img.min() + diff * 0.03
    res = np.where(img>highThreshold, 255, np.where(img>lowThreshold, 80, 0)) #thresholding 255 / 80 / 0
    return res


DT = double_thresholding(NMS.astype(np.uint8))
DTImg = Image.fromarray(DT.astype(np.uint8))
DTImg.show() #show double_threshodling img
#DTImg.save("DT.jpg")


def hysteresis(img):
    q = Queue()
    dir = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]] # adjacent direction
    for i in range(1,img.shape[0]-1):
        for j in range(1,img.shape[1]-1):
            if(img[i][j]==255): # breadth first search from strong to weak 
                q.put((i,j))
                while(not(q.empty())):
                    x,y = q.get()
                    for nx,ny in dir:
                        if(img[x+nx][y+ny]==80):
                            img[x+nx][y+ny]=255
                            q.put((x+nx,y+ny))
                            
    res = np.where(img==255,255,0)
    
    return res


hy = hysteresis(DT)
hyImg = Image.fromarray(hy.astype(np.uint8))
hyImg.show() #show hysteresis img
#hyImg.save("hy.jpg")





