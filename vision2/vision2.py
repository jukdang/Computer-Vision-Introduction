from PIL import Image
import numpy as np
import math


def boxfilter(n):
    #exit if n is odd
    assert n % 2 == 1, 'Dimension must be odd'
    
    # n,n size array fill with values sum is 1
    return np.full((n,n), 1.0/(n*n))

print(boxfilter(3)) # print(boxfilter(4)) # print(boxfilter(7))


def gauss1d(sigma):
    #get array size with sigma
    length = math.ceil(sigma * 6)
    if(length % 2 == 0):
        length += 1
    
    x = np.array(range(-int(length/2), int(length/2)+1))
    
    #mapping gaussian function to array
    fun = np.vectorize(lambda x : (math.exp(-(float)(x**2) / (float)(2*(sigma**2)))))
    arr = fun(x)
    
    #normalize to sum is 1
    sum = np.sum(arr)
    arr /= sum
    
    return arr

print(gauss1d(0.3)) # print(gauss1d(0.5)) # print(gauss1d(1)) # print(gauss1d(2))


def gauss2d(sigma):
    #make gauss2d using outer two gauss1d 
    arr = np.outer(gauss1d(sigma), gauss1d(sigma))
    
    #normalize to sum is 1
    sum = np.sum(arr)
    arr /= sum
    
    return arr


print(gauss2d(0.5)) # print(gauss2d(1))


def convolve2d(array, filter):
    #make padding with 0, because of calculating edge pixels with filter
    padSize = int(len(filter) / 2)
    paddingImg = np.pad(array,((padSize,padSize),(padSize,padSize)),mode='constant',constant_values=0)
    
    #flip filter, make eaiser calculating convolution 
    fliter = np.flip(filter)
    
    #caculated array
    arr = np.ones((len(array),len(array[0]))).astype(np.float32)
    
    #calculate
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            arr[i][j] = np.sum(paddingImg[i:i+len(filter), j:j+len(filter)] * fliter)
    
    return arr


def gaussconvolve2d(array, sigma):
    #filter is gauss2d
    return convolve2d(array,gauss2d(sigma))


img = Image.open("2b_dog.bmp")
img = img.convert('L')

#to calculate convert type int to float
imgArr = np.asarray(img).astype(np.float32)

#get calculated array, and to make image data convert type float to int
convolvedImg = gaussconvolve2d(imgArr,3).astype(np.uint8)

#array to image
convolvedImg = Image.fromarray(convolvedImg)

img.show()
convolvedImg.show()


def getLowFrequencyImage(array, sigma):
    #split RGB Channel to calculate seperately
    split = np.dsplit(array,3)
    for i in range(3):
        #calculate seperately 
        #reshape input arg (x,x,1)->(x,x)
        #reshape output (x,x)->(x,x,1)
        split[i] = gaussconvolve2d(split[i].reshape(len(split[i]),len(split[i][0])),sigma).reshape(len(split[i]),len(split[i][0]),1)
    
    #concatenate splited RGB channel
    arr = np.concatenate((split[0],split[1],split[2]),axis=2)
    return arr

lImg = Image.open("2b_dog.bmp")
lImgArr = np.asarray(lImg).astype(np.float32)
lImgArr = getLowFrequencyImage(lImgArr,5) #set sigma=5
lowImg = Image.fromarray(lImgArr.astype(np.uint8))
lowImg.show()


def getHighFrequencyImage(array,sigma):
    #get high-frequency Image, add 128 to visualize
    return array - getLowFrequencyImage(array,sigma) + 128

hImg = Image.open("2a_cat.bmp")
hImgArr = np.asarray(hImg).astype(np.float32)
hImgArr = getHighFrequencyImage(hImgArr,5) #set sigma=5
highImg = Image.fromarray(hImgArr.astype(np.uint8))
highImg.show()


def getHybridImage(Image1, Image2, sigma1, sigma2):
    #get low-frequency Iamge
    firstImg = Image.open(Image1)
    firstImgArr = np.asarray(firstImg).astype(np.float32)
    LowImgArr = getLowFrequencyImage(firstImgArr,sigma1)
    Image.fromarray(LowImgArr.astype(np.uint8)).show()
    
    #get high-frequency Iamge
    secondImg = Image.open(Image2)
    secondImgArr = np.asarray(secondImg).astype(np.float32)
    ighImgArr = getHighFrequencyImage(secondImgArr,sigma2)
    #Image.fromarray(HighImgArr.astype(np.uint8)).show()

    #add low & high, to make hybrid Image
    #change invalid value(not in range between 0 and 255)
    HybridImgArr = LowImgArr + HighImgArr - 128
    HybridImgArr[HybridImgArr>255]=255
    HybridImgArr[HybridImgArr<0]=0
    HybridImg = Image.fromarray(HybridImgArr.astype(np.uint8))
    
    return HybridImg


HybridImg = getHybridImage("2b_dog.bmp", "2a_cat.bmp", 5, 5)
HybridImg.show() # HybridImg.save("hybrid.jpg")




