from PIL import Image
import numpy as np

# open the test image
im = Image.open('chipmunk.png')

#im.show()

# convert the image to a black and white "luminance" greyscale image
im = im.convert('L')

# select a 100x100 sub region (containing the chipmunk's head)
im2 = im.crop((280,150,430,300))

im2.save('chipmunk_head.png','PNG')

# convert the image to a numpy array (for subsequent processing)
im2_array = np.asarray(im2)

# Note: we need to make a copy to change the values of an array created using
im3_array = im2_array.copy()

# add 50 to each pixel value (clipping above at 255, the maximum uint8 value)
for x in range(0,150):
    for y in range(0,150):
        im3_array[y,x] = min(im3_array[y,x] + 50, 255)
        
im3 = Image.fromarray(im3_array)
im3.save('chipmunk_head_bright.png','PNG')

# again make a copy of the (original) 100x100 sub-region
im4_array = im2_array.copy()

# reduce the intensity of each pixel by half
im4_array = im4_array * 0.5
im4_array = im4_array.astype('uint8')

im4 = Image.fromarray(im4_array)
im4.save('chipmunk_head_dark.png','PNG')

# let's generate our own image, a simple gradient test pattern
# make a 1-D array of length 256 with the values 0 - 255
grad = np.arange(0,256)

# repeat this 1-D array 256 times to create a 256x256 2-D array
grad = np.tile(grad,[256,1])

im5 = Image.fromarray(grad.astype('uint8'))
im5.save('gradient.png','PNG')
