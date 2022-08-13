###Canny Edge Detection

1. noise reduction 
 - blurring image
 
2. sobel filter
 - find gradient, theta
 
3. non maximum suppression
 - make edge sharp
 - find local maximum 
 
4. Double thresholding
 - Weak edge, Strong edge
 
5. Edge Tracking by hysteresis
 - transform weak edge attached to strong edge 
 - by bfs
