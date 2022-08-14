### Canny Edge Detection

1. noise reduction 
    - blurring image
        + 원본 이미지에 가우시안 필터를 적용해 이미지에 블러 효과를 줌으로써 noise를 줄인다.
 
2. sobel filter
    - find gradient, theta
        + 수직, 수평 성분을 검출해내는 sobel filter를 2개 적용해 gradient와 방향을 구한다.
3. non maximum suppression
    - make edge sharp
        + find local maximum 
        + local maximum을 제외한 나머지 점들은 edge가 아닌것으로 판단한다.
 
4. Double thresholding
    - Weak edge, Strong edge
        + 상하단 threshlod를 이용해 strong edge와 weak edge로 구분한다.
 
5. Edge Tracking by hysteresis
    - transform weak edge attached to strong edge
        + strong edge를 기준으로 strond edge와 붙어있는 weak edge를 연결해주어 edge들을 이어지게 해준다.
        + 잘못 검출 된 weak edge들을 지워준다.
        + using BFS
