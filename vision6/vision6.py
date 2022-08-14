import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = plt.imread('./data/warrior_a.jpg')
img2 = plt.imread('./data/warrior_b.jpg')
cor1 = np.load("./data/warrior_a.npy")
cor2 = np.load("./data/warrior_b.npy")

#img1 = plt.imread('./data/graffiti_a.jpg')
#img2 = plt.imread('./data/graffiti_b.jpg')
#cor1 = np.load("./data/graffiti_a.npy")
#cor2 = np.load("./data/graffiti_b.npy")

def compute_fundamental(x1,x2):
    n = x1.shape[1]
    if x2.shape[1] != n:
        exit(1)
        
    F = None

    matrix = [] # build matrix for equation
    for i in range(n):
        line = [x1[0][i]*x2[0][i], x1[1][i]*x2[0][i], x2[0][i], 
                x1[0][i]*x2[1][i], x1[1][i]*x2[1][i], x2[1][i],
                 x1[0][i], x1[1][i], 1]
        matrix.append(line)
    matrix = np.asarray(matrix)

    u,s,v = np.linalg.svd(matrix)
    F = v[8].reshape(3,3) # F matrix when eigenvector is smallest

    u,s,v = np.linalg.svd(F)
    new_s = np.diag(s)
    new_s[2][2] = 0 # make rank 2 by zeroing out last singular value
    F = u @ new_s @ v
    
    
    return F


def compute_norm_fundamental(x1,x2):
    n = x1.shape[1]
    if x2.shape[1] != n:
        exit(1)

    # normalize image coordinates
    x1 = x1 / x1[2]
    mean_1 = np.mean(x1[:2],axis=1)
    S1 = np.sqrt(2) / np.std(x1[:2])
    T1 = np.array([[S1,0,-S1*mean_1[0]],[0,S1,-S1*mean_1[1]],[0,0,1]])
    x1 = T1 @ x1
    
    x2 = x2 / x2[2]
    mean_2 = np.mean(x2[:2],axis=1)
    S2 = np.sqrt(2) / np.std(x2[:2])
    T2 = np.array([[S2,0,-S2*mean_2[0]],[0,S2,-S2*mean_2[1]],[0,0,1]])
    x2 = T2 @ x2

    # compute F with the normalized coordinates
    F = compute_fundamental(x1,x2)

    # reverse normalization
    F = T2.T @ F @ T1
    
    return F


def compute_epipoles(F):
    e1 = None
    e2 = None

    
    u,s,v = np.linalg.svd(F)
    e1 = v[-1] #null space of F (Fx=0)
    e1 = e1 / e1[2] #regularization
    e1 = e1[:2] 
    
    u,s,v = np.linalg.svd(F.T)
    e2 = v[-1] #null space of F (Fx=0)
    e2 = e2 / e2[2] #regularization
    e2 = e2[:2]

    
    return e1, e2


def draw_epipolar_lines(img1, img2, cor1, cor2):
    F = compute_norm_fundamental(cor1, cor2)

    e1, e2 = compute_epipoles(F)
    print(e1,e2)

    # 이미지가 warrior일때 이미지 인식을 0~1사이 값으로 인식해 변환과정을 거침
    img1 = img1 * 255
    img2 = img2 * 255
    # graffiti 실행시 제거

    fig = plt.figure(figsize=(40,20))
    rows = 1
    cols = 2

    r,c,v = img1.shape
    for cor1, cor2 in zip(cor1.T, cor2.T):
        color = tuple(np.random.randint(125,255,3).tolist()) #select color

        cor1X = cor1[0]
        cor1Y = cor1[1]
        slope1 = (cor1Y - e1[1]) / (cor1X - e1[0])
        intercept1 = cor1Y - slope1 * cor1X
        line1 = np.poly1d([slope1,intercept1]) # epipole line

        cor2X = cor2[0]
        cor2Y = cor2[1]
        slope2 = (cor2Y - e2[1]) / (cor2X - e2[0])
        intercept2 = cor2Y - slope2 * cor2X
        line2 = np.poly1d([slope2,intercept2]) # epipole line
        
        # draw in img
        x0,y0 = map(int, [0, np.polyval(line1,0)])
        x1,y1 = map(int, [c, np.polyval(line1,c)])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,3)
        img1 = cv2.circle(img1,(int(cor1X),int(cor1Y)),9,color,-1)

        x0,y0 = map(int, [0, np.polyval(line2,0)])
        x1,y1 = map(int, [c, np.polyval(line2,c)])
        img2 = cv2.line(img2, (x0,y0), (x1,y1), color,3)
        img2 = cv2.circle(img2,(int(cor2X),int(cor2Y)),9,color,-1)


    ax1 = fig.add_subplot(rows,cols,1)

    #ax1.imshow(img1)
    ax1.imshow(img1.astype('uint8'), vmin=0,vmax=255)
    # graffiti 실행시 변환
    # 이미지가 warrior일때 이미지 인식을 0~1사이 값으로 인식해 변환과정을 거침
    
    ax2 = fig.add_subplot(rows,cols,2)

    #ax2.imshow(img2)
    ax2.imshow(img2.astype('uint8'), vmin=0,vmax=255)
    # graffiti 실행시 변환
    # 이미지가 warrior일때 이미지 인식을 0~1사이 값으로 인식해 변환과정을 거침
    
    plt.show()

    return

draw_epipolar_lines(img1, img2, cor1, cor2)
