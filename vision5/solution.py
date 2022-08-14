import numpy as np
import cv2
import math
import random

def FindBestMatches(descriptors1, descriptors2, threshold):
    """
    This function takes in descriptors of image 1 and image 2,
    and find matches between them. See assignment instructions for details.
    Inputs:
        descriptors: a K-by-128 array, where each row gives a descriptor
        for one of the K keypoints.  The descriptor is a 1D array of 128
        values with unit length.
        threshold: the threshold for the ratio test of "the distance to the nearest"
                   divided by "the distance to the second nearest neighbour".
                   pseudocode-wise: dist[best_idx]/dist[second_idx] <= threshold
    Outputs:
        matched_pairs: a list in the form [(i, j)] where i and j means
                       descriptors1[i] is matched with descriptors2[j].
    """
    assert isinstance(descriptors1, np.ndarray)
    assert isinstance(descriptors2, np.ndarray)
    assert isinstance(threshold, float)

    matched_pairs = []
    
    for i,d1 in enumerate(descriptors1): #check all vectors in descriptor1
        pair_list = []
        for j,d2 in enumerate(descriptors2): # compare angles with vectors in descriptor2
            pair = math.acos( np.dot(d1,d2) ) 
            pair_list.append(pair)
        sorted_list = sorted(pair_list)
        
        # compare best match angle to the second best angle
        if(sorted_list[0] <= sorted_list[1] * threshold):
            j = pair_list.index(sorted_list[0])
            matched_pairs.append([i,j])

    return matched_pairs


def KeypointProjection(xy_points, h):
    """
    This function projects a list of points in the source image to the
    reference image using a homography matrix `h`.
    Inputs:
        xy_points: numpy array, (num_points, 2)
        h: numpy array, (3, 3), the homography matrix
    Output:
        xy_points_out: numpy array, (num_points, 2), input points in
        the reference frame.
    """
    assert isinstance(xy_points, np.ndarray)
    assert isinstance(h, np.ndarray)
    assert xy_points.shape[1] == 2
    assert h.shape == (3, 3)

    xy_points_out = np.empty((0,2))

    for xy_point in xy_points:
        xy_point = np.append(xy_point,1)
        proj_point = np.dot(h,xy_point) # projection
        if(proj_point[2]==0): # If extra dimension is zero, replace 0 to 1e-10
            proj_point[2] = 1e-10
        proj_point = proj_point / proj_point[2] # make extra dimension to 1
        xy_points_out = np.append(xy_points_out, np.array([proj_point[0:2]]),axis=0)

    return xy_points_out

def RANSACHomography(xy_src, xy_ref, num_iter, tol):
    """
    Given matches of keyponit xy coordinates, perform RANSAC to obtain
    the homography matrix. At each iteration, this function randomly
    choose 4 matches from xy_src and xy_ref.  Compute the homography matrix
    using the 4 matches.  Project all source "xy_src" keypoints to the
    reference image.  Check how many projected keyponits are within a `tol`
    radius to the coresponding xy_ref points (a.k.a. inliers).  During the
    iterations, you should keep track of the iteration that yields the largest
    inlier set. After the iterations, you should use the biggest inlier set to
    compute the final homography matrix.
    Inputs:
        xy_src: a numpy array of xy coordinates, (num_matches, 2)
        xy_ref: a numpy array of xy coordinates, (num_matches, 2)
        num_iter: number of RANSAC iterations.
        tol: float
    Outputs:
        h: The final homography matrix.
    """
    assert isinstance(xy_src, np.ndarray)
    assert isinstance(xy_ref, np.ndarray)
    assert xy_src.shape == xy_ref.shape
    assert xy_src.shape[1] == 2
    assert isinstance(num_iter, int)
    assert isinstance(tol, (int, float))
    tol = tol*1.0

    maxH = None
    maxInlier = 0
    for _ in range(num_iter): # RANSAC
        idx = random.sample(range(len(xy_src)),4) #get 4 points to make homography matrix
        zero = [0,0,0]
        matrix = []
        for i in idx:
            src = list(xy_src[i])
            ref = list(xy_ref[i])
            mix1 = [-ref[0]*src[0], -ref[0]*src[1],-ref[0]]
            mix2 = [-ref[1]*src[0], -ref[1]*src[1],-ref[1]]
            matrix.append(src+[1]+zero+mix1)
            matrix.append(zero+src+[1]+mix2)
        matrix = np.asarray(matrix) # homography matrix
        # Solving for homographies,
        # we need to get eigenvector of A.T * A with smallest eigenvalue 
        u,s,v = np.linalg.svd(matrix) 
        h = np.reshape(v[8],(3,3))

        inlierCount = 0
        for i in range(len(xy_src)):
            src = xy_src[i]
            ref = xy_ref[i]
            proj = np.dot(h,np.append(src,1).T) # projection
            if(proj[2]==0): # If extra dimension is zero, replace 0 to 1e-10
                proj[2]=1e-10
            proj = proj / proj[2] # make extra dimension to 1

            distance = np.hypot(proj[0]-ref[0],proj[1]-ref[1]) # euclidean distance
            if(distance<tol): #threshold with 'tol'
                inlierCount += 1
        if(inlierCount>maxInlier): # if it's better H
            maxInlier = inlierCount
            maxH = h
        
    h = maxH
    
    assert isinstance(h, np.ndarray)
    assert h.shape == (3, 3)
    return h
