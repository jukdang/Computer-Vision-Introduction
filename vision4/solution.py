import numpy as np
import math
import random


def RANSACFilter(
        matched_pairs, keypoints1, keypoints2,
        orient_agreement, scale_agreement):
    """
    This function takes in `matched_pairs`, a list of matches in indices
    and return a subset of the pairs using RANSAC.
    Inputs:
        matched_pairs: a list of tuples [(i, j)],
            indicating keypoints1[i] is matched
            with keypoints2[j]
        keypoints1, 2: keypoints from image 1 and image 2
            stored in np.array with shape (num_pts, 4)
            each row: row, col, scale, orientation
        *_agreement: thresholds for defining inliers, floats
    Output:
        largest_set: the largest consensus set in [(i, j)] format

    """
    assert isinstance(matched_pairs, list)
    assert isinstance(keypoints1, np.ndarray)
    assert isinstance(keypoints2, np.ndarray)
    assert isinstance(orient_agreement, float)
    assert isinstance(scale_agreement, float)
    

    SCALE = 2 
    ORIENT = 3
    
    random_pair = random.sample(matched_pairs, 10) # get 10 random sample
    inline_score = []
    inline_data = []
    
    def scaleOrient(r): # function for scale orientation to (0~2*pi)
        while not(0 <= r <= np.pi * 2):
            if(r<0):
                r += np.pi * 2
            else:
                r -= np.pi * 2
        return r
            
        
    for pair in random_pair:
        score = 0
        i,j = pair
        
        # get range for comparing each pair is inline
        #scale
        scale_std  = keypoints2[j][SCALE] / keypoints1[i][SCALE]
        minScale = scale_std - scale_std * scale_agreement
        maxScale = scale_std + scale_std * scale_agreement
        #orientation
        orient_std = keypoints2[j][ORIENT] - keypoints1[i][ORIENT]
        minOrient = scaleOrient(orient_std - math.radians(orient_agreement))
        maxOrient = scaleOrient(orient_std + math.radians(orient_agreement))
        
        data = []
        for comp in matched_pairs: # compare with others
            i,j = comp
            
            cScale = keypoints2[j][SCALE] / keypoints1[i][SCALE]
            cOrient = scaleOrient(keypoints2[j][ORIENT] - keypoints1[i][ORIENT])
            if(minOrient < maxOrient): # k degree ~ k+a degree
                if( minScale < cScale < maxScale and minOrient < cOrient < maxOrient ): 
                    score += 1
                    data.append(comp)
            else: # when 0 degree exist between two angle
                if( minScale < cScale < maxScale and (minOrient < cOrient or maxOrient > cOrient) ):
                    score += 1
                    data.append(comp)
            
                
        inline_score.append(score)
        inline_data.append(data)
    # get inline ones when the number of inline ones is the largest
    largest_set = inline_data[inline_score.index(max(inline_score))]
    

    assert isinstance(largest_set, list)
    return largest_set



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


def FindBestMatchesRANSAC(
        keypoints1, keypoints2,
        descriptors1, descriptors2, threshold,
        orient_agreement, scale_agreement):
    """
    Note: you do not need to change this function.
    However, we recommend you to study this function carefully
    to understand how each component interacts with each other.

    This function find the best matches between two images using RANSAC.
    Inputs:
        keypoints1, 2: keypoints from image 1 and image 2
            stored in np.array with shape (num_pts, 4)
            each row: row, col, scale, orientation
        descriptors1, 2: a K-by-128 array, where each row gives a descriptor
        for one of the K keypoints.  The descriptor is a 1D array of 128
        values with unit length.
        threshold: the threshold for the ratio test of "the distance to the nearest"
                   divided by "the distance to the second nearest neighbour".
                   pseudocode-wise: dist[best_idx]/dist[second_idx] <= threshold
        orient_agreement: in degrees, say 30 degrees.
        scale_agreement: in floating points, say 0.5
    Outputs:
        matched_pairs_ransac: a list in the form [(i, j)] where i and j means
        descriptors1[i] is matched with descriptors2[j].
    Detailed instructions are on the assignment website
    """
    orient_agreement = float(orient_agreement)
    assert isinstance(keypoints1, np.ndarray)
    assert isinstance(keypoints2, np.ndarray)
    assert isinstance(descriptors1, np.ndarray)
    assert isinstance(descriptors2, np.ndarray)
    assert isinstance(threshold, float)
    assert isinstance(orient_agreement, float)
    assert isinstance(scale_agreement, float)
    matched_pairs = FindBestMatches(
        descriptors1, descriptors2, threshold)
    matched_pairs_ransac = RANSACFilter(
        matched_pairs, keypoints1, keypoints2,
        orient_agreement, scale_agreement)
    return matched_pairs_ransac
