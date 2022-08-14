import hw_utils as utils
import matplotlib.pyplot as plt


def main():
    # Test run matching with no ransac
    plt.figure(figsize=(20, 20))
    
    first="library"
    second="library2"
    ratio = 0.6
    orient = 5
    scale = 0.2
    
    im = utils.Match('./data/'+first, './data/'+second, ratio_thres=ratio)
    plt.title('Match')
    plt.imshow(im)
    #im.save(first+"-"+second+"-ratio_"+str(ratio)+".png")



    # Test run matching with ransac
    plt.figure(figsize=(20, 20))
    im = utils.MatchRANSAC(
        './data/'+first, './data/'+second,
        ratio_thres=ratio, orient_agreement=orient, scale_agreement=scale)
    plt.title('MatchRANSAC')
    plt.imshow(im)
    #im.save(first+"-"+second+"-ratio_"+str(ratio)+"-orient_"+str(orient)+"-scale_"+str(scale)+".png")

if __name__ == '__main__':
    main()
