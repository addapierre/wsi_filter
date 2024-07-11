import numpy as np
import cv2
import sys

def filter_background(bgrimg, filter_hue_start = None,  filter_hue_end = None):
    """returns filtered image and mask astype int"""
    hls = cv2.cvtColor(bgrimg.copy(), cv2.COLOR_BGR2HLS)
    h, l, _ = cv2.split(hls)
    s = cv2.cvtColor(bgrimg.copy(), cv2.COLOR_BGR2HSV)[:,:,1]

    test_white = np.logical_or(l>220,  s<20) # select white and light grey

    if filter_hue_start and filter_hue_end:
        if filter_hue_start < filter_hue_end:
            test_color = (h>=filter_hue_start) & (h<=filter_hue_end) # select hues
        else:
            test_color = (h>=filter_hue_start) | (h<=filter_hue_end)
        test = np.logical_not(np.logical_or(test_color, test_white))
    else:
        test = np.logical_not(test_white)

    hls[:,:,1] = hls[:,:,1]*test
    hls[:,:,2] = hls[:,:,2]*test

    return cv2.cvtColor(hls, cv2.COLOR_HLS2BGR), np.clip(hls[:,:,1], 0, 1, dtype = np.uint8)


def filter_small_holes(mask, resolution, thresh = 0.004):

        """
        thresh : filter holes smaller than threshold in mm2
        """
        nb_pixels = thresh * resolution[0] * resolution[1]
        
        mask = np.logical_not(mask).astype(np.uint8) 
        stats = cv2.connectedComponentsWithStats(mask, connectivity=8)
        sizes = stats[2][1:,4]
        small_holes_labels = np.where(sizes>nb_pixels)[0]+1
        tt = np.isin(stats[1], small_holes_labels)
        mask = np.logical_not(tt).astype(np.uint8)
        return mask


def filter_small_blobs(bgrimg, resolution, coloration = 'hes', thresh = 0.1 ):
    """
    filter out blobs with an area smaller than the threshold.
    threshold in mm2
    resolution in pix/mm in (x,y) direction
    """
    hls = cv2.cvtColor(bgrimg, cv2.COLOR_BGR2HLS)
    _, l, s = cv2.split(hls)
    nb_pixels = (thresh * resolution[0] * resolution[1])
    _, CC_mask, stats, _ = cv2.connectedComponentsWithStats(s, connectivity=8)
    sizes = stats[1:,4]                                                         # we ignore the first label, which is the background
    index = (np.where(sizes > nb_pixels)[0]+1).astype(np.int32)                 # +1 because we did stats [1:,4]
    _ = np.isin(CC_mask, index)
    loc0 = np.where(_)
    
    mask_index = CC_mask[loc0]
    # hackey optimization
    sizes = sizes[index-1]                  # sizes corresponds to the start value of each label in the following argsort
    sorted_index = np.argsort(mask_index)

    if coloration == "hes":
        l_lim = 100
    else:
        l_lim = 20

    mask = np.zeros(CC_mask.shape)
    j = 1
    n = 0
    for i in range(len(index)):
        loc1 = sorted_index[n:n+sizes[i]]
        n+=sizes[i]
        loc = (loc0[0][loc1], loc0[1][loc1])
        lmean = l[loc].mean()
        smean = s[loc].mean()

        if lmean > l_lim:
            if smean > 20: 
                if np.mean((s-smean)**2)>10: #faster way to get s std
                    mask[loc] = j
                    j+=1
    
    return mask.astype(np.uint8)
    

def lum_contrast(bgrimg):
    hls = cv2.cvtColor(bgrimg, cv2.COLOR_BGR2HLS)
    h,l,s = cv2.split(hls)
    cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = cl.apply(l)
    hls = cv2.merge((h,l,s))
    return cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)

def separate_purple_orange(bgrimg, mask, rotate_wheel = True, keep_images = True):
    """returns purple image, orange image and orange ratio, if keep images is True"""

    hls = cv2.cvtColor(bgrimg.copy(), cv2.COLOR_BGR2HLS)
    h = hls[:,:,0]
    #90° rotation of the hue wheel
    if rotate_wheel:
        h = h-90 # there will be a gap of 76, closing it is too time consuming

    limit = 84
    if rotate_wheel and limit>90:
        limit += 76
    # mask = 0 on background
    test_orange = np.logical_and((h>limit) , mask) # orange
    orange_ratio = (test_orange.sum() / mask.sum()) * 100
    
    if not keep_images:
        return orange_ratio, None, None

    im_purple = bgrimg.copy()
    im_orange = bgrimg.copy()

    test_purple = np.logical_and((h<limit) , mask) # purple
    test_purple = np.dstack([test_purple,]*3).astype(np.uint8)
    test_orange = np.dstack([test_orange,]*3).astype(np.uint8)
    im_purple = im_purple * test_purple + (test_purple-1)
    im_orange = im_orange * test_orange + (test_orange-1)

    return  orange_ratio, im_purple, im_orange

def separate_colors(bgrimg, mask, min_hue = 0, max_hue = 84, keep_images = True):
    """
    separates the image into 2 hue regions, calculates the ratio of each color area over the whole sample area, returns the images if keep_image = True.\n
    returns: ratio_color1, ratio_color2, img_color1, img_color2
    """
    hls = cv2.cvtColor(bgrimg.copy(), cv2.COLOR_BGR2HLS)
    h = hls[:,:,0]
    loc0 = np.where(mask)
    data = h[loc0]
    
    min_hue = min_hue%180
    max_hue = max_hue%180
    if min_hue < max_hue:
        test_color1 = np.where((data>=min_hue) & (data<=max_hue))
        test_color2 = np.where((data<min_hue) | (data>max_hue))
    else:
        test_color1 = np.where((data>=min_hue) | (data<=max_hue))
        test_color2 = np.where((data<min_hue) & (data>max_hue))
    blob_area = mask.sum()
    ratio1 = test_color1[0].shape[0]/blob_area*100
    ratio2 = test_color2[0].shape[0]/blob_area*100

    if not keep_images:
        return ratio1, ratio2, None, None
    
    img_color1 = bgrimg.copy()
    img_color2 = bgrimg.copy()
    loc1 = (loc0[0][test_color1], loc0[1][test_color1])
    loc2 = (loc0[0][test_color2], loc0[1][test_color2])
    img_color1[loc1] = 255
    img_color2[loc2] = 255
    return ratio1, ratio2, img_color1, img_color2




def _separate_colors(bgrimg, mask, min1 = 0, max1 = 84, min2 = 85, max2 = 180, rotate_wheel = True, keep_images = True):
    """separates 2 colors, calculates the ratio of each color area over the whole sample area, returns the images if keep_image = True.\n
    returns ratio_color1, ratio_color2, img_color1, img_color2
    """
    hls = cv2.cvtColor(bgrimg.copy(), cv2.COLOR_BGR2HLS)
    h = hls[:,:,0]
    loc0 = np.where(mask)
    data = h[loc0]
    #90° rotation of the hue wheel
    if rotate_wheel:
        data = data-90 
        _ = np.where(data>100)
        data[_] = data[_]-76
    test_color1 = np.where((data>=min1) & (data<=max1))
    test_color2 = np.where((data>=min2) & (data<=max2))
    blob_area = mask.sum()
    ratio1 = test_color1[0].shape[0]/blob_area*100
    ratio2 = test_color2[0].shape[0]/blob_area*100

    if not keep_images:
        return ratio1, ratio2, None, None
    
    img_color1 = bgrimg.copy()
    img_color2 = bgrimg.copy()

    # test_color1 = np.dstack((test_color1,)*3).astype(np.uint8)
    # test_color2 = np.dstack((test_color2,)*3).astype(np.uint8)

    # img_color1 = img_color1*test_color1 + (test_color1-1)
    # img_color2 = img_color1*test_color1 + (test_color1-2)
    loc1 = (loc0[0][test_color1], loc0[1][test_color1])
    loc2 = (loc0[0][test_color2], loc0[1][test_color2])
    img_color1[loc1] = 255
    img_color2[loc2] = 255

    return ratio1, ratio2, img_color1, img_color2




if __name__ == "__main__":
    img = cv2.imread("data/test_images/bird.png")
    im0 = img.copy()
    cv2.namedWindow('test-window', cv2.WINDOW_NORMAL)
    while True:

        cv2.imshow('test-window', img)
        if (tt := cv2.waitKey(10) & int(0xFF)) != 255:

            if tt == ord('q'):
                sys.exit()

            if tt == ord('m'):
                tt, mask = filter_background(im0)
                mask = mask.astype(np.uint8)*255
                mask = np.dstack((mask,)*3)
                print( mask.sum())
                img = mask

            # if tt == ord('f'):
            #     tt, mask = filter_blue_white(im0)
            #     img = tt
