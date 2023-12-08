import subprocess
import pandas as pd
from subprocess import Popen, PIPE, STDOUT
import re
import time
import numpy as np
import os
import sys
import cv2
import concurrent.futures 
from itertools import repeat

class DisplayImage():

    def __init__(self, image) -> None:
        self.image = image.copy() # currently displayed image
        self.image_prezoom = self.image.copy()
        self.height = image.shape[0]
        self.width = image.shape[1]

        self.zoom_triggered = 0
        self.is_zoomed = 0

        self.x_init_zoom = None
        self.y_init_zoom = None
        self.resize_ratio = None
 
    def begin_zoom(self, x, y): 
        self.image_prezoom = self.image.copy()
        self.zoom_triggered = 1
        self.x_init_zoom = x
        self.y_init_zoom = y

    def zoom_rectangle(self, x, y):
        self.image = self.image_prezoom.copy()
        cv2.rectangle(self.image, (self.x_init_zoom, self.y_init_zoom), (x,y), (255,0,0), 2)

    def cancel_zoom(self):
        self.zoom_triggered = 0
        self.is_zoomed = 0
        self.image = self.image_prezoom.copy()

    def zoom(self, x, y):
        #try:
            self.image = self.image_prezoom.copy()
            self.zoom_triggered = 0
            self.is_zoomed = 1
            self.x_final_zoom = x
            self.y_final_zoom = y

            x_zoom_1 = self.x_init_zoom ; x_zoom_2 = self.x_final_zoom
            y_zoom_1 = self.y_init_zoom ; y_zoom_2 = self.y_final_zoom

            # cropping
            self.xlist = np.sort(np.array([x_zoom_1, x_zoom_2]))
            self.ylist = np.sort(np.array([y_zoom_1, y_zoom_2]))
            self.image = self.image[self.ylist[0] : self.ylist[1], self.xlist[0] : self.xlist[1]]

            # resize so that the area covered by the image is the same
            new_w = self.xlist[1] - self.xlist[0]
            new_h = self.ylist[1] - self.ylist[0]
            self.resize_ratio = np.sqrt(self.height*self.width/(new_h*new_w))
            try:
                new_size = np.array([int(new_w * self.resize_ratio), int(new_h * self.resize_ratio)])
            except :
                self.cancel_zoom()
                return None
            self.image = cv2.resize(self.image, new_size, interpolation = cv2.INTER_LANCZOS4)
        # except: 
        #     self.cancel_zoom()

    def display_text(self, text, x, y, color = (255,0,0), font_multiplicator = 1e-3, thickness_multiplicator = 1e-3, background_color = None):
        # Warning: this function changes self.image, make sure you made a copy before calling it
        font_scale = min(self.width, self.height) * font_multiplicator
        font_thickness = int(np.ceil(min(self.width, self.height) * thickness_multiplicator))
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, font_thickness)
        text_w, text_h = text_size
        if x+text_w > self.image.shape[1]:
            x = x-text_w
        if y - text_h < 0:
            y = y+text_h
        if isinstance(background_color, tuple):
            cv2.rectangle(self.image, (x, y-text_h-np.ceil(text_h/6).astype(int)), (x+text_w, y), background_color, -1)


        cv2.putText(self.image, text, (x, y), cv2.FONT_HERSHEY_DUPLEX,\
                    fontScale = font_scale, color = color, thickness = font_thickness, lineType = cv2.LINE_AA) 
        


###########################################################################################################################################################################

class ImageNDPI(DisplayImage):

    def __init__(self, ndpi_path):
        
        start = time.time()
        self.ndpi_path = ndpi_path
        assert os.path.exists(self.ndpi_path), f'Path error: no file found at   {self.ndpi_path}'
        cmd = ['tifftopnm', '-headerdump', ndpi_path]
        process = Popen(cmd, stdout=PIPE, stderr=STDOUT)
        header = process.communicate()[0].decode('ISO-8859-1')

        # Get Magnifications
        pattern = r'Tag 65421: (\d+)'
        _ =re.findall(pattern, header)
        self.mag_max = int(_[0])
        self.mag_list = np.array([self.mag_max/4.**i for i in range(6)])
        self.current_mag = self.wsi_mag = self.mag_list[np.where(self.mag_list>1)].min() # magnification closest to 1
        #Get resolution at mag_max
        pattern = r'Resolution: (\d+), (\d+) pixels/cm'
        numbersStrings = re.findall(pattern, header)
        self.max_resolution = np.array([int(n) for tup in numbersStrings for n in tup]) / 10 # in pix/mm for maximum resolution
        self.current_resolution = self.wsi_resolution = self.max_resolution * self.current_mag / self.mag_max

        # read wsi image:
        cmd = ['ndpisplit', f'-x{str(self.current_mag)}' ,'-e0,0,1,1', self.ndpi_path]
        subprocess.call(cmd)
        wsi_path = self.ndpi_path.replace('.ndpi', f'_x{str(self.current_mag)}_z0_1.tif')
        self.raw_wsi = cv2.imread(wsi_path) #WSI in BGR 
        
        os.remove(wsi_path)
        super().__init__(self.raw_wsi.copy())# current image is the wsi
        #print(f"open image: {time.time()-start}")
        self.blob_thresh = 0.1
        self.is_wsi = 1
        self.labels = None
        self.wsi_mask = None
        self.mask = None
        self.ratios = []
        self.areas = []
        self.ratio_displayed = False
        self.wsi = None


    def process_wsi(self, threshold = None):
        """
        removes blue, white, small blobs and black blobs
        Params: 
            - img: RGB image to preprocess
            - threshold: in mm2, blobs smaller than threshold are removed from the image. default: 0.2 mm2
        Return: 
            - preprocessed image
            - mask with only blobs larger than threshold. blob values in mask correspond to their label
        """
        if not self.is_wsi or self.wsi is not None:
            return None
        
        if threshold is None:
            threshold = self.blob_thresh
        start = time.time()
        self.image = filter_blue_white(self.image) 
        #print(f"filter blue/white : {time.time()-start}")

        start = time.time()
        self.wsi_mask = filter_small_blobs(self.image.copy(), thresh = threshold, resolution = self.current_resolution)
        #print(f"filter small blobs : {time.time()-start}")

        start = time.time()
        tt = np.stack((np.clip(self.wsi_mask.copy(), 0,1),)*3, axis = 2)
        self.image = (self.image * tt) + (tt-1) #mask the image with 255 values
        #print(f"mask the image : {time.time()-start}")

        start = time.time()
        self.labels = np.unique(self.wsi_mask)[1:]
        self.display_labels()
        #print(f"display labels : {time.time()-start}")
        self.wsi = self.image.copy()
        

            
    def display_labels(self):
        """
        displays blob labels on the wsi 
        """
        if self.labels is None or self.wsi_mask is None:
            return None
        for label in self.labels:
                
                y,x = np.where(self.wsi_mask == label)
                xmin = x.min() ; ymin = y.min()
                xmax = x.max() ; ymax = y.max()
                self.display_text(str(label), int((xmax+xmin)/2)-40, int((ymax+ymin)/2)+40)

    def open_label(self, label):
        if (label == 0) or not self.is_wsi:
            return None
        print(f"opening label {label}...")
        self.is_wsi = 0
        y,x = np.where(self.wsi_mask == label)
        xmin = x.min() ; ymin = y.min()
        xmax = x.max() ; ymax = y.max()

        #select new mag and new resolution based on target image size
        size_init = (xmax - xmin)*(ymax-ymin)/1e6 # target image size in Megapixel
        target_size = np.square(self.mag_list)*size_init/self.wsi_mag
        self.current_mag = self.mag_list[np.where(target_size<70)].max() # target image must be less than 50Mpix
        self.current_resolution = self.max_resolution * self.current_mag / self.mag_max

        #create, read and remove cropped image
        start = time.time()
        tt = str(self.current_mag).split('.')
        if int(tt[-1])==0:
            path = self.ndpi_path.replace('.ndpi', f'_x{int(self.current_mag)}_z0_1.tif')
        else:
            path = self.ndpi_path.replace('.ndpi', f'_x{round(self.current_mag, 2)}_z0_1.tif')
        cmd = ['ndpisplit', f'-x{str(self.current_mag)}' ,f'-e{xmin / self.width}, {ymin / self.height}, {(xmax-xmin) / self.width}, {(ymax-ymin) / self.height}', self.ndpi_path]
        subprocess.call(cmd)
        self.image = cv2.imread(path)
        
        if os.path.exists(path):
            os.remove(path)
        #print(f"create, read and remove cropped image: {time.time()-start}")

        # create label mask while we have the bounding box coordinates
        start = time.time()
        self.mask = self.wsi_mask[ymin:ymax, xmin:xmax].copy()
        self.mask[self.mask!=label] = 0
        self.mask = cv2.resize(self.mask, (self.image.shape[1], self.image.shape[0]))
        #print(f"crop/resize mask: {time.time()-start}")

        start = time.time()
        self.filter_small_holes()
        #print(f"filter small holes: {time.time()-start}")

        start = time.time()
        ratio = self.process_label()
        #print(f"process label: {time.time()-start}")
        return ratio

    def process_label(self):

        if self.is_wsi == 1:
            return None
                
        # loc = np.where(self.mask==0)
        # self.image[loc] = 255
        self.image = self.image * np.stack((self.mask,)*3, axis = 2) # masked parts set to 0
        self.image = self.image + (np.stack((self.mask,)*3, axis = 2)-1) # masked parts set to 255 <= way faster than doing img[mask==0] = 255
        self.image = filter_blue_white(self.image)
        # calculate area
        nb_pixel = (self.mask!=0).sum()
        spacing = 1/self.current_resolution
        pix_area = spacing[0]*spacing[1] #mm2/pix
        area = nb_pixel*pix_area

        self.pink, self.orange, ratio = separate_pink_orange(self.image)
        self.display_text(str(round(ratio, 2))+"%", 0, self.image.shape[0], font_multiplicator=0.005, thickness_multiplicator=0.008)
        self.original_image = self.image.copy()
        return ratio, area

    def process_all_labels(self):
        if self.wsi_mask is None:
            return None
        self.ratios = []
        self.areas = []
        for label in self.labels:
            #print(label, end=' | ')
            ratio, area = self.open_label(label)
            self.ratios.append(ratio)
            self.areas.append(area)
            #reinit
            self.reset()
            self.pink = self.orange = self.original_image = None
        self.ratios = np.array(self.ratios)
        self.areas = np.array(self.areas)
        full_area = self.areas.sum()

        ratio = (self.ratios*self.areas/full_area).sum()
        self.display_text(str(round(ratio, 2))+"%", 0, self.image.shape[0], font_multiplicator=0.005, thickness_multiplicator=0.004)
        self.wsi = self.image.copy()

    def filter_small_holes(self, thresh = 0.004):

        """
        thresh : filter holes smaller than threshold in mm2
        """
        nb_pix = thresh * self.current_resolution[0] * self.current_resolution[1]
        
        self.mask = np.logical_not(self.mask).astype(np.uint8) 
        stats = cv2.connectedComponentsWithStats(self.mask, connectivity=8)
        sizes = stats[2][1:,4]
        small_holes_labels = np.where(sizes>nb_pix)[0]+1
        tt = np.isin(stats[1], small_holes_labels)
        self.mask = np.logical_not(tt).astype(np.uint8)
        return None

    def display_ratio(self, label):
        if (len(self.ratios) == 0) | (self.ratio_displayed):
            return None
        i = np.where(self.labels == label)[0][0]
        y,x = np.where(self.wsi_mask == label)
        xmax = x.max()
        ymin = y.min()
        ratio = self.ratios[i]
        area = self.areas[i]
        text = f"{round(ratio, 1)}% , {round(area, 2)} mm2"
        self.display_text(text, xmax, ymin, background_color=(200,200,200))


    def reset(self, full_reset = False):
        if full_reset:
            self.image = self.raw_wsi.copy()
        else:
            self.image = self.wsi.copy()
        self.mask = self.wsi_mask 
        self.current_mag = self.wsi_mag
        self.current_resolution = self.wsi_resolution
        self.is_wsi = 1
        self.is_zoomed = 0
        self.zoom_triggered = 0
        self.ratio_displayed = False
    


def lum_contrast(bgrimg):
    hls = cv2.cvtColor(bgrimg, cv2.COLOR_BGR2HLS)
    h,l,s = cv2.split(hls)
    cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = cl.apply(l)
    hls = cv2.merge((h,l,s))
    return cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)

def hue_contrast(bgrimg, flip_hue_wheel = False):
    hls = cv2.cvtColor(bgrimg, cv2.COLOR_BGR2HLS)
    h,l,s = cv2.split(hls)
    if flip_hue_wheel:
        h = h.astype(float)-90
        h[np.where(h<0)] = h[np.where(h<0)]+180
        h = h.astype(np.uint8)
    cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    h = cl.apply(h)
    hls = cv2.merge((h,l,s))
    return cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)

def filter_blue_white(bgrimg):
    hls = cv2.cvtColor(bgrimg.copy(), cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(bgrimg.copy(), cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2]
    h, l, s = cv2.split(hls)
    # remove blue 
    test = np.logical_not((h>130) | (h<30) )
    hls[:,:,1][np.where(test)] = 255
    hls[:,:,2][np.where(test)] = 0

    # remove white
    test = np.logical_or(l>220,  s<10)
    hls[:,:,1][np.where(test)] = 255
    hls[:,:,2][np.where(test)] = 0

    return cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)

def filter_small_blobs(img, thresh = 0.1, resolution = np.array([2176.3, 2176]), keep_one = False ):
    """
    img: RGB image to filter
    filter out blobs with an area smaller than the threshold.
    threshold in mm2
    resolution in pix/mm in x,y direction
    keep_one: (bool) keep only the biggest blob
    """
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h,l,s = cv2.split(hls)
    #resolution_1_25 = resolution*mag_min/mag_max
    spacing = 1/resolution
    pix_area = spacing[0]*spacing[1] #mm2/pix
    nb_pixels = thresh/pix_area
    stats = cv2.connectedComponentsWithStats(s, connectivity=8)
    sizes = stats[2][1:,4]
    index = np.where(sizes > nb_pixels)[0]+1
    mask = np.zeros(stats[1].shape)
    j = 1
    for label in index:
        loc = np.where(stats[1]==label)
        if l[loc].mean()>100 and s[loc].mean()>20 and s[loc].std()>10:
            mask[loc] = j
            j+=1
    # if keep_one:
    #     blob_sizes = stats[2][:,4].squeeze()
    #     test = True
    #     a = 0
    #     while test:
    #         biggest_size = np.sort(blob_sizes)[-1-a] 
    #         i = np.where(blob_sizes == biggest_size)
    #         loc = np.where(stats[1]==i)
    #         if hls[:,:,1][loc].mean()>240:
    #             a+=1
    #         else:
    #             test = False
    #     mask[loc] = 1
    #     return mask.astype(np.uint8)
    
    return mask.astype(np.uint8)

def func(i, sizes, new_mask, mask, nb_pix):
    if sizes[i] > nb_pix:
        loc = np.where(mask==i)
        new_mask[loc] = 1
    return None
# def filter_small_holes(mask, thresh = 0.004, resolution = np.array([2176.3, 2176])):
#     """
#     mask : binary mask 
#     thresh : filter holes smaller than threshold in mm2
#     """
#     mask = np.logical_not(mask).astype(np.uint8)
#     stats = cv2.connectedComponentsWithStats(mask, connectivity=8)

#     new_mask = np.zeros_like(mask)
#     nb_pix = thresh * resolution[0] * resolution[1]
#     for i in range(1, stats[0]):
#         if stats[2][i,4] > nb_pix: # filter by size
#             loc = np.where(stats[1]==i)
#             new_mask[loc] = 1
#     new_mask = np.logical_not(new_mask).astype(np.uint8)
#     return new_mask


def separate_pink_orange(bgrimg):
    hls = cv2.cvtColor(bgrimg.copy(), cv2.COLOR_BGR2HLS)
    hue = hls[:,:,0]
    lum = hls[:,:,1]
    sat = hls[:,:,2]
    loc = np.where(lum<255)
    #90° rotation of the hue wheel
    hue = hue-90
    hue[np.where(hue<0)] = hue[np.where(hue<0)] +180

    im1 = bgrimg.copy()
    im2 = bgrimg.copy()
    #color filtering
    #pink
    test = np.logical_and(np.logical_not(hue<84) , lum<255)
    nb_orange = test.sum()
    im1[np.where(test)] = 255
    
    #orange
    test = np.logical_and(np.logical_not(hue>84) , lum<255)
    nb_pink = test.sum()
    im2[np.where(test)] = 255
    orange_ratio = nb_orange / (nb_pink + nb_orange)*100

    return im1, im2, orange_ratio


def get_screen_size():
    script = """
        tell application "Finder"
            set screenResolution to bounds of window of desktop
        end tell
        return screenResolution
    """
    output = os.popen(f"osascript -e '{script}'").read().strip().split(',')
    width, height = output[2:]
    return int(width), int(height)

def mouse_CB(event, x, y, flags, param):
    #img = param
    if param.is_wsi and not (param.wsi_mask is None):
        if event == cv2.EVENT_MOUSEMOVE:
            label = param.wsi_mask[y,x]
            if label != 0 and not param.ratio_displayed:
                param.display_ratio(label)
                param.ratio_displayed = True
            if label == 0 and param.ratio_displayed:
                param.ratio_displayed = False
                param.image = param.wsi.copy()

    if param.zoom_triggered:
        if event == cv2.EVENT_LBUTTONUP:
            param.zoom(x,y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if flags == 17: #shift pressed down
                param.zoom_rectangle(x,y)
            else:
                param.cancel_zoom()

    else:
        if event == cv2.EVENT_LBUTTONDOWN:
            if (flags == 17) & (param.is_zoomed):
                param.cancel_zoom()
            elif flags == 17:
                param.begin_zoom(x,y)
            else:
                pass #TO IMPLEMENT: clic on blob to select it
        elif (event == cv2.EVENT_MBUTTONDOWN) | (event == cv2.EVENT_RBUTTONDOWN):
            if param.is_zoomed:
                param.cancel_zoom()
            elif param.is_wsi:
                label = param.wsi_mask[y,x]
                param.open_label(label)


    pass

if __name__ == "__main__":

    # screen_width, screen_heigh = get_screen_size()
    # win_w = int(screen_width/4)
    # win_h = screen_heigh-screen_heigh//10

    path = 'data/19AA07204 04 HES.ndpi'#"/Volumes/G-DRIVE with Thunderbolt/Lames-plexus/LAMES PLEXUS/20AA00402 03 HES.ndpi" 
    
    Image = ImageNDPI(path)
    

    cv2.namedWindow('test-window', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('test-window', mouse_CB, Image)
    while True:

        cv2.imshow('test-window', Image.image)
        #cv2.resizeWindow('test-window', win_w, win_h)
        #cv2.moveWindow('test-window', int((screen_width - win_w)/2), 0)

        if (tt := cv2.waitKey(10) & int(0xFF)) != 255:
            if tt == ord('q'):
                sys.exit()
            if tt == ord('c'):
                Image.image = lum_contrast(Image.image)
            if tt == ord('w'):
                Image.process_wsi()
            if tt == ord('x'):
                if Image.wsi_mask is None:
                    Image.process_wsi()
                Image.process_all_labels()
            if tt == ord('r'):
                Image.reset()
            if tt == ord('f'):
                Image.reset(full_reset=True)
            if tt == ord('i'):
                if not Image.is_wsi and not Image.is_zoomed:
                    Image.image = Image.original_image
                elif not Image.is_wsi and Image.is_zoomed:
                    Image.image_prezoom = Image.original_image
                    Image.image = Image.original_image
                    Image.zoom(Image.x_final_zoom, Image.y_final_zoom)
            if tt == ord('p'):
                if not Image.is_wsi and not Image.is_zoomed:
                    Image.image = Image.pink
                elif not Image.is_wsi and Image.is_zoomed:
                    Image.image_prezoom = Image.pink
                    Image.image = Image.pink
                    Image.zoom(Image.x_final_zoom, Image.y_final_zoom)
            if tt == ord('o'):
                if not Image.is_wsi and not Image.is_zoomed:
                    Image.image = Image.orange
                elif not Image.is_wsi and Image.is_zoomed:
                    Image.image_prezoom = Image.orange
                    Image.image = Image.orange
                    Image.zoom(Image.x_final_zoom, Image.y_final_zoom)