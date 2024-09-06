import os
import sys
import cv2
import re
import time
import numpy as np
import subprocess
from subprocess import Popen, PIPE, STDOUT
from utils import filter_background, filter_small_blobs, lum_contrast, filter_small_holes, \
                    separate_colors

from display import DisplayImage, mouse_CB

class ImageNDPI(DisplayImage):
    def __init__(self, ndpi_path, coloration = 'hes'):

        assert os.path.exists(ndpi_path), f"ndpi file {ndpi_path} not found"
        assert ndpi_path.split('.')[-1] == 'ndpi', f'{ndpi_path} is not a .ndpi file'

        self.ndpi_path = ndpi_path
        self.coloration = coloration
        if self.coloration == 'hes':
            self.color_limit = 84
        elif self.coloration == 'masson':
            self.color_limit = 132#129
        self.mag_list = np.array([40, 20, 10, 5, 2.5, 1.25])
        self.mag_max, self.max_resolution = self.get_header_info()

        if self.mag_max == 20:
            self.mag_list = self.mag_list[1:]


        # get magnification closest to 1
        self.current_mag = self.wsi_mag = 1.25
        self.current_resolution = self.wsi_resolution = self.max_resolution * self.current_mag / self.mag_max
        self.raw_wsi = self.ndpi_split(self.current_mag, 0, 0, 1, 1, normalized=True)
        super().__init__(self.raw_wsi.copy()) # current_image = wsi, raw_wsi remains unchanged
        self.wsi_width = self.raw_wsi.shape[1] ; self.wsi_height = self.raw_wsi.shape[0]

        self.blob_thresh = 0.1 # everything smaller in mm2 gets filtered
        self.is_wsi = 1
        self.labels = None
        self.label = None
        self.wsi_mask = np.zeros(self.raw_wsi.shape[:2])
        self.mask = None
        self.ratios = []
        self.areas = []
        self.ratio_displayed = False
        self.wsi = None

    def process_wsi(self, threshold = 0.1):
        """
        removes background, small blobs and black blobs, creates a wsi_mask with a value/label per blob
        Params: 
            - img: RGB image to preprocess
            - threshold: in mm2, blobs smaller than threshold are removed from the image. default: 0.2 mm2
        Return: 
            - preprocessed image
            - mask with only blobs larger than threshold. blob values in mask correspond to their label
        """
        if not self.is_wsi:
            return None

        if threshold is None:
            threshold = self.blob_thresh

        if self.coloration=='masson':
            self.current_image, _ = filter_background(self.current_image, filter_blue=False) 
        else:
            self.current_image, _ = filter_background(self.current_image, filter_blue=True) 

        self.wsi_mask = filter_small_blobs(self.current_image.copy(), thresh = threshold, resolution = self.current_resolution, coloration = self.coloration)

        #mask the image with 255 values
        tt = np.dstack((np.clip(self.wsi_mask.copy(), 0, 1),)*3)
        self.current_image = (self.current_image * tt) + (tt-1) 

        self.labels = np.unique(self.wsi_mask)[1:]
        # self.display_labels() # too slow
        self.wsi = self.current_image.copy()


    def open_label(self, label, mag = None):
        if (label == 0) or not self.is_wsi:
            return None
        #print(f"opening label {label}...")
        y,x = np.where(self.wsi_mask == label)
        xmin = x.min() ; ymin = y.min()
        xmax = x.max() ; ymax = y.max()
        
        #select new mag and new resolution based on target image size
        if mag is None:
            size_init = (xmax - xmin)*(ymax-ymin)/1e6 # target image size in Megapixel
            target_size = size_init * np.square(self.mag_list/self.wsi_mag)
            self.current_mag = self.mag_list[np.where(target_size<70)].max() # target image must be less than 70Mpix
        else:
            self.current_mag = mag
        self.current_resolution = self.max_resolution * self.current_mag / self.mag_max

        #read cropped image
        
        self.current_image = self.ndpi_split(self.current_mag, xmin, ymin, xmax, ymax, normalized=False )
        
        # mask label img with cropped wsi mask
        self.mask = self.wsi_mask[ymin:ymax, xmin:xmax].copy()
        test = self.mask!=label
        self.mask[test] = 0
        
        self.mask = cv2.resize(self.mask, (self.current_image.shape[1], self.current_image.shape[0]))

        #complete mask
        
        self.mask = filter_small_holes(self.mask, resolution=self.current_resolution) # mask = 0 on background and 1 on blob
        mask_3ch = np.dstack((self.mask,)*3)
        self.current_image = self.current_image * mask_3ch + (mask_3ch-1)
        self.current_image, self.mask = filter_background(self.current_image, filter_blue=(self.coloration=='hes'))
        self.current_image = self.current_image + np.dstack((self.mask-1,)*3) #replace 0 with 255

    def process_label(self, img = None, keep_images = True):
        "returns the blob's area in mm2 and the ratio of fibrosis orange/(orange+purple). can create separated self.im_color1 and self.im_color2 image if keep_image = True"
        # if self.is_wsi:
        #     return None
        # calculate blob area
        nb_pixel = self.mask.sum()
        pix_area = 1/(self.current_resolution[0]*self.current_resolution[1]) #mm2/pix
        area = nb_pixel*pix_area

        if img is None:
            img = self.current_image

        # orange ratio
        if self.coloration == 'hes':
            
            ratio1, ratio2, self.im_color1, self.im_color2 = separate_colors(img, self.mask,0, self.color_limit, self.color_limit+1, 180, rotate_wheel = True,  keep_images=keep_images)
            ratio = ratio2
        elif self.coloration == 'masson':
            ratio1, ratio2, self.im_color1, self.im_color2 = separate_colors(img, self.mask,100, self.color_limit, self.color_limit+1, 180, rotate_wheel = False,  keep_images=keep_images)
            ratio = ratio2
        if keep_images:
            self.original_image = img
            self.display_text(str(round(ratio, 2))+"%", 0, self.current_image.shape[0], font_multiplicator=0.005, thickness_multiplicator=0.008)#, background_color=(255,255,255))

        return area, ratio

    def change_limit(self, increment : bool):
        if increment:
            self.color_limit += 1
        else:
            self.color_limit -= 1


    def display_blob(self, label):
        self.open_label(label)
        self.is_wsi = 0
        area, ratio = self.process_label(keep_images=True)


    def process_all_labels(self):
        if self.labels is None:
            return None
        self.ratios = []
        self.areas = []
        for label in self.labels:
            self.open_label(label, mag = 10)
            area, ratio = self.process_label(keep_images=False)
            # print(f"area : {round(area, 3)} mm2 |  ratio :  {round(ratio, 2)}%")
            self.ratios.append(ratio)
            self.areas.append(area)
            #reinit
            self.reset()
        self.ratios = np.array(self.ratios)
        self.areas = np.array(self.areas)
        full_area = self.areas.sum()
        ratio = (self.ratios*self.areas/full_area).sum()
        self.display_text(str(round(ratio, 2))+"%", 0, self.current_image.shape[0], font_multiplicator=0.005, thickness_multiplicator=0.004)
        self.wsi = self.current_image.copy()

    def display_label(self, label):
        """
        displays blob label on the wsi 
        """

        if self.labels is None or not self.is_wsi:
            return None 
        self.label = label
        y,x = np.where(self.wsi_mask == label)
        
        xmin = x.min() ; ymin = y.min()
        xmax = x.max() ; ymax = y.max()
        x = int((xmax+xmin)/2)-40
        y = int((ymax+ymin)/2)+40
        
        self.display_text(text=str(label), x=x, y=y, background_color= (100,100,100))
        self.wsi = self.current_image.copy()


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
            self.current_image = self.raw_wsi.copy()
            
        else:
            self.current_image = self.wsi.copy()
        self.mask = self.wsi_mask 
        self.current_mag = self.wsi_mag
        self.current_resolution = self.wsi_resolution
        self.is_wsi = 1
        self.is_zoomed = 0
        self.zoom_triggered = 0
        self.ratio_displayed = False
        self.label = None

    def get_header_info(self):
        """
        reads the ndpi header, returns max magnification and max resolution in pix/mm
        """
        cmd = ['tifftopnm', '-headerdump', self.ndpi_path]
        process = Popen(cmd, stdout=PIPE, stderr=STDOUT)
        header = process.communicate()[0].decode('ISO-8859-1')

        # Get Magnifications
        pattern = r'Tag 65421: (\d+)'
        _ =re.findall(pattern, header)
        mag_max = int(_[0])

        #Get resolution at mag_max
        pattern = r'Resolution: (\d+), (\d+) pixels/cm'
        numbersStrings = re.findall(pattern, header)
        max_resolution = np.array([int(n) for tup in numbersStrings for n in tup]) / 10 # in pix/mm for maximum resolution
        return mag_max, max_resolution

    def ndpi_split(self, mag, xmin, ymin, xmax, ymax, normalized = True):
        """ 
        coordinates are tranformed in this method form xyxy into xywh
        if normalized = True, coordinates are already normalized when passed to this method"""
        if not normalized:
            xmin = xmin / self.wsi_width  ; xmax = xmax / self.wsi_width
            ymin = ymin / self.wsi_height ; ymax = ymax / self.wsi_height
        if round(mag) == mag:
            mag = str(round(mag))
        else:
            mag = str(round(mag,2))
        cmd = ['ndpisplit', f'-x{mag}' ,f'-e{xmin},{ymin},{xmax-xmin},{ymax-ymin}', self.ndpi_path]
        #print(cmd)
        subprocess.call(cmd)
        path = self.ndpi_path.replace('.ndpi', f'_x{mag}_z0_1.tif')
        im = cv2.imread(path) 
        os.remove(path)

        return im







if __name__ == "__main__":
    pass
