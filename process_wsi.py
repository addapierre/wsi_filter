import os
import sys
import cv2
import numpy as np
import openslide
from utils import filter_background, filter_small_blobs, lum_contrast, filter_small_holes, \
                    separate_colors

from display import DisplayImage, mouse_CB


class ImageWSI(DisplayImage):
    
    wsi_formats = ("svs", "tif", "dcm", "ndpi", "vms", "vmu", "scn", "mrxs", "tiff", "svslide", "tif", "bif")

    def __init__(self, wsi_path, coloration = 'hes'):

        
        assert os.path.exists(wsi_path), f"file {wsi_path} not found"
        assert wsi_path.split('.')[-1] in self.wsi_formats, f'{wsi_path} is not a wsi file'
        coloration = coloration.lower()
        self.wsi_path = wsi_path
        self.coloration = coloration
        if self.coloration == 'hes':
            self.hue_limit_min = 174
            self.hue_limit_max = 210
        elif self.coloration == 'masson':
            self.hue_limit_min = 100
            self.hue_limit_max = 132
        else:
            raise Exception(f"invalid coloration, enter 'hes' or 'masson'. you entered {coloration}") 
        
        self.slide = openslide.OpenSlide(wsi_path)
        self.level_count = self.slide.level_count
        self.dimensions = np.array(self.slide.level_dimensions)
        self.level_dimensions = np.array(self.slide.level_dimensions)
        self.level_downsample = np.array(self.slide.level_downsamples)
        self.mmp_x_l0 = float(self.slide.properties['openslide.mpp-x'])/1e3 # mm per pixel in x direction
        self.mmp_y_l0 = float(self.slide.properties['openslide.mpp-y'])/1e3 # mm per pixel in y direction

        # display wsi at a resolution for which the image is lower than 5 Mpixel
        wsi_sizes = np.array([x[0]*x[1]/1e6 for x in self.level_dimensions])
        wsi_level = (self.level_count - np.where(wsi_sizes<5)[0]).max()
        self.current_level = self.level_count - wsi_level
        self.raw_wsi = np.array(self.slide.read_region((0,0), self.slide.level_count - wsi_level,self.slide.level_dimensions[-wsi_level]))[:,:,:3]
        self.raw_wsi = cv2.cvtColor(self.raw_wsi, cv2.COLOR_RGB2BGR)
        self.current_resolution = (1 / (self.mmp_x_l0 * self.level_downsample[-wsi_level]), 1 / (self.mmp_y_l0 * self.level_downsample[-wsi_level])) #resolution in pix/mm
        self.wsi_resolution = self.current_resolution
        super().__init__(self.raw_wsi.copy()) # self.current_image is wsi, raw_wsi remains unchanged
        self.wheel_window_init()

        #init values
        self.color = "both"
        self.is_wsi = True
        self.labels = None
        self.label = None
        self.mask = None
        self.ratios = []
        self.areas = []
        self.ratio_displayed = False
        self.wsi = None
        self.wsi_mask = np.zeros(self.raw_wsi.shape[:2])

    def process_wsi(self, threshold = 0.1):
        """
        removes background, small blobs and black blobs, creates a wsi_mask with a value/label per blob
        Params: 
            - img: BRG image to preprocess
            - threshold: in mm2, blobs smaller than threshold are removed from the image. default: 0.1 mm2
        Return: 
            - preprocessed image
            - mask with only blobs larger than threshold. blob values in mask correspond to their label
        """

        if not self.is_wsi:
            return None
        if self.coloration == 'masson':
            self.current_image, _ = filter_background(self.current_image) 
        elif self.coloration == 'hes':
            self.current_image, _ = filter_background(self.current_image, filter_hue_start = 30,  filter_hue_end = 130) 

        self.wsi_mask = filter_small_blobs(self.current_image.copy(), thresh = threshold, resolution = self.current_resolution, coloration = self.coloration)
        #mask the image with 255 values
        tt = np.dstack((np.clip(self.wsi_mask.copy(), 0, 1),)*3)
        self.current_image = (self.current_image * tt) + (tt-1) # (tt-1) because (0-1 = 255)
        self.labels = np.unique(self.wsi_mask)[1:]
        self.wsi = self.current_image.copy()


    def open_label(self, label, max_size = 70):
        """
        opens a blob in a higher resolution to process it and/or to vizalize it. 
        the label identifies the blob.
        max size is the maximum size of the generated image in Megapixels
        """
        if (label == 0) or not self.is_wsi:
            return None
        y,x = np.where(self.wsi_mask == label)
        xmin = x.min() ; ymin = y.min()
        xmax = x.max() ; ymax = y.max()
        

        #select new resolution based on target image size
        init_size = (xmax - xmin)*(ymax-ymin)/1e6 # blob image subsection in Megapixel
        target_sizes = init_size * np.square(self.level_downsample[::-1]) # wsi is last level so level_downsample[::-1] == level upsample
        max_size = 70   # target image must be less than 70Mpix
        target_level = np.where(target_sizes<max_size)[0].min()

        #read cropped image
        xmin_l0 = int(xmin * self.level_dimensions[0][0]/self.level_dimensions[self.current_level][0])
        ymin_l0 = int(ymin * self.level_dimensions[0][1]/self.level_dimensions[self.current_level][1])

        resolution_ratio = self.level_dimensions[target_level]/self.level_dimensions[self.current_level]
        width = int((xmax - xmin ) * resolution_ratio[0])
        height = int((ymax - ymin ) * resolution_ratio[1])
        self.current_image = np.array(self.slide.read_region((xmin_l0,ymin_l0), target_level, (width, height)))[:,:,:3]
        self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR)

        #new current resolution in pix/mm2
        self.current_resolution = (self.current_resolution[0]*resolution_ratio[0], self.current_resolution[1]*resolution_ratio[1])

        # mask label img with cropped wsi mask
        self.mask = self.wsi_mask[ymin:ymax, xmin:xmax].copy()
        test = self.mask!=label
        self.mask = np.logical_not(test).astype(np.uint8)
        self.mask = cv2.resize(self.mask, (self.current_image.shape[1], self.current_image.shape[0]))

        #complete mask
        self.mask = filter_small_holes(self.mask, resolution=self.current_resolution) # mask = 0 on background and 1 on blob
        mask_3ch = np.dstack((self.mask,)*3)

        self.current_image = self.current_image * mask_3ch + (mask_3ch-1)

        if self.coloration == 'masson':
            self.current_image, self.mask = filter_background(self.current_image) 
        elif self.coloration == 'hes':
            self.current_image, self.mask = filter_background(self.current_image, filter_hue_start = 30,  filter_hue_end = 130) 
        self.current_image = self.current_image + np.dstack((self.mask-1,)*3)  #replace 0 with 255

    def process_label(self, img = None, keep_images = True):
        """returns the blob's area in mm2 and the ratio of fibrosis orange/(orange+purple). 
        can create separated self.im_color1 and self.im_color2 image if keep_image = True"""
        nb_pixel = self.mask.sum()
        
        pix_area = 1/(self.current_resolution[0]*self.current_resolution[1]) #mm2/pix
        
        area = nb_pixel*pix_area

        if img is None:
            if self.color == "both":
                img = self.current_image
            else:
                img = self.original_image
        
        ratio1, ratio2, self.im_color1, self.im_color2 = \
            separate_colors(img, self.mask, min_hue = self.hue_limit_min, max_hue = self.hue_limit_max, keep_images=keep_images)
        ratio = ratio1

        if keep_images:
            self.original_image = img

        return area, ratio

    def change_limit(self, lower_bound :bool, increment : bool):
        """change color bound for color separation. 
        makes sure bounds are not equal, that they are between 0 and 179, and that min an max bounds don't switch.
        """
        self.update_wheel(mean_fibrosis=-1)
        if lower_bound:
            if increment:
                self.hue_limit_min += 1
                if self.hue_limit_min == self.hue_limit_max:
                    self.hue_limit_min += 1
            else:
                self.hue_limit_min -= 1
                if self.hue_limit_min == self.hue_limit_max:
                    self.hue_limit_min -= 1
        else:
            if increment:
                self.hue_limit_max += 1
                if self.hue_limit_min == self.hue_limit_max:
                    self.hue_limit_max += 1
            else:
                self.hue_limit_max -= 1
                if self.hue_limit_min == self.hue_limit_max:
                    self.hue_limit_max -= 1

        if self.hue_limit_min < 0:
            self.hue_limit_min, self.hue_limit_max = self.hue_limit_max, self.hue_limit_min
        
        if not self.is_wsi:
            self.display_blob(self.current_label)
            if self.color == "color_1":
                
                if self.is_zoomed:
                    self.image_prezoom = self.im_color1.copy()
                    self.current_image = self.im_color1.copy()
                    self.zoom(self.x_final_zoom, self.y_final_zoom)
                else:
                    self.current_image = self.im_color1.copy()
            if self.color == "color_2":
                if self.is_zoomed:
                    self.image_prezoom = self.im_color2.copy()
                    self.current_image = self.im_color2.copy()
                    self.zoom(self.x_final_zoom, self.y_final_zoom)
                else:
                    self.current_image = self.im_color2.copy()

        self.update_wheel()


    def display_blob(self, label):
        self.current_label = label
        self.open_label(label)
        self.is_wsi = 0
        area, ratio = self.process_label(keep_images=True)
        self.update_wheel(sample_fibrosis=ratio)


    def process_all_labels(self):
        if self.labels is None:
            return None
        self.ratios = []
        self.areas = []
        for label in self.labels:
            self.open_label(label)
            area, ratio = self.process_label(keep_images=False)
            # print(f"area : {round(area, 3)} mm2 |  ratio :  {round(ratio, 2)}%")
            self.ratios.append(ratio)
            self.areas.append(area)
            #reinit
            self.reset()
        self.ratios = np.array(self.ratios)
        self.areas = np.array(self.areas)
        full_area = self.areas.sum()
        self.mean_ratio = (self.ratios*self.areas/full_area).sum()
        self.update_wheel(mean_fibrosis = self.mean_ratio)
        
        # display ratio on lower left corner of the image (CHANGE TO BE DISPLAYED ON A NEW WINDOW)
        #self.display_text(str(round(self.mean_ratio, 2))+"%", 0, self.current_image.shape[0], font_multiplicator=0.005, thickness_multiplicator=0.004)
        self.wsi = self.current_image.copy()

    # writing the label on each blob is too slow, as it requires generating a new wsi image for each blob (that's how cv2.putText works)
    # def display_label(self, label):
    #     """
    #     displays blob label on the wsi 
    #     """

    #     if self.labels is None or not self.is_wsi:
    #         return None 
    #     self.label = label
    #     y,x = np.where(self.wsi_mask == label)
        
    #     xmin = x.min() ; ymin = y.min()
    #     xmax = x.max() ; ymax = y.max()
    #     x = int((xmax+xmin)/2)-40
    #     y = int((ymax+ymin)/2)+40
        
    #     self.display_text(text=str(label), x=x, y=y, background_color= (100,100,100))
    #     self.wsi = self.current_image.copy()


    def display_ratio(self, label):
        """
        When the arrow hovers over the blob, display its ratio and area next to it.
        """
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
        self.ratio_displayed = True


    def reset(self, full_reset = False):
        if full_reset:
            self.current_image = self.raw_wsi.copy()
            
        else:
            self.current_image = self.wsi.copy()
        
        self.color = "both"
        self.mask = self.wsi_mask 
        self.current_resolution = self.wsi_resolution
        self.is_wsi = 1
        self.is_zoomed = 0
        self.zoom_triggered = 0
        self.ratio_displayed = False
        self.label = None


    def wheel_window_init(self):
        """generates an image with the color wheel and the fibrosis %. to be displayed in another window"""
        wheel = cv2.imread('images/wheel.png')
        margin = 30
        wheel_size = wheel.shape[0]
        self.wheel_size = wheel_size
        self.wheel_window = np.ones((int(wheel_size*2.1), wheel_size+2*margin, 3), dtype = np.uint8)*255
        self.wheel_window[margin:wheel_size+margin, margin:wheel_size+margin, :] = wheel.copy()
        width = self.wheel_window.shape[1]

        cv2.ellipse(self.wheel_window, (wheel_size//2+margin,)*2, (210,)*2, -90, (self.hue_limit_min)*2, (self.hue_limit_max)*2, (0,)*3, 15)

        cv2.putText(self.wheel_window, str(self.hue_limit_min%180), (290, wheel_size//2-10), fontFace = 2, fontScale= 2, color = (20,20,20), thickness = 4)
        cv2.putText(self.wheel_window, str(self.hue_limit_max%180), (290, wheel_size//2+90), fontFace = 2, fontScale= 2, color = (20,20,20), thickness = 4)

        cv2.putText(self.wheel_window, '-', (220, wheel_size//2-10), fontFace = 2, fontScale= 2, color = (20,20,20), thickness = 4)
        cv2.putText(self.wheel_window, '+', (430, wheel_size//2-10), fontFace = 2, fontScale= 2, color = (20,20,20), thickness = 4)
        cv2.putText(self.wheel_window, '-', (220, wheel_size//2+90), fontFace = 2, fontScale= 2, color = (20,20,20), thickness = 4)
        cv2.putText(self.wheel_window, '+', (430, wheel_size//2+90), fontFace = 2, fontScale= 2, color = (20,20,20), thickness = 4)

        font = 3
        div1 = wheel_size+2*margin+30
        cv2.line(self.wheel_window, (0, div1), (width, div1), (0,0,0), 2)
        cv2.line(self.wheel_window, (0, div1+10), (width, div1+10), (0,0,0), 2)

        cv2.putText(self.wheel_window, "MEAN FIBROSIS:", (2*margin, div1 + 120),fontFace = font, fontScale= 2, color = (0,0,0), thickness = 2)
        cv2.putText(self.wheel_window, "--- %", ((width)//3, div1 + 230), fontFace = 2, fontScale= 2, color = (20,20,20), thickness = 4)
        

        div2 = div1 + 330
        cv2.line(self.wheel_window, (0, div2), (width, div2), (0,0,0), 2)
        cv2.line(self.wheel_window, (0, div2+10), (width, div2+10), (0,0,0), 2)

        cv2.putText(self.wheel_window, "SAMPLE FIBROSIS:", (2*margin, div2 + 120),fontFace = font, fontScale= 2, color = (0,0,0), thickness = 2)
        cv2.putText(self.wheel_window, "--- %", ((width)//3, div2 + 230), fontFace = 2, fontScale= 2, color = (20,20,20), thickness = 4)


    def update_wheel(self, mean_fibrosis = None, sample_fibrosis = None):

        margin = 30
        width = self.wheel_window.shape[1]
        div1 = self.wheel_size+90

        if mean_fibrosis is not None:
            cv2.rectangle(self.wheel_window, (0,div1 + 130), (width, div1+300), (255,)*3, -1)
            if mean_fibrosis == -1:
                cv2.putText(self.wheel_window, "--- %", ((width)//3, div1 + 230), fontFace = 2, fontScale= 2, color = (20,20,20), thickness = 4)
            else:
                cv2.putText(self.wheel_window, f"{mean_fibrosis:.2f} %", ((width)//3, div1 + 230), fontFace = 2, fontScale= 2, color = (20,20,20), thickness = 4)

        if sample_fibrosis is not None:
            div2 = div1 + 330
            cv2.rectangle(self.wheel_window, (0,div2 + 130), (width, div2+300), (255,)*3, -1)
            if sample_fibrosis == -1:
                cv2.putText(self.wheel_window, "--- %", ((width)//3, div2 + 230), fontFace = 2, fontScale= 2, color = (20,20,20), thickness = 4)
            else:
                cv2.putText(self.wheel_window, f"{sample_fibrosis:.2f} %", ((width)//3, div2 + 230), fontFace = 2, fontScale= 2, color = (20,20,20), thickness = 4)

        else: # update wheel
            cv2.ellipse(self.wheel_window, (self.wheel_size//2+margin,)*2, (210,)*2, 0, 0, 370, (255,)*3, 17)
            cv2.ellipse(self.wheel_window, (self.wheel_size//2+margin,)*2, (210,)*2, -90, (self.hue_limit_min)*2, (self.hue_limit_max)*2, (0,)*3, 15)
            
            cv2.rectangle(self.wheel_window, (280, self.wheel_size//3), (420, self.wheel_size-130), (255,)*3, -1)
            cv2.putText(self.wheel_window, str(self.hue_limit_min%180), (290, self.wheel_size//2-10), fontFace = 2, fontScale= 2, color = (20,20,20), thickness = 4)
            cv2.putText(self.wheel_window, str(self.hue_limit_max%180), (290, self.wheel_size//2+90), fontFace = 2, fontScale= 2, color = (20,20,20), thickness = 4)



if __name__ == "__main__":
    pass
