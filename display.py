import cv2
import numpy as np



class DisplayImage():

    def __init__(self, image) -> None:
        self.current_image = image.copy() # currently displayed image
        self.image_prezoom = self.current_image.copy()
        self.height = image.shape[0]
        self.width = image.shape[1]

        self.zoom_triggered = 0
        self.is_zoomed = 0

        self.x_init_zoom = None
        self.y_init_zoom = None
        self.resize_ratio = None
 
    def begin_zoom(self, x, y): 
        self.image_prezoom = self.current_image.copy()
        self.zoom_triggered = 1
        self.x_init_zoom = x
        self.y_init_zoom = y

    def zoom_rectangle(self, x, y):
        self.current_image = self.image_prezoom.copy()
        cv2.rectangle(self.current_image, (self.x_init_zoom, self.y_init_zoom), (x,y), (255,0,0), 2)

    def cancel_zoom(self):
        self.zoom_triggered = 0
        self.is_zoomed = 0
        self.current_image = self.image_prezoom.copy()

    def zoom(self, x, y):
        #try:
            self.current_image = self.image_prezoom.copy()
            self.zoom_triggered = 0
            self.is_zoomed = 1
            self.x_final_zoom = x
            self.y_final_zoom = y

            x_zoom_1 = self.x_init_zoom ; x_zoom_2 = self.x_final_zoom
            y_zoom_1 = self.y_init_zoom ; y_zoom_2 = self.y_final_zoom

            # cropping
            self.xlist = np.sort(np.array([x_zoom_1, x_zoom_2]))
            self.ylist = np.sort(np.array([y_zoom_1, y_zoom_2]))
            self.current_image = self.current_image[self.ylist[0] : self.ylist[1], self.xlist[0] : self.xlist[1]]

            # resize so that the area covered by the image is the same
            new_w = self.xlist[1] - self.xlist[0]
            new_h = self.ylist[1] - self.ylist[0]
            self.resize_ratio = np.sqrt(self.height*self.width/(new_h*new_w))
            try:
                new_size = np.array([int(new_w * self.resize_ratio), int(new_h * self.resize_ratio)])
            except :
                self.cancel_zoom()
                return None
            self.current_image = cv2.resize(self.current_image, new_size, interpolation = cv2.INTER_LANCZOS4)
        # except: 
        #     self.cancel_zoom()

    def display_text(self, text, x, y, color = (255,0,0), font_multiplicator = 1e-3, thickness_multiplicator = 1e-3, background_color = None):
        # Warning: this function changes self.current_image, make sure you made a copy before calling it
        font_scale = min(self.width, self.height) * font_multiplicator
        font_thickness = int(np.ceil(min(self.width, self.height) * thickness_multiplicator))
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, font_thickness)
        text_w, text_h = text_size
        if x+text_w > self.current_image.shape[1]:
            x = x-text_w
        if y - text_h < 0:
            y = y+text_h
        if isinstance(background_color, tuple):
            cv2.rectangle(self.current_image, (x, y-text_h-np.ceil(text_h/5).astype(int)), (x+text_w, y), background_color, -1)

        cv2.putText(self.current_image, text, (x, y), cv2.FONT_HERSHEY_DUPLEX,\
                    fontScale = font_scale, color = color, thickness = font_thickness, lineType = cv2.LINE_AA)
        

        
def mouse_CB(event, x, y, flags, param):
    """opencv mouse callback"""

    if param.zoom_triggered:
        if event == cv2.EVENT_LBUTTONUP:
            param.zoom(x,y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if flags == 17: #shift pressed down
                param.zoom_rectangle(x,y)
            else:
                param.cancel_zoom()

    else:

        if (event == cv2.EVENT_MOUSEMOVE) and (param.labels is not None) and (param.is_wsi):
            label = param.wsi_mask[y,x]
            if label != 0 and not param.ratio_displayed:
                param.display_ratio(label)
                param.ratio_displayed = True
            if label == 0 and param.ratio_displayed:
                param.ratio_displayed = False
                param.current_image = param.wsi.copy()

        if event == cv2.EVENT_LBUTTONDOWN:
            if (flags == 17) & (param.is_zoomed):
                param.cancel_zoom()
            elif flags == 17:
                param.begin_zoom(x,y)
            else:
                pixel = param.current_image[y,x]
                print('bgr : ',pixel)
                hls_pixel = cv2.cvtColor(pixel[None, None, :], cv2.COLOR_BGR2HLS).squeeze()
                print('hls : ',hls_pixel)
                hsv_pixel = cv2.cvtColor(pixel[None, None, :], cv2.COLOR_BGR2HSV).squeeze()
                print('hsv : ',hsv_pixel)
                # if param.is_zoomed:
                #     param.cancel_zoom()
                # elif param.is_wsi:
                #     label = param.wsi_mask[y,x]
                #     param.open_label(label)
        elif (event == cv2.EVENT_MBUTTONDOWN) | (event == cv2.EVENT_RBUTTONDOWN):

            if param.is_zoomed:
                param.cancel_zoom()
            elif param.is_wsi:
                if param.ratio_displayed:
                    param.ratio_displayed = False
                label = param.wsi_mask[y,x]
                param.display_blob(label)


if __name__ == "__main__":
    print('ok')