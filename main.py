import cv2
import sys
import os
import argparse
import numpy as np
from process_wsi import ImageWSI
from display import mouse_CB, color_wheel_CB
from utils import lum_contrast

parser = argparse.ArgumentParser(
                    prog='wsiFilter',
                    description='takes a whole slide image, identifies and isolates each sample on the slide, and performs a colorimetric analysis of each sample',
                    epilog='Text at the bottom of help')

parser.add_argument(
    '-c', '--coloration', 
    dest = 'coloration',
    default='hes',
    type=str,
    )
parser.add_argument('-f', '--filename',
                    dest='filename',
                    type=str)
# parser.add_argument('-h', '--help', dest='help',
#                     action='store_true')

def wsi_filter(filename, coloration):
    Image = ImageWSI(filename, coloration)
    cv2.namedWindow('wsi-window', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('wsi-window', mouse_CB, Image)

    cv2.namedWindow('wheel-window', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('wheel-window', color_wheel_CB, Image)
    
    while True:
        cv2.imshow('wsi-window', Image.current_image)
        cv2.imshow('wheel-window', Image.wheel_window) 
        if (tt := cv2.waitKey(10) & int(0xFF)) != 255:
            """Q, C, W, X, R, F, I, O, P, D, A, Z"""
            if tt == ord('q'):
                sys.exit()
            if tt == ord('c'):
                Image.current_image = lum_contrast(Image.current_image)
            if tt == ord('w'):
                # process wsi
                Image.process_wsi()
            if tt == ord('x'):
                # process wsi and all labels, displays ratio
                if Image.labels is None:
                    Image.process_wsi()
                    cv2.imshow('wsi-window', Image.current_image)
                    cv2.waitKey(10)
                Image.process_all_labels()
            if tt == ord('r'):
                Image.reset()
                Image.update_wheel(sample_fibrosis=-1)
            if tt == ord('f'):
                Image.reset(full_reset=True)
                Image.update_wheel(sample_fibrosis=-1)

            if tt == ord('a'):
                Image.change_limit(increment=False)
                if not Image.is_wsi:
                    if not Image.is_zoomed:
                        Image.current_image = Image.original_image.copy()
                        Image.process_label(keep_images = True)
                    if Image.is_zoomed:
                        Image.image_prezoom = Image.original_image.copy()
                        Image.current_image = Image.original_image.copy()
                        Image.process_label(keep_images = True)
                        Image.zoom(Image.x_final_zoom, Image.y_final_zoom)
            if tt == ord('z'):
                Image.change_limit(increment=True)
                if not Image.is_wsi:
                    if not Image.is_zoomed:
                        Image.current_image = Image.original_image
                        Image.process_label(keep_images = True)
                    if Image.is_zoomed:
                        Image.image_prezoom = Image.original_image
                        Image.current_image = Image.original_image
                        Image.process_label(keep_images = True)
                        Image.zoom(Image.x_final_zoom, Image.y_final_zoom)

            if tt == ord('z'):
                Image.change_limit(increment=True)

            if tt == ord('i'):
                # if img is a label and one color is selected, return to original image
                Image.color = "both" 
                if not Image.is_wsi and not Image.is_zoomed:
                    Image.current_image = Image.original_image
                elif not Image.is_wsi and Image.is_zoomed:
                    Image.image_prezoom = Image.original_image
                    Image.current_image = Image.original_image
                    Image.zoom(Image.x_final_zoom, Image.y_final_zoom)
            if tt == ord('o'):
                # if img is a label, keep only on of the 2 colors
                
                if not Image.is_wsi:
                    Image.color = "color_1" 
                    if not Image.is_zoomed:
                        Image.current_image = Image.im_color1.copy()
                    else:  # image is zoomed
                        Image.image_prezoom = Image.im_color1.copy()
                        Image.current_image = Image.im_color1.copy()
                        Image.zoom(Image.x_final_zoom, Image.y_final_zoom)
            if tt == ord('p'):
                # if img is a label, keep only on of the 2 colors
                if not Image.is_wsi:
                    Image.color = "color_2" 
                    if not Image.is_zoomed:
                        Image.current_image = Image.im_color2.copy()
                    else: # image is zoomed
                        Image.image_prezoom = Image.im_color2.copy()
                        Image.current_image = Image.im_color2.copy()
                        Image.zoom(Image.x_final_zoom, Image.y_final_zoom)

            if tt == ord('d'): 
                #process wsi and then each label, but diplays labels as they're being processed
                if Image.labels is None:
                    Image.process_wsi()
                    cv2.imshow('wsi-window', Image.current_image)
                    cv2.waitKey(1)
                
                Image.ratios = []
                Image.areas = []
                for label in Image.labels:
                    Image.open_label(label, mag = 10)
                    area, ratio = Image.process_label(keep_images=False)
                    Image.ratios.append(ratio)
                    Image.areas.append(area)
                    #reinit
                    Image.reset()
                    Image.display_label(label)
                    cv2.imshow('wsi-window', Image.current_image)
                    cv2.waitKey(1)
                Image.ratios = np.array(Image.ratios)
                Image.areas = np.array(Image.areas)
                full_area = Image.areas.sum()
                ratio = (Image.ratios*Image.areas/full_area).sum()
                Image.display_text(str(round(ratio, 2))+"%", 0, Image.current_image.shape[0], font_multiplicator=0.005, thickness_multiplicator=0.004)
                Image.wsi = Image.current_image.copy()

            





if __name__=="__main__":
    args = parser.parse_args()
    #test args
    coloration = args.coloration.lower()
    assert coloration in ['hes', 'masson'], f'Error: coloration should be either hes or masson, you entered {coloration}'
    filename = args.filename.lower()
    assert os.path.exists(filename), f"Error: file not found at path {filename}"
    assert filename.split('.')[-1] == 'ndpi', f"Error: file should be a .ndpi file, you entered {filename}"

    # if args.help:
    #     help = """ help:\n
    #     q: quit program\n
    #     w: process whole slide image (remove background and identify samples)\n
    #     x: process whole slide image and calculate ratio for each sample individually\n
    #     d: same as x but display label names\n
    #     c: increase contrast\n
    #     **click on label to open magnified view**
    #     o: *label view* select 1st color
    #     p: *label view* select 2nd color
    #     i: *label view* view all colors
    #     """
    #     print(help)
    wsi_filter(filename, coloration)


