# -*- coding: utf-8 -*-
"""
--------------- LANE DETECTION ---------------
"""

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import scipy.misc

# --- Importing our road images

data_dir = "test_images/"
image_names = os.listdir(data_dir)
# image_names.remove(".DS_Store")
image_names = list(map(lambda name: data_dir + name, image_names))

# --- Function to show a list of images

def showImageList(img_list, cols=4, fig_size=(15,15), img_labels=image_names, show_ticks=True):
    img_count = len(img_list)
    rows = img_count / cols
    cmap = None
    plt.figure(figsize=fig_size)
    
    for i in range(img_count):
        img_name = img_labels[i]
        plt.subplot(rows, cols, i+1)
        img = img_list[i]
        if len(img.shape) < 3:
            cmap = "gray"
        if not show_ticks:
            plt.xticks([])
            plt.yticks([])
        plt.title(img_name[len(data_dir):])
        plt.imshow(img, cmap=cmap)
        
    plt.tight_layout()
    plt.show()
    
# --- Plotting our road images
    
images = list(map(lambda img_name: mpimg.imread(img_name), image_names))

for img in images:
    if img.shape != (960,540) and img.shape != (1280,720):
        scipy.misc.imresize(img,(960,540)) # for later purposes
        
print("Total image count: ", len(images))
# showImageList(images)

""" NB #1: our images dimensions are 540x960 and 720x1280 (3 stands for RGB channels) """

""" Pipeline:
    - converting original images to HSL
    - isolate white and yellow from HSL 
    - convert to grayscale
    - apply Gaussian Blur for smooth edges
    - apply Canny Edge Detection
    - discard uninteresting lines outside the road
    - Hough transform to find lanes equations
    - separate left & right lanes
    - create smooth guiding lines
    """
    
# --- converting our road images to HSL images

def convertToHSL(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

hsl_images = list(map(lambda img: convertToHSL(img), images))

# --- plotting HSL images _DOC_RP_

"""
img_count = len(hsl_images)
interleaved_hsl = list(zip(images, hsl_images))
k = 0
for hsl in interleaved_hsl:
    img_name = image_names[k]
    showImageList(hsl, cols=2, fig_size=(15,15), img_labels=[img_name,img_name])
    k += 1
"""

# --- isolating white and yellow in our HSL images (see report for more info on HSL equivalences) _DOC_RP_

def HSLWhiteIsolate(img):
    # The image is already in HSL color space
    # Also, OpenCV encodes data in HLS format instead
    # Lower threshold equivalent in pure HSL is (30,45,15) (dark brown)
    low_threshold = np.array([0,200,0], dtype=np.uint8)
    # Higher value equivalent in pure HSL in (360,100,100) (pure white)
    high_threshold = np.array([180,255,255], dtype=np.uint8)
    white_mask = cv2.inRange(img, low_threshold, high_threshold)
    return white_mask


def HSLYellowIsolate(img):
    # Lower threshold equivalent in pure HSL is (30,45,15)
    low_threshold = np.array([15,38,115], dtype=np.uint8)
    # Higher value equivalent in pure HSL in (360,100,100)
    high_threshold = np.array([35,204,255], dtype=np.uint8)
    yellow_mask = cv2.inRange(img, low_threshold, high_threshold)
    return yellow_mask

hsl_white_images = list(map(lambda img: HSLWhiteIsolate(img), hsl_images))
hsl_yellow_images = list(map(lambda img: HSLYellowIsolate(img), hsl_images))

# --- plotting white-filtered and yellow-filtered HSL images _DOC_RP_

"""
img_count = len(hsl_images)
interleaved_isolated_hsl = list(zip(images,hsl_white_images,hsl_yellow_images))

k = 0
for iso_hsl in interleaved_isolated_hsl:
    img_name = image_names[k]
    showImageList(iso_hsl, cols=3, fig_size=(15,15), img_labels=[img_name,img_name,img_name])
    k += 1
"""

# --- combining yellow and white masks then applying to the original images

def HSLCombine(img, hsl_yellow, hsl_white):
    hsl_mask = cv2.bitwise_or(hsl_yellow, hsl_white)
    return cv2.bitwise_and(img,img,mask=hsl_mask)

def HSLImageFilter(img):
    hsl_img = convertToHSL(img)
    hsl_yellow = HSLYellowIsolate(hsl_img)
    hsl_white = HSLWhiteIsolate(hsl_img)
    return HSLCombine(img,hsl_yellow,hsl_white)

combined_yw_hsl_images = hsl_images = list(map(lambda img: HSLImageFilter(img), images))

# --- plotting the combined Y+W filtered versions _DOC_RP_

"""
img_count = len(combined_yw_hsl_images)
interleaved_combined_hsl = list(zip(images,combined_yw_hsl_images))

k = 0
for cb_hsl in interleaved_combined_hsl:
    img_name = image_names[k]
    showImageList(cb_hsl, cols=2, fig_size=(15,15), img_labels=[img_name,img_name])
    k += 1
"""

# --- conversion to grayscale (for higher contrast)

def grayScale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

grayscale_images = list(map(lambda img: grayScale(img), combined_yw_hsl_images))
# showImageList(grayscale_images) _DOC_RP_

# --- applying gaussian blur to make details vanish / feel less important than important straight lines

def gaussianBlur(img, kernel_size=3):
    return cv2.GaussianBlur(img, (kernel_size,kernel_size),0)

# For this program's sake, we'll keep a value of 7 for the gaussian blur kernel (more in the report). Here's a brief comparison with a higher value
    
blurred_images7 = list(map(lambda img: gaussianBlur(img,kernel_size=7), grayscale_images))
blurred_images23 = list(map(lambda img: gaussianBlur(img,kernel_size=23), grayscale_images))

# --- plotting our grayscaled, blurred, yellow+white filtered versions (low blur vs. high blur) _DOC_RP_ 

"""
img_count = len(blurred_images7)
interleaved_blur = list(zip(blurred_images7,blurred_images23))

k = 0
for blurred in interleaved_blur:
    img_name = image_names[k]
    showImageList(blurred, cols=2,fig_size=(15,15),img_labels=[img_name,img_name])
    k += 1
"""

# --- Canny Edge detection (more info in RP) _DOC_RP_

def cannyEdgeDetector(blurred_img, low, high):
    return cv2.Canny(blurred_img, low, high)

# We're testing two low/high thresholds settings: (50,150) and (0,10). The first one seems to be more effective. _DOC_RP_

canny_images50_150 = list(map(lambda img: cannyEdgeDetector(img,50,150), blurred_images7))
canny_images0_10 = list(map(lambda img: cannyEdgeDetector(img,0,10), blurred_images7))

# --- plotting canny edge processed pictures at this stage. _DOC_RP_

"""
img_count = len(canny_images50_150)
interleaved_canny = list(zip(canny_images50_150, canny_images0_10))

k = 0
for canny in interleaved_canny:
    img_name = image_names[k]
    showImageList(canny, cols=2, fig_size=(15,15), img_labels=[img_name,img_name])
    k += 1
"""

# --- defining a region of interest. The idea is to dismiss any detail outside a set polygon on the image, assuming that the road is flat and the camera in the car remains in the same place.

def getArea(img):
    h,w = img.shape[0],img.shape[1]
    area = None
    
    # Since we have 2 possible image sizes, we treat them separately. Letters stand for bottom, top, left, right.
    
    if (w,h) == (960,540):
        bl = (130,h - 1)
        tl = (410,330)
        tr = (650,350)
        br = (w - 30, h - 1)
        area = np.array([[bl,tl,tr,br]], dtype=np.int32)
    else:
        bl = (200,680)
        tl = (600,450)
        tr = (750,450)
        br = (1100,650)
        area = np.array([[bl,tl,tr,br]], dtype=np.int32)
        
    return area

def regionOfInterest(img):
    # starting by setting a blank mask
    mask = np.zeros_like(img)
    # depending on the source image, we might need more than 1 channel color to fill the mask (eg. img.shape can be 2 (width, height) or more (counting RGB, alpha, etc...))
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        mask_color = (255,) * channel_count
    else:
        mask_color = 255
        
    area = getArea(img)
    # filling pixels inside the polygon with the fill color
    cv2.fillPoly(mask,area,mask_color)
    # returning the image only where mask pixels are nonzero
    return cv2.bitwise_and(img,mask)

segmented_images = list(map(lambda img: regionOfInterest(img), canny_images50_150))
canny_segmented_images = list(zip(canny_images50_150, segmented_images))

# --- displaying canny segmented images (i.e. canny edge processed, lanes-focused images) _DOC_RP_

"""
showImageList(segmented_images)
"""

# --- Hough transform. More on this in RP
    
def houghTransform(canny_img, rho, theta, threshold, min_line_len, max_line_gap):
    return cv2.HoughLinesP(canny_img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

rho = 1 # 1 degree
theta = (np.pi/180) * 1
threshold = 15
min_line_length, max_line_gap = 20,10

hough_lines_per_image = list(map(lambda img: houghTransform(img,rho,theta,threshold,min_line_length,max_line_gap), segmented_images))
    
# --- drawing new Hough lines
def drawLines(img, lines, color=[0,255,0], thickness=10, make_copy=True):
    # Copy the passed image
    img_copy = np.copy(img) if make_copy else img
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img_copy, (x1,y1), (x2,y2), color, thickness)
    
    return img_copy

# --- showing our images with identified lines overlay _DOC_RP_
"""
img_with_lines = list(map(lambda img, lines: drawLines(img,lines), images, hough_lines_per_image))

showImageList(img_with_lines, fig_size=(15,15))
"""
# --- lane separation. We use the fact that the left line has a positive slope, while the right one has a negative one.

def lineSeparator(lines, img):
    img_shape = img.shape
    middle_x = img_shape[1] / 2
    left_lane_lines,right_lane_lines = [],[]
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            dx = x2 - x1
            if dx == 0:
                continue # discarding the line
            dy = y2 - y1
            if dy == 0:
                continue # same
            slope = dy/dx
            
            # We're only keeping lines with a significant slope. Some slight slopes can simply mean horizontal noised lines therefore these aren't important.
            eps = 0.1
            if abs(slope) < eps:
                continue
            
            if slope < 0 and x1 <  middle_x and x2 < middle_x:
                left_lane_lines.append([[x1,y1,x2,y2]])
            elif x1 >= middle_x and x2 >= middle_x:
                right_lane_lines.append([[x1,y1,x2,y2]])
                
    return left_lane_lines, right_lane_lines

separated_lanes_per_image = list(map(lambda lines, img: lineSeparator(lines, img), hough_lines_per_image, images))

# --- coloring the line fragments we previously got (red left, blue right)
def colorLanes(img, left_lane_lines, right_lane_lines, left_lane_color=[255,0,0], right_lane_color=[0,0,255]):
    left_colored_img = drawLines(img, left_lane_lines, color=left_lane_color, make_copy=True)
    right_colored_img = drawLines(left_colored_img, right_lane_lines, color=right_lane_color, make_copy=False)
    return right_colored_img

img_different_lane_colors = list(map(lambda img, separated_lanes: colorLanes(img, separated_lanes[0], separated_lanes[1]), images, separated_lanes_per_image))

# showImageList(img_different_lane_colors)

# --- final step before testing on other pictures/videos: lane extrapolation. To do so, we'll interpolate the points on a given lane and find the line that minimises the distance across the points (linear regression) 

from scipy import stats

def findLinesCoef(lines):
    xs,ys = [],[]
    print(lines)
    for line in lines:
        for x1,y1,x2,y2 in line:
            xs.append(x1)
            xs.append(x2)
            ys.append(y1)
            ys.append(y2)
        
    a,b,r,p,std_err = stats.linregress(xs,ys)
    return(a,b)

def drawLaneLine(img, lines, top_y, make_copy=True):
    a,b = findLinesCoef(lines)
    img_shape = img.shape
    
    # For a line, y = ax + b <=> x = (y - b)/a
    bottom_y = img_shape[0] - 1
    x_to_bottom_y = (bottom_y - b)/a
    top_x_to_y = (top_y - b)/a
    
    new_lines = [[[int(x_to_bottom_y),int(bottom_y),int(top_x_to_y),int(top_y)]]]
    
    return drawLines(img, new_lines, make_copy=make_copy)

def drawFull(img, left_lane_lines, right_lane_lines):
    area = getArea(img)
    tl = area[0][1]
    
    full_left_lane_img = drawLaneLine(img,left_lane_lines,tl[1], make_copy=True)
    full_LR_lanes_img = drawLaneLine(full_left_lane_img,right_lane_lines,tl[1],make_copy=False)
    
    laneweight_img = cv2.addWeighted(img, 0.7, full_LR_lanes_img,0.3,0.0)
    return laneweight_img

final_images = list(map(lambda img, separated_lanes : drawFull(img, separated_lanes[0], separated_lanes[1]), images, separated_lanes_per_image))

showImageList(final_images)





