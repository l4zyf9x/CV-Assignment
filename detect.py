from moviepy.editor import VideoFileClip
# from IPython.display import HTML
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import deque
from scipy import stats
import cv2
import numpy as np
import os

MAXIMUM_SLOPE_DIFF = 0.1
MAXIMUM_INTERCEPT_DIFF = 50.0
rho = 1
# 1 degree
theta = (np.pi/180) * 1
threshold = 15
min_line_length = 20
max_line_gap = 10

# Convenience function used to show a list of images
def show_image_list(img_list, img_labels, cols=2, fig_size=(15, 15), show_ticks=True):
    img_count = len(img_list)
    rows = img_count / cols
    cmap = None
    plt.figure(figsize=fig_size)
    for i in range(0, img_count):
        img_name = img_labels[i]
        
        plt.subplot(rows, cols, i+1)
        img = img_list[i]
        if len(img.shape) < 3:
            cmap = "gray"
        
        if not show_ticks:
            plt.xticks([])
            plt.yticks([])
            
        plt.title(img_name[len(test_img_dir):])    
        plt.imshow(img, cmap=cmap)

    plt.tight_layout()
    plt.show()

def to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
def to_hsl(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

# Image should have already been converted to HSL color space
def isolate_yellow_hsl(img):
    # Caution - OpenCV encodes the data in ****HLS*** format
    # Lower value equivalent pure HSL is (30, 45, 15)
    low_threshold = np.array([15, 38, 115], dtype=np.uint8)
    # Higher value equivalent pure HSL is (75, 100, 80)
    high_threshold = np.array([35, 204, 255], dtype=np.uint8)  
    
    yellow_mask = cv2.inRange(img, low_threshold, high_threshold)
    
    return yellow_mask
                            

# Image should have already been converted to HSL color space
def isolate_white_hsl(img):
    # Caution - OpenCV encodes the data in ***HLS*** format
    # Lower value equivalent pure HSL is (30, 45, 15)
    low_threshold = np.array([0, 200, 0], dtype=np.uint8)
    # Higher value equivalent pure HSL is (360, 100, 100)
    high_threshold = np.array([180, 255, 255], dtype=np.uint8)  
    
    yellow_mask = cv2.inRange(img, low_threshold, high_threshold)
    
    return yellow_mask

def combine_hsl_isolated_with_original(img, hsl_yellow, hsl_white):
    hsl_mask = cv2.bitwise_or(hsl_yellow, hsl_white)
    return cv2.bitwise_and(img, img, mask=hsl_mask)

def filter_img_hsl(img):
    hsl_img = to_hsl(img)
    hsl_yellow = isolate_yellow_hsl(hsl_img)
    hsl_white = isolate_white_hsl(hsl_img)
    return combine_hsl_isolated_with_original(img, hsl_yellow, hsl_white)

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def gaussian_blur(grayscale_img, kernel_size=3):
    return cv2.GaussianBlur(grayscale_img, (kernel_size, kernel_size), 0) 

def canny_edge_detector(blurred_img, low_threshold, high_threshold):
    return cv2.Canny(blurred_img, low_threshold, high_threshold)

def get_vertices_for_img(img):
    img_shape = img.shape
    # print('Image Shape', img.shape)
    height = img_shape[0]
    width = img_shape[1]

    vert = None
    
    if (width, height) == (960, 540):
        region_bottom_left = (130 ,img_shape[0] - 1)
        region_top_left = (410, 330)
        region_top_right = (650, 350)
        region_bottom_right = (img_shape[1] - 30,img_shape[0] - 1)
        vert = np.array([[region_bottom_left , region_top_left, region_top_right, region_bottom_right]], dtype=np.int32)
    else:
        # region_bottom_left = (200 , 680)
        # region_top_left = (600, 450)
        # region_top_right = (750, 450)
        # region_bottom_right = (1100, 650)
        # chanlengen
        region_bottom_left = (width*0.1, height*0.95)
        region_top_left = (width*0.4, height*0.6)
        region_bottom_right = (width*0.9, height*0.95)
        region_top_right = (width*0.6, height*0.6)
    
        # # test_2
        # region_bottom_left = (width*0.25, height*0.95)
        # region_bottom_right = (width*0.65, height*0.95)
        # region_top_left = (width*0.49, height*0.7)
        # region_top_right = (width*0.51, height*0.7)



        vert = np.array([[region_bottom_left , region_top_left, region_top_right, region_bottom_right]], dtype=np.int32)


    return vert

def region_of_interest(img):
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
        
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    vert = get_vertices_for_img(img)    
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vert, ignore_mask_color)
    
    mpimg.imsave(out_dir + "6_mask_img.jpg",mask, cmap='gray')

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def hough_transform(canny_img, rho, theta, threshold, min_line_len, max_line_gap):
    return cv2.HoughLinesP(canny_img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

def draw_lines(img, lines, color=[255, 0, 0], thickness=10, make_copy=True):
    # Copy the passed image
    img_copy = np.copy(img) if make_copy else img
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img_copy, (x1, y1), (x2, y2), color, thickness)
    
    return img_copy

def separate_lines(lines, img):
    img_shape = img.shape
    # print('img_shape',img_shape)

    middle_x = img_shape[1] / 2
    
    left_lane_lines = []
    right_lane_lines = []
    left_slope = []
    right_slope = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            dx = x2 - x1 
            if dx == 0:
                #Discarding line since we can't gradient is undefined at this dx
                continue
            dy = y2 - y1
            
            # Similarly, if the y value remains constant as x increases, discard line
            if dy == 0:
                continue
            
            slope = dy / dx
            
            # This is pure guess than anything... 
            # but get rid of lines with a small slope as they are likely to be horizontal one
            epsilon = 0.1
            if abs(slope) <= epsilon:
                continue
            
            if slope < 0 and x1 < middle_x and x2 < middle_x:
                # Lane should also be within the left hand side of region of interest
                left_lane_lines.append([[x1, y1, x2, y2]])
                left_slope.append(slope)
            elif x1 >= middle_x and x2 >= middle_x:
                # Lane should also be within the right hand side of region of interest
                right_lane_lines.append([[x1, y1, x2, y2]])
                right_slope.append(slope)
    return left_lane_lines, right_lane_lines ,left_slope, right_slope

def color_lanes(img, left_lane_lines, right_lane_lines, left_lane_color=[255, 0, 0], right_lane_color=[0, 0, 255]):
    left_colored_img = draw_lines(img, left_lane_lines, color=left_lane_color, make_copy=True)
    right_colored_img = draw_lines(left_colored_img, right_lane_lines, color=right_lane_color, make_copy=False)
    
    return right_colored_img

def find_lane_lines_formula(lines):
    xs = []
    ys = []
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            xs.append(x1)
            xs.append(x2)
            ys.append(y1)
            ys.append(y2)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(xs, ys)
    
    # Remember, a straight line is expressed as f(x) = Ax + b. Slope is the A, while intercept is the b
    return (slope, intercept)

# def trace_lane_line(img, lines, top_y, make_copy=True):
#     A, b = find_lane_lines_formula(lines)
#     vert = get_vertices_for_img(img)

#     img_shape = img.shape
#     bottom_y = img_shape[0] - 1
#     # y = Ax + b, therefore x = (y - b) / A
#     x_to_bottom_y = (bottom_y - b) / A
    
#     top_x_to_y = (top_y - b) / A 
    
#     new_lines = [[[int(x_to_bottom_y), int(bottom_y), int(top_x_to_y), int(top_y)]]]
#     return draw_lines(img, new_lines, make_copy=make_copy)

# def trace_both_lane_lines(img, left_lane_lines, right_lane_lines):
#     vert = get_vertices_for_img(img)
#     region_top_left = vert[0][1]
    
#     full_left_lane_img = trace_lane_line(img, left_lane_lines, region_top_left[1], make_copy=True)
#     full_left_right_lanes_img = trace_lane_line(full_left_lane_img, right_lane_lines, region_top_left[1], make_copy=False)
    
#     # image1 * α + image2 * β + λ
#     # image1 and image2 must be the same shape.
#     img_with_lane_weight =  cv2.addWeighted(img, 0.7, full_left_right_lanes_img, 0.3, 0.0)
    
#     return img_with_lane_weight


def create_lane_line_coefficients_list(length = 10):
    return deque(maxlen=length)

def trace_lane_line_with_coefficients(img, line_coefficients, top_y, make_copy=True):
    A = line_coefficients[0]
    b = line_coefficients[1]
    
    img_shape = img.shape
    bottom_y = img_shape[0] - 1
    # y = Ax + b, therefore x = (y - b) / A
    x_to_bottom_y = (bottom_y - b) / A
    
    top_x_to_y = (top_y - b) / A 
    
    new_lines = [[[int(x_to_bottom_y), int(bottom_y), int(top_x_to_y), int(top_y)]]]
    return draw_lines(img, new_lines, make_copy=make_copy)


def trace_both_lane_lines_with_lines_coefficients(img, left_line_coefficients, right_line_coefficients):
    vert = get_vertices_for_img(img)
    region_top_left = vert[0][1]
    
    full_left_lane_img = trace_lane_line_with_coefficients(img, left_line_coefficients, region_top_left[1], make_copy=True)
    full_left_right_lanes_img = trace_lane_line_with_coefficients(full_left_lane_img, right_line_coefficients, region_top_left[1], make_copy=False)
    
    # image1 * α + image2 * β + λ
    # image1 and image2 must be the same shape.
    img_with_lane_weight =  cv2.addWeighted(img, 0.7, full_left_right_lanes_img, 0.3, 0.0)
    
    return img_with_lane_weight

class LaneDetectorWithMemory:
    def __init__(self):
        self.left_lane_coefficients  = create_lane_line_coefficients_list()
        self.right_lane_coefficients = create_lane_line_coefficients_list()
        
        self.previous_left_lane_coefficients = None
        self.previous_right_lane_coefficients = None
        
    
    def mean_coefficients(self, coefficients_queue, axis=0):        
        return [0, 0] if len(coefficients_queue) == 0 else np.mean(coefficients_queue, axis=axis)
    
    def determine_line_coefficients(self, stored_coefficients, current_coefficients):
        if len(stored_coefficients) == 0:
            stored_coefficients.append(current_coefficients) 
            return current_coefficients
        
        mean = self.mean_coefficients(stored_coefficients)
        abs_slope_diff = abs(current_coefficients[0] - mean[0])
        abs_intercept_diff = abs(current_coefficients[1] - mean[1])
        
        if abs_slope_diff > MAXIMUM_SLOPE_DIFF or abs_intercept_diff > MAXIMUM_INTERCEPT_DIFF:
            # print("Identified big difference in slope (", current_coefficients[0], " vs ", mean[0],
            #     ") or intercept (", current_coefficients[1], " vs ", mean[1], ")")
            
            # In this case use the mean
            return mean
        else:
            # Save our coefficients and returned a smoothened one
            stored_coefficients.append(current_coefficients)
            return self.mean_coefficients(stored_coefficients)
        

    def lane_detection_pipeline(self, img, out_dir="test_images_output/", step_save= True):
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        
        mpimg.imsave(out_dir + "1_in_img.jpg",img)
        
        combined_hsl_img = filter_img_hsl(img)
        mpimg.imsave(out_dir + "2_combined_hsl_img.jpg",combined_hsl_img)
        
        grayscale_img = grayscale(combined_hsl_img)
        mpimg.imsave(out_dir + "3_grayscale_img.jpg",grayscale_img, cmap='gray')

        # gaussian_smoothed_img = gaussian_blur(grayscale_img, kernel_size=5)
        # mpimg.imsave(out_dir + "4_gaussian_smoothed_img.jpg",gaussian_smoothed_img)

        # canny_img = canny_edge_detector(gaussian_smoothed_img, 50, 150)
        canny_img = canny_edge_detector(grayscale_img, 0, 10)
        mpimg.imsave(out_dir + "5_canny_img_0_10.jpg",canny_img, cmap='gray')
        canny_img = canny_edge_detector(grayscale_img, 10, 50)
        mpimg.imsave(out_dir + "5_canny_img_10_50.jpg",canny_img, cmap='gray')
        canny_img = canny_edge_detector(grayscale_img, 150, 250)
        mpimg.imsave(out_dir + "5_canny_img_150_250.jpg",canny_img, cmap='gray')
        canny_img = canny_edge_detector(grayscale_img, 50, 150)
        mpimg.imsave(out_dir + "5_canny_img_50_150.jpg",canny_img, cmap='gray')


        segmented_img = region_of_interest(canny_img)
        mpimg.imsave(out_dir + "6_segmented_img.jpg",segmented_img, cmap='gray')

        hough_lines = hough_transform(segmented_img, rho, theta, threshold, min_line_length, max_line_gap)
        # print('Hough Lines ', hough_lines.shape)
        hough_lines_img = draw_lines(img, hough_lines)
        mpimg.imsave(out_dir + "7_hough_lines_img.jpg",hough_lines_img)


        try:
            left_lane_lines, right_lane_lines, left_slope, right_slope = separate_lines(hough_lines, img)
            print('left:', left_slope)
            print('right', right_slope)
            different_lane_colors_img = color_lanes(img, left_lane_lines, right_lane_lines)
            mpimg.imsave(out_dir + "8_different_lane_colors_img.jpg",different_lane_colors_img)

            left_lane_slope, left_intercept = find_lane_lines_formula(left_lane_lines)
            right_lane_slope, right_intercept = find_lane_lines_formula(right_lane_lines)
            smoothed_left_lane_coefficients = self.determine_line_coefficients(self.left_lane_coefficients, [left_lane_slope, left_intercept])
            smoothed_right_lane_coefficients = self.determine_line_coefficients(self.right_lane_coefficients, [right_lane_slope, right_intercept])
            img_with_lane_lines = trace_both_lane_lines_with_lines_coefficients(img, smoothed_left_lane_coefficients, smoothed_right_lane_coefficients)
        
            mpimg.imsave(out_dir + "9_with_lane_lines_img.jpg",img_with_lane_lines)

            return img_with_lane_lines

        except Exception as e:
            print("*** Error - will use saved coefficients ", e)
            smoothed_left_lane_coefficients = self.determine_line_coefficients(self.left_lane_coefficients, [0.0, 0.0])
            smoothed_right_lane_coefficients = self.determine_line_coefficients(self.right_lane_coefficients, [0.0, 0.0])
            img_with_lane_lines = trace_both_lane_lines_with_lines_coefficients(img, smoothed_left_lane_coefficients, smoothed_right_lane_coefficients)
        
            # mpimg.imsave("test_images_output/with_lane_lines_img.jpg",img_with_lane_lines)

            return img_with_lane_lines






# white_output = 'test_videos_output/test_2.mp4'
# detector = LaneDetectorWithMemory()

# clip1 = VideoFileClip("test_videos/test_2.mp4")
# white_clip = clip1.fl_image(detector.lane_detection_pipeline)
# white_clip.write_videofile(white_output, audio=False)

# white_output = 'test_videos_output/challenge.mp4'
# detector = LaneDetectorWithMemory()

# clip1 = VideoFileClip("test_videos/challenge.mp4")
# white_clip = clip1.fl_image(detector.lane_detection_pipeline)
# white_clip.write_videofile(white_output, audio=False)


detector = LaneDetectorWithMemory()
clip1 = VideoFileClip("test_videos/challenge.mp4")
in_img = clip1.get_frame(1)
out_dir = "test_images_output/changlenge_frame_1/"
out_img = detector.lane_detection_pipeline(in_img, out_dir, step_save=False)

# detector = LaneDetectorWithMemory()
# clip1 = VideoFileClip("test_videos/test_2.mp4")
# in_img = clip1.get_frame(10)
# out_dir = "test_images_output/test_2_frame_10/"
# out_img = detector.lane_detection_pipeline(in_img, out_dir)
