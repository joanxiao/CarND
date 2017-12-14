
# coding: utf-8

# ## Advanced Lane Finding Project
# 
# The goals / steps of this project are the following:
# 
# * Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# * Apply a distortion correction to raw images.
# * Use color transforms, gradients, etc., to create a thresholded binary image.
# * Apply a perspective transform to rectify binary image ("birds-eye view").
# * Detect lane pixels and fit to find the lane boundary.
# * Determine the curvature of the lane and vehicle position with respect to center.
# * Warp the detected lane boundaries back onto the original image.
# * Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

import numpy as np
import math
import cv2
import glob
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from line import Line
from threshold import combine_threshold, color_threshold
from moviepy.editor import VideoFileClip

#compute the camera calibration using chessboard images
def calibrate():
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('../camera_cal/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
            cv2.imshow('img',img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()
        
    img_size = (img.shape[1], img.shape[0])
    #calculate distortion coefficients and test undistortion on test images
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    # pickle the data and save it for later use
    calib_dict = {'mtx':mtx, 'dist':dist, 'img_size':img_size}
    with open(calib_file, 'wb') as f:
        pickle.dump(calib_dict, f)    
        print('calibration mtx and dist saved')            
    
    testfile = '../camera_cal/calibration1.jpg'
    img = cv2.imread(testfile)
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(undistorted)
    ax2.set_title('Undistorted Image', fontsize=50)

    f.savefig('../output_images/calibration1_undistort.png')      
    
    return mtx, dist, img_size 

#load the calibration matrix and distortion coefficients if available, otherwise calibrate the camera using chessboard corners.
def load_calib(calib_file):
    if os.path.exists(calib_file):    
        with open(calib_file, 'rb') as f:
            calib_dict = pickle.load(f)
            mtx = calib_dict["mtx"]
            dist = calib_dict["dist"]
            img_size = calib_dict["img_size"]
            print('calibration mtx and dist loaded. img_size={}'.format(img_size))
    else:
        mtx, dist, img_size = calibrate(calib_file)
        
    return mtx, dist, img_size   

mtx, dist, img_size = load_calib('calib.p')  
   
def undistort(img):          
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    return undistorted
 
default_src = np.float32(
        [[(img_size[0] / 2) - 60, img_size[1] / 2 + 100],
        [(img_size[0] / 6) - 10, img_size[1]],
        [(img_size[0] * 5 / 6) + 60, img_size[1]],
        [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])

default_dst = np.float32(
        [[(img_size[0] / 6) - 10, 0],
        [(img_size[0] / 6) - 10, img_size[1]],
        [(img_size[0] * 5 / 6) + 60, img_size[1]],
        [(img_size[0] * 5 / 6) + 60, 0]])

def warp_perspective(img, src=default_src, dst=default_dst, visualize=False):
    img_src = np.copy(img)
    # Grab the image shape
    img_size = (img.shape[1], img.shape[0])  
    
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    
    if visualize:
        cv2.line(img_src, (src[0][0], src[0][1]), (src[1][0], src[1][1]), [255, 0, 0], 2)
        cv2.line(img_src, (src[1][0], src[1][1]), (src[2][0], src[2][1]), [255, 0, 0], 2)
        cv2.line(img_src, (src[2][0], src[2][1]), (src[3][0], src[3][1]), [255, 0, 0], 2)
        cv2.line(img_src, (src[3][0], src[3][1]), (src[0][0], src[0][1]), [255, 0, 0], 2)

        cv2.line(warped, (dst[0][0], dst[0][1]), (dst[1][0], dst[1][1]), [255, 0, 0], 2)
        cv2.line(warped, (dst[1][0], dst[1][1]), (dst[2][0], dst[2][1]), [255, 0, 0], 2)
        cv2.line(warped, (dst[2][0], dst[2][1]), (dst[3][0], dst[3][1]), [255, 0, 0], 2)
        cv2.line(warped, (dst[3][0], dst[3][1]), (dst[0][0], dst[0][1]), [255, 0, 0], 2)
    
    return warped, img_src, Minv

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

#calculate radius of curvature
def calc_radius(x_vals, y_vals):
    # Fit new polynomials to x,y in orld space       
    y_eval = np.max(y_vals)     
    # Calculate the new radii of curvature
    fit_cr = np.polyfit(y_vals*ym_per_pix, x_vals*xm_per_pix, 2)
    line_curverad = ((1+ (2*fit_cr[0]*y_eval*ym_per_pix   + fit_cr[1])**2)**1.5)         / np.absolute(2*fit_cr[0])
    return int(line_curverad)

#not used
def check_slopes(h, left_fit, right_fit):
    top = h*5//6
    ploty = np.linspace(top, h-1, h) 
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
       
    left_slope = (ploty[-1]-ploty[0])/(left_fitx[-1]-left_fitx[0])
    print('left slope={}'.format(left_slope))
    right_slope = (ploty[-1]-ploty[0])/(right_fitx[-1]-right_fitx[0])
    print('right slope={}'.format(right_slope))
    
    if abs(left_slope) < 0.1 or abs(right_slope) < 0.1:      
        return False
        
    if left_slope * right_slope < 0.:
        if abs(left_slope/right_slope) > 2. or abs(right_slope/left_slope) > 2.:          
            return False
    
    return True

#fit a second order polynomial using a sliding windows method on a binary warped image.
#left_line and right_line are used to keep track of the history of fits.
def fit_poly(binary_warped, left_line, right_line): 
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
  
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one   
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices    
    left_lane_inds = np.concatenate(left_lane_inds)      
    right_lane_inds = np.concatenate(right_lane_inds)
       
    # Extract left and right line pixel positions
    if len(left_lane_inds)>=  minpix or left_line.best_fit is None:               
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        # Fit a second order polynomial
        left_fit = np.polyfit(lefty, leftx, 2)
        left_curverad = calc_radius(leftx, lefty) 
        left_line.add_fit(left_fit, left_curverad) 
            
    if len(right_lane_inds) >=  minpix or right_line.best_fit is None:       
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]        
        # Fit a second order polynomial
        right_fit = np.polyfit(righty, rightx, 2) 
        right_curverad = calc_radius(rightx, righty)   
        right_line.add_fit(right_fit, right_curverad)

#fit a new 2nd order polynomial based on an existing fit from previous frame. 
##left_line and right_line are used to keep track of the history of fits.     
def refine_poly(binary_warped, left_line, right_line):
    # Assume we have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!    
    left_fit = left_line.best_fit
    right_fit = right_line.best_fit
    
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # do not use the existing fit if not enough points are found
    if lefty.shape[0] < 5 or righty.shape[0] < 5:
        if lefty.shape[0] < 5: 
            left_line.detected = False   
        if righty.shape[0] < 5: 
            right_line.detected = False     
        return 
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    left_curverad = calc_radius(leftx, lefty)    
    right_fit = np.polyfit(righty, rightx, 2)
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y   
    right_curverad = calc_radius(rightx, righty)
    
    #if not check_slopes(binary_warped.shape[0], left_fit, right_fit):
    #    return
        
    left_line.add_fit(left_fit, left_curverad) 
    right_line.add_fit(right_fit, right_curverad) 

#Warp the detected lane boundaries back onto the original image and visually display lane curvature and vehicle position. 
def draw_lane_lines(binary_warped, undistorted, Minv, left_line, right_line):        
    if left_line.detected and right_line.detected:        
        refine_poly(binary_warped, left_line, right_line)                                                                                   
    if not left_line.detected or not right_line.detected:        
        fit_poly(binary_warped, left_line, right_line)                                   
                      
    left_fit = left_line.best_fit
    right_fit = right_line.best_fit
    left_curverad = left_line.curverad
    right_curverad = right_line.curverad        
                
    # draw the lines on the original image
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))          
    
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
   
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (binary_warped.shape[1], binary_warped.shape[0])) 
    
    # Combine the result with the original image
    result = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)
        
    #print('left_curverad={} right_curverad={}'.format(left_curverad, right_curverad))
    #display the average curvature of left and right lanes
    avg_curverad = (left_curverad + right_curverad)//2
    center = (left_fitx[-1] + right_fitx[-1])/2
    diff = (center - binary_warped.shape[1]/2) * xm_per_pix
    diff_pos = 'left'
    if diff < 0:
        diff_pos = 'right'
    cv2.putText(result, 'Radius of Curvature: ' + str(round(avg_curverad, 3)) + ' m', 
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.putText(result, 'Vehicle is ' + str(abs(round(diff, 3))) +  ' m ' + diff_pos + ' of center', 
               (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    
    return result
    
def process_video_image(img):      
    undistorted = undistort(img)
    binary_img = combine_threshold(img, (50, 100), (170, 255))
    binary_warped, _, Minv = warp_perspective(binary_img, visualize=False)   
    global left_line, right_line     
    result = draw_lane_lines(binary_warped, undistorted, Minv, left_line, right_line)
    
    return result
    
#for challenge video, use different src points for warping
challenge_src = np.float32(
    [[(img_size[0] / 2) - 15, (img_size[1] * 2) / 3],
    [(img_size[0] / 4) - 30, img_size[1]],
    [(img_size[0] * 5 / 6), img_size[1]],
    [(img_size[0] / 2 + 93), (img_size[1] * 2) / 3]])

#use r_channel and s_channel threshold to identify the lanes in challenge video
def process_challenge_video_image(img):     
    undistorted = undistort(img)
    binary_img = color_threshold(img, (50, 100), (170, 255))
    binary_warped, _, Minv = warp_perspective(binary_img, src=challenge_src, visualize=False)
    global left_line, right_line       
    result = draw_lane_lines(binary_warped, undistorted, Minv, left_line, right_line)      
        
    return result

if __name__ == '__main__':
    print('processing project video')
    #global left_line, right_line      
    left_line = Line(5)
    right_line = Line(5)
    #process project video
    input_video = '../project_video.mp4'
    ouput_video = '../project_video_output.mp4'
    clip = VideoFileClip(input_video)
    output_clip = clip.fl_image(process_video_image)
    output_clip.write_videofile(ouput_video, audio=False)
    
    print('------------------------------')
    print('processing challenge video')    
    #process challenge video
    input_video = '../challenge_video.mp4'
    ouput_video = '../challenge_video_output.mp4'
    left_line = Line(5)
    right_line = Line(5) 
    clip = VideoFileClip(input_video)
    output_clip = clip.fl_image(process_challenge_video_image)
    output_clip.write_videofile(ouput_video, audio=False)



