**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: ./output_images/calibration1_undistort.png "Calibration"
[image1]: ./test_images/straight_lines1.jpg "Original"
[image2]: ./output_images/straight_lines1_undistorted.jpg "Undistorted"
[image3]: ./output_images/straight_lines1_binary.jpg "Road Transformed"
[image4]: ./output_images/straight_lines1_binary_warped.jpg "Warp Example"
[image5]: ./output_images/straight_lines1_histogram.jpg "Histogram"
[image6]: ./output_images/straight_lines1_windows.jpg "Windows Search"
[image7]: ./output_images/straight_lines1_fit.jpg "Polynomial Fit"
[image8]: ./output_images/straight_lines1_result.jpg "Output"
[video1]: ./project_video_output.mp4 "Project Video"
[video2]: ./challenge_video_output.mp4 "Challenge Video"
[animated1]: ./project_output.gif "Animated Project Video"
[animated2]: ./challenge_output.gif "Animated Challenge Project Video"
## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!
### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the method calibrate() (line # 30 to 84) of the file 'process_video.py'.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera matrix and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image0]

I then save the camera matrix and distortion coefficients to a pickle file so that I can load them from the file instead of doing the calculation again.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
I then apply the cv2.undistort method() to one of the test images like this one:
![alt text][image1]

Below is the undistorted image:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image in the method combine_threshold() (line # 83 through 104 in the file `threshold.py`).  Here's an example of my output for this step.  

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a method called `warp_perspective()`, which appears in line # 118 to 140 in the file 'process_video.py'. The method takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
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
```
This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 580, 460      | 203, 0        |
| 203, 720      | 203, 720      |
| 1127, 720     | 1127, 720      |
| 695, 460      | 1127, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I first take a histogram along all the columns in the lower half of the warped image. With this histogram I am adding up the pixel values along each column in the image.

![alt text][image5]

In the warped image, pixels are either 0 or 1, so the two most prominent peaks in this histogram will be good indicators of the x-position of the base of the lane lines. I can use that as a starting point for where to search for the lines. From that point, I can use a sliding window, placed around the line centers, to find and follow the lines up to the top of the frame. The code appears  in the 8th code cell of the IPython notebook 'visualization.ipynb' and the visulization below is produced by the 9th code cell.

![alt text][image6]

I then fit my lane lines with a 2nd order polynomial. The visulization below is produced by the 10th code cell of the IPython notebook 'visualization.ipynb'.

![alt text][image7]

For video processing, this is implemented as the method fit_ploy() (line # 178 to 247) in the file 'process_video.py'.

Once we have the lines fit for a frame, in the next frame of video we don't need to do a blind search again. Instead we can just search in a margin around the previous line. This is implemented as the method refine_poly (line # 251 to 291) in the file 'process_video.py'. This improves speed and provide a more robust method for rejecting outliers.

To smooth the lane detection over frames, I keep a history of 10 most recent fits for each lane, and return the mean of these 10 fits as the new fit.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculated the curvature in the method calc_radius() (line # 147 to 153) in the file 'process_video.py', according to the formula listed in the [Measuring Curvature lecture](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/2b62a1c3-e151-4a0e-b6b6-e424fa46ceab/lessons/40ec78ee-fb7c-4b53-94a8-028c5c60b858/concepts/2f928913-21f6-4611-9055-01744acc344f)

Once these have been calculated, I do some checks to determine if the curvature is good or not. First I check to see that curvature of the lane is above a minimum threshold of 30. I selected this threshold by looking at the U.S. government specifications for highway curvature.

The second check I do is wheather each lane's curvature is within 5 times (larger or smaller) from the previous curvature. This is implemented in the add_fit() method in the file line.py using the curverad_factor variable.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in the 11th code cell of IPython notebook 'visualization.ipynb', which calls the method draw_lane_lines() (line # 294 to 339) of the file 'process_video.py'.  Here is an example of my result on a test image:

![alt text][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

And below is the animated gif version:
![alt text][animated1]
---

Here's a [link to the challenge video result](./challenge_video_output.mp4)

And below is corresponding animated gif version:
![alt text][animated2]

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My pipeline consists of using a combination of x directional derivative and s_chanel threshold values to extract the lane lines. This may not work under different lighting or pavement conditions. For exmaple, it does poorly on the challenge video where the pavement has uneven color and the edge between the different colors would be extracted instead of the dotted white lines. The source points used for the perspective transformation also do not work very well as the the distance between the lanes are different.

To workaround this probelm, I used the following set of source points for the challenge video:

```
challenge_src = np.float32(
    [[(img_size[0] / 2) - 15, (img_size[1] * 2) / 3],
    [(img_size[0] / 4) - 30, img_size[1]],
    [(img_size[0] * 5 / 6), img_size[1]],
    [(img_size[0] / 2 + 93), (img_size[1] * 2) / 3]])
```
This resulted in the following source points:

| Source        |
|:-------------:|
| 625, 480      |
| 290, 720      |
| 1067, 720     |
| 733, 480      |  

and I used a combination of r_channel and s_channel threshold values to extract the lane lines.

Further work needs to be done to investigate how to handle different conditions systematically.
