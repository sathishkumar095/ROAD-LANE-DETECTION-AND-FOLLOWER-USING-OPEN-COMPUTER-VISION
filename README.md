ROAD LANE DETECTION AND FOLLOWER USING OPENCV
In this project, I have designed a Lane Follower using OpenCV and Python Programming. Its capable of follow a lane detection (in roadways). It can be used in industries for assisting the Production Process, human assistance etc.
Concept of Lane Detection
A Lane is part of Roadway that is designated to be used by a single line of vehicles, to control and guide drivers and reduce traffic conflicts.
Problem Statement
The task that we wish to perform is that of real-time lane detection and follower in a video. There are multiple ways we can perform lane detection. Using the OpenCV Library in Python, simpler methods to perform lane detection as well. As we can see in this image, we have four lanes separated by colored lane markings. So, to detect a lane, we must detect the red-markings on either side of that Lane.
Implementing Lane Detection and follower using OpenCV in python
Here, we using the Pycharm editor to build a code for lane detection and follower.
download here!
You can also the download any others editor or frameworks
First step, install the libraries in your system.
1. Opencv – pip install opencv-python
2. Numpy – pip install numpy
3. imutils – pip install imutils
Now, Import the libraries into python code. # Import Libraries import cv2 import numpy as np import math import imutils
Read the image from your directory video = cv2.VideoCapture("C:/Users/robot/Downloads/test5.mp4") cv2.imshow('Frame', video)
OpenCV is a vast library that helps in providing various functions for image and video operations. With OpenCV, we can capture a video from the camera. cv2.VideoCapture() - create a video capture object which is helpful to capture videos through webcam and then you may perform desired operations on that video.
Preprocessing image for Lane Detection and Follower
To apply a mask to all the frames in our input video. Then, we will apply image thresholding followed by Hough Line Transformation to detect lane markings.
Image Thresholding
The pixel values of a grayscale image are assigned one of the two values representing black and white colors based on a threshold value. So, if the value of a pixel is greater than a threshold value, it is assigned one value, else it is assigned the other value. gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
Above the line, cv2.COLOR_BGR2GRAY is convert the color image to Grayscale image for better identification from the image or frame (video part).
Using the Image Smoothing using OpenCV Gaussian Blur
As in any other signals, images also can contain different types of noise, especially because of the source (camera). Image Smoothing techniques help in reducing the noise. In OpenCV, image smoothing (also called blurring) could be done in many ways, cv2.GaussianBlur() using the Gaussian filter for image smoothing.
Note: Gaussian filters have the properties of having no overshoot to a step function input while minimizing the rise and fall time. In terms of image processing, any sharp edges in images are smoothed while minimizing too much blurring. blurr = cv2.GaussianBlur(gray, (5,5), 0)
Here, we taken the values of blur,
src= grayscale image,
kernel size = (5, 5)
bordertype =0.
Using the Canny Edge Detector
The Canny edge detector is used for edge detector in all of computer vision and image processing. In OpenCV has already implemented it for us in the cv2.Canny function. It is only purpose of edges detection part based on image smoothing part. edges = cv2.Canny(blurr, 85, 85)
Here,
image = Blur image or frame (Source/Input image of n-dimensional array).
threshold1 = 85 (High threshold value of intensity gradient).
threshold2 = 85 (Low threshold value of intensity gradient).
After applying thresholding on the image, we get only the lane markings in the output image. Now it easier detect these markings with the help of Hough Line Transformation.
Using the Hough Line Transformation
Hough Transform is a technique to detect any shape that can be represented mathematically click here!. For example, it can detect shapes like rectangles, circles, triangles, or lines. We are interested in detecting lane markings that can be represented as lines. line_segments = cv2.HoughLinesP(edges, 1, np.pi/180, 10, minLineLength=5, maxLineGap=10)
Here, values of Hough Lines,
image = from the canny edges image or frame.
Rho = 1 (distance resolution of the accumulator in pixels).
Theta = np.pi/180 (angle resolution of the accumulator in radians).
Threshold = 10
minLineLength = 5 (Line segments shorter than that are rejected).
maxLineGap = 10 (maximum allowed gap between points on the same line to link them).
Applying Hough Line Transformation on the image after performing image thresholding will give us the below output:
To follow this process for all the frames and then stitch the resultant frames into a new video. After the frame transformation into Hough lines, Now finds out the lines of slope and intercept better than of lines identification moves direction path.
Average of Slope and Intercept
Remember, Line equation is given by y = mx + b, where m is the slope of the line and bis the y-intercept. In this section, the average of slopes and intercepts of line segments detected using Hough Transformation will be calculated.
From the frame or image, multiple lines detected for each lane line, so we need to average of all lines and points that each lane line to cover the full lane line length.
1. If line or lines is not available is frame, it will show No Detected.
2. Instead, the line is available, then its frame can shape into height and width parameters. Based Line segments from lines, they can calculate the slope and intercept points of x and y-axes.
a) The left lane appears to be going upwards so it has a negative slope (remember the coordinate system start point?). In other words, the left lane line has x1 < x2 and y2 < y1 and the slope = (y2 - y1) / (x2 - x1) which will give a negative slope. Therefore, all lines with negative slopes are considered left lane points.
b) The right lane is the complete opposite, we can see that the right lane is going downwards and will have positive slope. Right lane has x2 > x1 and y2 > y1 which will give a positive slope. So, all lines with positive slope are considered right lane points.
c) In case of vertical lines (x1 = x2), the slope will be infinity. In this case, we will skip all vertical lines to prevent getting an error.
Note: Either the slope is less than zero, it will store values in left-side region or slope is greater than, it will store values in right-side region. Then finding the average part of left-side and right-side region.
To add more accuracy to this detection, each frame is divided into two regions (right and left) through 2 boundary lines. All width points (x-axis points) greater than right boundary line, are associated with right lane calculation. And if all width points are less than the left boundary line, they are associated with left lane calculation.
Below of code for average of Slope and intercept: def avg_slope_intercept(frame, line_segments): lane_lines=[] if line_segments is None: print("No Detected") return lane_lines height, width,_ = frame.shape left_fit = [] right_fit = [] boundary = 1/3 left_region = width * (1-boundary) right_region = width * boundary for line_segment in line_segments: for x1,y1,x2,y2 in line_segment: if x1 == x2: continue fit = np.polyfit((x1,x2),(y1,y2), 1) slope = (y2-y1) /(x2-x1) intercept = y1 - (slope * x1) if slope < 0: if x1 < left_region and x2 < left_region: left_fit.append((slope, intercept)) else: if x1 > right_region and x2 > right_region: right_fit.append((slope, intercept)) left_fit_avg = np.average(left_fit, axis=0) if len(left_fit) > 0: lane_lines.append(points(frame,left_fit_avg)) right_fit_avg = np.average(right_fit, axis=0) if len(right_fit) > 0: lane_lines.append(points(frame,right_fit_avg)) return lane_lines
The points() is a helper function for avg_slope_intercept() function which will return the bounded coordinates of the lane lines (from the bottom to the middle of the frame).
Below of code for points: def points(frame,line): height, width, _ = frame.shape slope, intercept = line y1 = height y2 = int(y1 / 2) if slope == 0: slope = 0.1
x1 = int((y1-intercept) / slope) x2 = int((y2-intercept) / slope) return np.array([x1,y1,x2,y2])
To prevent dividing by 0, a condition is presented. If slope = 0 which means y1 = y2 (horizontal line), give the slope a value near 0. This will not affect the performance of the algorithm as well as it will prevent impossible case (dividing by 0).
Display Lines
The display lines, is detected and following of lane_lines from hough line, the lines are not none, then it iterates the coordinates values x and y axes. line_image = cv2.line(frame,(x1,y1),(x2,y2), (0,255,0),line_width)
Using cv2.line(), to draw the line on lane part or region and based the theta values and threshold values (we fix threshold values = 6), we get the left or right or straight. cv2.putText(frame, text, (100, 200), font, 4, (0, 0, 255), 3, cv2.LINE_AA, False)
Using cv2.putText() method is used to draw a text string on any image. Here, the ‘font’ is denote the font types are FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_PLAIN, etc.
Below of code for display lines # display_lines def display_lines(frame, lines,line_width=5): # print('f= ',frame) theta=0 line_image = np.zeros_like(frame) if lines is not None: for i in range(0, len(lines)): for x1,y1,x2,y2 in lines[i]: line_image = cv2.line(frame,(x1,y1),(x2,y2), (0,255,0),line_width) theta = theta + math.atan2((y2 - y1), (x2 - x1)) threshold = 6 if (theta > threshold): text = 'Left' print("Left") if (theta < (-threshold)): text = 'Right' print("Right") if (abs(theta) < threshold): text = 'Straight' print("Straight") theta = 0 font = cv2.FONT_HERSHEY_COMPLEX # fontScale = 4 # color = (0, 0, 255) # (B, G, R) # thickness = 3 lineType = cv2.LINE_AA cv2.putText(frame, text, (100, 200), font, 4, (0, 0, 255), 3, lineType, False) return line_image
Detection the angle of lane line region
Finally, the angle of Lane line region, from the input part as lane_lines (average of slope and intercept function), taking the lane_lines,
a) if length_of_lines is 2, then we get the values of x-axis of left and right.
b) if length_of_lines is 1, then we get the values of x-axis are same.
c) if length_of_lines is 0, the no value.
Above three cases are using the same formulation of angle detection to x_offset is the average ((right x2 + left x2) / 2) differs from the middle of the screen and y_offset is always taken to be height / 2.
If angle = 90, it means that the object has a display line perpendicular to "height / 2" line and the object will move forward. If angle > 90, the object should turn to right otherwise it should move left.
These are values are return in to camera_angle, here the display_line from the angle detection part to frame-out of lines to display the screen with colored lines on it.
Below of code for angle detection: # angle detection def angle_detect(frame, lane_lines): height, width,_ = frame.shape if len(lane_lines) == 2: left_x2 = lane_lines[0][0] right_x2 = lane_lines[1][0] mid_val = int(width / 2) x_offset = left_x2 + right_x2 / 2 - mid_val y_offset = int(height / 2) elif len(lane_lines)==1: x1 = lane_lines[0][0] x2 = lane_lines[0][0] x_offset = x2 - x1 y_offset = int(height / 2) elif len(lane_lines) == 0: x_offset = 0 y_offset = int (height / 2) angle_mid = math.atan2(x_offset, y_offset) angle_mid_deg = int(angle_mid * ((180.0 )/ math.pi)) camera_angle = angle_mid_deg + 90 return camera_angle # display line def display_line(frame, camera_angle, line_color =(0,0,255), line_width =5): get_frame = np.zeros_like(frame) height, width, _ = frame.shape camera_angle_val = camera_angle / (180.0 * math.pi) x1 = int(width / 2) y1 = height x2 = int(x1 - height / 2 / (math.tan(camera_angle_val))) y2 = int(height / 2) cv2.line(get_frame,(x1,y1),(x2,y2), line_color, line_width)
get_frame_out = cv2.addWeighted(frame, 0.8, get_frame, 1,1) # print('get_frame_out',get_frame_out) return get_frame_out
Okay, Now combing the all lines of code to main program python file: # Import Libraries import cv2 import numpy as np import math import imutils # Average of Slope and intercept def avg_slope_intercept(frame, line_segments): lane_lines=[] if line_segments is None: print("No Detected") return lane_lines height, width,_ = frame.shape left_fit = [] right_fit = [] boundary = 1/3 left_region = width * (1-boundary) right_region = width * boundary for line_segment in line_segments: for x1,y1,x2,y2 in line_segment: if x1 == x2: # print('x1',x1) # print('x2', x2) # print("No slope, the line is vertical") continue fit = np.polyfit((x1,x2),(y1,y2), 1) slope = (y2-y1) /(x2-x1) intercept = y1 - (slope * x1) if slope < 0: if x1 < left_region and x2 < left_region: left_fit.append((slope, intercept)) else: if x1 > right_region and x2 > right_region: right_fit.append((slope, intercept)) left_fit_avg = np.average(left_fit, axis=0) if len(left_fit) > 0: lane_lines.append(points(frame,left_fit_avg)) right_fit_avg = np.average(right_fit, axis=0) if len(right_fit) > 0: lane_lines.append(points(frame,right_fit_avg)) return lane_lines # points def points(frame,line): height, width, _ = frame.shape slope, intercept = line
y1 = height y2 = int(y1 / 2) if slope == 0: slope = 0.1 x1 = int((y1-intercept) / slope) x2 = int((y2-intercept) / slope) return np.array([x1,y1,x2,y2]) # display_lines def display_lines(frame, lines,line_width=5): theta=0 line_image = np.zeros_like(frame) if lines is not None: for i in range(0, len(lines)): for x1,y1,x2,y2 in lines[i]: line_image = cv2.line(frame,(x1,y1),(x2,y2), (0,255,0),line_width) theta = theta + math.atan2((y2 - y1), (x2 - x1)) threshold = 6 if (theta > threshold): text = 'Left' print("Left") if (theta < (-threshold)): text = 'Right' print("Right") if (abs(theta) < threshold): text = 'Straight' print("Straight") theta = 0 font = cv2.FONT_HERSHEY_COMPLEX lineType = cv2.LINE_AA cv2.putText(frame, text, (100, 200), font, 4, (0, 0, 255), 3, lineType, False) return line_image # angle detection def angle_detect(frame, lane_lines): height, width,_ = frame.shape if len(lane_lines) == 2: left_x2 = lane_lines[0][0] right_x2 = lane_lines[1][0] mid_val = int(width / 2) x_offset = left_x2 + right_x2 / 2 - mid_val y_offset = int(height / 2) elif len(lane_lines)==1: x1 = lane_lines[0][0] x2 = lane_lines[0][0] x_offset = x2 - x1 y_offset = int(height / 2) elif len(lane_lines) == 0: x_offset = 0 y_offset = int (height / 2) angle_mid = math.atan2(x_offset, y_offset) angle_mid_deg = int(angle_mid * ((180.0 )/ math.pi)) camera_angle = angle_mid_deg + 90 return camera_angle
# display line def display_line(frame, camera_angle, line_color =(0,0,255), line_width =5): get_frame = np.zeros_like(frame) height, width, _ = frame.shape camera_angle_val = camera_angle / (180.0 * math.pi) x1 = int(width / 2) y1 = height x2 = int(x1 - height / 2 / (math.tan(camera_angle_val))) y2 = int(height / 2) cv2.line(get_frame,(x1,y1),(x2,y2), line_color, line_width) get_frame_out = cv2.addWeighted(frame, 0.8, get_frame, 1,1) # print('get_frame_out',get_frame_out) return get_frame_out # video input video = cv2.VideoCapture("C:/Users/robot/Downloads/test5.mp4") while True: if video.isOpened(): ret, frame = video.read() frame = imutils.resize(frame, width=650) gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) blurr = cv2.GaussianBlur(gray, (5,5), 0) edges = cv2.Canny(blurr, 85, 85) line_segments = cv2.HoughLinesP(edges, 1, np.pi/180, 10, minLineLength=5, maxLineGap=10) lane_lines = avg_slope_intercept(frame, line_segments) lines_image = display_lines(frame,line_segments) angle = angle_detect(frame,lane_lines) camera_angle = display_line(lines_image,angle) cv2.imshow('Frame',frame) if cv2.waitKey(1) == ord('q'): break video.release() cv2.destroyAllWindows()
Here, the little bit change of cv2.VideoCapture, we taken the only the video format manner above the code.
Now another way we are tried out, based on IP-address manner of using this road lane detection and follower using real-time function. We use the IP webcam software application should be install in your phone (Android or iPhone).
Then the taken IPv4 address, enter or paste in this code, where the cv2.VideoCapture() video=cv2.VideoCapture('https://192.168.1.152:8080/video')
You can get the real-time detection and follower part.!!
