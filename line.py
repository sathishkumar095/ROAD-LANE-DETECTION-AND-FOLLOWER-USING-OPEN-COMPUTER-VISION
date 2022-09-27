import cv2
import numpy as np
import math
import imutils

# Average of Slope and intercept
def avg_slope_intercept(frame, line_segments):
    lane_lines=[]

    if line_segments is None:
        print("No Detected")
        return lane_lines

    height, width,_ = frame.shape
    left_fit = []
    right_fit = []
    boundary = 1/3
    left_region = width * (1-boundary)
    right_region = width * boundary

    for line_segment in line_segments:
        for x1,y1,x2,y2 in line_segment:
            if x1 == x2:
                # print('x1',x1)
                # print('x2', x2)
                # print("No slope, the line is vertical")
                continue

            fit = np.polyfit((x1,x2),(y1,y2), 1)
            slope = (y2-y1) /(x2-x1)
            intercept = y1 - (slope * x1)

            if slope < 0:
                if x1 < left_region and x2 < left_region:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region and x2 > right_region:
                    right_fit.append((slope, intercept))

    left_fit_avg = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(points(frame,left_fit_avg))

    right_fit_avg = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(points(frame,right_fit_avg))

    return lane_lines

# points
def points(frame,line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height
    y2 = int(y1 / 2)
    if slope == 0:
        slope = 0.1

    x1 = int((y1-intercept) / slope)
    x2 = int((y2-intercept) / slope)
    return np.array([x1,y1,x2,y2])

# display_lines
def display_lines(frame, lines,line_width=5):
    theta=0
    line_image = np.zeros_like(frame)
    if lines is not None:
        for i in range(0, len(lines)):
            for x1,y1,x2,y2 in lines[i]:
                line_image = cv2.line(frame,(x1,y1),(x2,y2), (0,255,0),line_width)
                theta = theta + math.atan2((y2 - y1), (x2 - x1))
    threshold = 6
    if (theta > threshold):
        text = 'Left'
        print("Left")
    if (theta < (-threshold)):
        text = 'Right'
        print("Right")
    if (abs(theta) < threshold):
        text = 'Straight'
        print("Straight")
    theta = 0
    font = cv2.FONT_HERSHEY_COMPLEX
    lineType = cv2.LINE_AA
    cv2.putText(frame, text, (100, 200), font, 4, (0, 0, 255), 3, lineType, False)
    return line_image

# angle detection
def angle_detect(frame, lane_lines):
    height, width,_ = frame.shape

    if len(lane_lines) == 2:
        left_x2 = lane_lines[0][0]
        right_x2 = lane_lines[1][0]
        mid_val = int(width / 2)
        x_offset = left_x2 + right_x2 / 2 - mid_val
        y_offset = int(height / 2)

    elif len(lane_lines)==1:
        x1 = lane_lines[0][0]
        x2 = lane_lines[0][0]
        x_offset = x2 - x1
        y_offset = int(height / 2)

    elif len(lane_lines) == 0:
        x_offset = 0
        y_offset = int (height / 2)

    angle_mid = math.atan2(x_offset, y_offset)
    angle_mid_deg = int(angle_mid * ((180.0 )/ math.pi))
    camera_angle = angle_mid_deg + 90
    return camera_angle

# display line
def display_line(frame, camera_angle, line_color =(0,0,255), line_width =5):
    get_frame = np.zeros_like(frame)
    height, width, _ = frame.shape
    camera_angle_val = camera_angle / (180.0 * math.pi)
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / (math.tan(camera_angle_val)))
    y2 = int(height / 2)
    cv2.line(get_frame,(x1,y1),(x2,y2), line_color, line_width)

    get_frame_out = cv2.addWeighted(frame, 0.8, get_frame, 1,1)
    # print('get_frame_out',get_frame_out)
    return get_frame_out

# video input
video = cv2.VideoCapture("D:/test_video.mp4")

while True:
    if video.isOpened():
        ret, frame = video.read()
        frame = imutils.resize(frame, width=650)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurr = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blurr, 85, 85)
        line_segments = cv2.HoughLinesP(edges, 1, np.pi/180, 10, minLineLength=5, maxLineGap=10)
        lane_lines = avg_slope_intercept(frame, line_segments)
        lines_image = display_lines(frame,line_segments)
        angle = angle_detect(frame,lane_lines)
        camera_angle = display_line(lines_image,angle)
        cv2.imshow('Frame',frame)
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
