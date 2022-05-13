#!/usr/bin/env python

# motor mapping 0.8 if angle is between 20 and 30

import cv2
import rospy
import rospkg
import sys
import os
import signal
import math
import numpy as np
import rospy, serial, time
from xycar_msgs.msg import xycar_motor
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from yolov3_trt_ros.msg import BoundingBoxes, BoundingBox

motor_msg = xycar_motor()
trt_msg = BoundingBoxes()
obj_id = -1
obj_width = 0
obj_position = []
stopped_before = False
# red_sign_before = True

def callback(data) :
    global obj_id
    global obj_width
    global obj_position
    lim = 40
    obj_id = -1
    obj_width = 0
    obj_position = []

    for bbox in data.bounding_boxes:
        print(bbox.id)
        if bbox.id == 5:
            if bbox.xmax - bbox.xmin > 40:
                obj_id = bbox.id
                obj_width = bbox.xmax - bbox.xmin
                obj_position = [bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]
            break
        if bbox.xmax - bbox.xmin > obj_width:
            if bbox.xmax - bbox.xmin > lim:
                obj_id = bbox.id
                obj_width = bbox.xmax - bbox.xmin
                obj_position = [bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]

def signal_handler(sig, frame):
    os.system('killall -9 python rosout')
    sys.exit(0)

# moving average


class MovingAverage:
    def __init__(self, n):
        self.samples = n
        self.data = []
        self.weights = list(range(1, n+1))

    def add_sample(self, new_sample):
        if len(self.data) < self.samples:
            self.data.append(new_sample)
        else:
            self.data = self.data[1:] + [new_sample]

    def get_wmm(self):
        s = 0
        for i, x in enumerate(self.data):
            s += x * self.weights[i]
        return float(s) / sum(self.weights[:len(self.data)])

    def get_mm(self):
        s = 0
        for x in self.data:
            s += x
        return float(s) / self.samples


# all global variables
tm = cv2.TickMeter()
Gap = 70
Offset = 0
# center offset of xycar in usb camera
center_offset = 20
# PID values
ie, de, pe = 0, 0, 0
# image shape
Height, Width = 480, 640
# bridge for changing /usb_cam/image_raw topic to OpenCV image type
bridge = CvBridge()

pub = None

l_ex, r_ex = 0, 640

# Set ROI points for bird eye view
src_pts = np.array([[20, 290], [550, 290], [600, 370],
                   [0, 370]], dtype=np.float32)
dst_pts = np.array([[0, 0], [639, 0], [639, 479], [0, 479]], dtype=np.float32)
mat_pers = cv2.getPerspectiveTransform(src_pts, dst_pts)
# For inRange function but this will not be used for now (22.4.5)
lbound = np.array([0, 0, 0], dtype=np.uint8)
rbound = np.array([75, 255, 255], dtype=np.uint8)

# calibrating for usb_camera.
calibrated = True
if calibrated:
    mtx = np.array([
        [422.037858, 0.0, 245.895397],
        [0.0, 435.589734, 163.625535],
        [0.0, 0.0, 1.0]
    ])
    dist = np.array([-0.289296, 0.061035, 0.001786, 0.015238, 0.0])
    cal_mtx, cal_roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (Width, Height), 1, (Width, Height))


def calibrate_image(frame, Height, Width):
    global mtx, dist
    global cal_mtx, cal_roi
    tf_image = cv2.undistort(frame, mtx, dist, None, cal_mtx)
    x, y, w, h = cal_roi
    tf_image = tf_image[y:y+h, x:x+w]
    return cv2.resize(tf_image, (Width, Height))

# callback function for Subscriber node


def img_callback(data):
    global image
    image = bridge.imgmsg_to_cv2(data, "bgr8")

# publish xycar_motor msg


def drive(Angle, Speed):
    global pub
    msg = xycar_motor()
    msg.angle = Angle
    msg.speed = Speed
    pub.publish(msg)

# dividing left, right lines with HoughLinesP result


def divide_left_right(lines):
    global Width
    low_slope_threshold = 0
    high_slope_threshold = 1000
    # calculate slope & filtering withcenter threshold
    slopes = []
    new_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            slope = 0
        else:
            slope = float(y2-y1) / float(x2-x1)

        if abs(slope) > low_slope_threshold and abs(slope) < high_slope_threshold:
            slopes.append(slope)
            new_lines.append(line[0])

    # divide lines left to right
    left_lines = []
    right_lines = []

    for j in range(len(slopes)):
        Line = new_lines[j]
        slope = slopes[j]

        x1, y1, x2, y2 = Line

        if (slope < 0) and (x1 < Width/2):
            left_lines.append([Line.tolist()])
        elif (slope > 0) and (x2 > Width/2):
            right_lines.append([Line.tolist()])

    return left_lines, right_lines

# get average m, b of lines


def get_line_params(lines):
    # sum of x, y, m
    x_sum = 0.0
    y_sum = 0.0
    m_sum = 0.0

    size = len(lines)
    if size == 0:
        return 0, 0

    for line in lines:
        x1, y1, x2, y2 = line[0]

        x_sum += x1 + x2
        y_sum += y1 + y2
        m_sum += float(y2 - y1) / float(x2 - x1)

    x_avg = x_sum / (size * 2)
    y_avg = y_sum / (size * 2)
    m = m_sum / size
    b = y_avg - m * x_avg

    return m, b


def get_line_pos(img, lines, left=False, right=False):
    global Width, Height
    global Offset, Gap

    m, b = get_line_params(lines)

    if m == 0 and b == 0:
        if left:
            pos = 0
        if right:
            pos = Width
    else:
        y = Gap / 2
        pos = (y - b) / m

        b += Offset
        x1 = (Height - b) / float(m)
        x2 = ((Height/2) - b) / float(m)

        cv2.line(img, (int(x1), Height), (int(x2), (Height/2)), 0, 3)

    return img, int(pos)


def process_image(frame):
    global Height, Width, mat_pers
    global l_ex, r_ex
    # gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # calibrate
    gray_cal = calibrate_image(gray, Height, Width)
    # ROI selection (bird eye view)
    roi = cv2.warpPerspective(gray_cal, mat_pers, (Width, Height))
    # blur
    roi = cv2.GaussianBlur(roi, (0, 0), 4)
    # Canny
    roi_canny = cv2.Canny(roi, 40, 50)
    # HoughLinesP
    lines = cv2.HoughLinesP(roi_canny, 1.0, math.pi/180,
                            30, minLineLength=30, maxLineGap=10)
    # divide_left_right
    if lines is None:
        return l_ex, r_ex
    left_lines, right_lines = divide_left_right(lines)

    # get center of lines
    frame, lpos = get_line_pos(roi, left_lines, left=True)
    frame, rpos = get_line_pos(roi, right_lines, right=True)

    if (lpos == 0 and rpos == 640):
        lpos = l_ex
        rpos = r_ex
    elif (lpos == 0):
        lpos = rpos - 500
    elif (rpos == 640):
        rpos = lpos + 500

    l_ex, r_ex = lpos, rpos
    return lpos, rpos


def PID(error, p_gain, i_gain, d_gain):
    global ie, de, pe
    de = error - pe
    pe = error
    if (-3500 < ie < 3500):
        ie += error
    else:
        ie = 0

    return p_gain*pe + i_gain*ie + d_gain*de


def stop_line_detect(frame):
    # perspective transform
    src_points = np.float32([[74, 350], [540, 350], [639, 479], [0, 479]])
    dst_points = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])
    per_mat = cv2.getPerspectiveTransform(src_points, dst_points)

    # gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # bird
    gray_bird = cv2.warpPerspective(gray, per_mat, (300, 300))
    # frame_roi
    gray_roi = gray_bird[60:136, 130:171]
    gray_roi = cv2.GaussianBlur(gray_roi, (3, 3), 1)
    gray_roi_bin = cv2.inRange(gray_roi, 0, 100)

    total_area = (135-30)*(170-130)
    line_area = np.count_nonzero(gray_roi_bin)
    # print(line_area, total_area)
    if (float(line_area)/total_area) > 0.1:
        cv2.rectangle(gray_bird, (130, 30), (170, 135), (255), 3)
        # print("stop_line")
        return ["stop", 0], gray_bird
    else:
        cv2.rectangle(gray_bird, (130, 30), (170, 135), (0), 3)
        return ["go", 1], gray_bird


def detection2drive(obj_id, obj_width, obj_position, frame):
    global stopped_before
    # global red_sign_before
    stopped_before = False

    if obj_id == 0:
        drive(-33, 8)
    elif obj_id == 1:
        drive(28, 8)
    elif obj_id == 2:
        drive(0, 0)
        stopped_before = True
        rospy.sleep(5)
    elif obj_id == 3:
        drive(0, 0)
        stopped_before = True
        rospy.sleep(5)
    elif obj_id == 5:
        isRED, _ = color_detect_by_histogram(frame[obj_position[1]:obj_position[3],obj_position[0]:obj_position[2]])
        if isRED == 0:
            drive(0, 8)
        else:
            drive(0, 0)
            rospy.sleep(5)
            drive(0, 8)
            rospy.sleep(1)


def color_detect_by_histogram(frame):
    # global red_sign_before

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_bin = cv2.inRange(frame_hsv, (0, 0, 225), (255, 255, 255))

    sign_len = len(frame_bin)
    hist_list = [0] * sign_len

    for y in range(len(frame_bin)):
        for x in range(len(frame_bin[0])):
            if frame_bin[y, x] != 0:
                hist_list[y] += 1

    max_idx = hist_list.index(max(hist_list))

    if max_idx < sign_len/3:
        print("red")
        # red_sign_before = True
        return 1, frame_bin
    elif max_idx < sign_len*2/3:
        print("yello")
        #red_sign_before = True
        return 1, frame_bin
    else:
        print("green")
        # red_sign_before = False
        return 0, frame_bin



def start():
    global pub
    global image
    global cap
    global Width, Height
    global tm
    global obj_id
    global obj_width
    global obj_position
    global stopped_before

    angle_mov = MovingAverage(5)

    stop_line_flag = False

    rospy.init_node('auto_drive')
    pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)

    image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, img_callback)
    rospy.Subscriber('/yolov3_trt_ros/detections', BoundingBoxes, callback, queue_size=1)
    rospy.sleep(2)

    tm.reset()
    tm.start()
    angle_ex = 0
    drive(0, 0)

    rospy.sleep(5)

    while True:
        while not image.size == (640*480*3):
            continue

        if obj_id != -1:
            if obj_id == 2 or obj_id == 3:
                if not stopped_before:
                    detection2drive(obj_id, obj_width, obj_position, image)
                else: pass
            else:
                detection2drive(obj_id, obj_width, obj_position, image)
                rospy.sleep(0.4)
                continue

        # if red_sign_before:
            # if obj_id != -1:
                # detection2drive(obj_id, obj_width, obj_position, image)
            # else:
                # continue

        stop_sig, gray_bird = stop_line_detect(image)

        lpos, rpos = process_image(image)

        center = (lpos + rpos) / 2
        error = -(320 + 0 - center)
        # print("lpos: ",lpos, ", rpos:", rpos,"\ncenter: ",center,", error: ",error)
        # print("--------------------------")
        tm.stop()
        time_now = tm.getTimeSec()
        tm.start()

        angle = max(min(50, error), -50)
        angle_mov.add_sample(angle)
        angle = angle_mov.get_wmm()

        angle = PID(angle, 0.75, 0.0005, 0.03)
        # angle = PID(angle, 1, 0, 0)

        if (abs(angle-angle_ex) > 70):
            angle = angle_ex

        if stop_sig[0] == "stopAAZZ":
            if stop_line_flag:
                drive(angle, 5)
                stop_line_flag = False
                rospy.sleep(0.2)
            else:
                drive(0, 0)
                rospy.sleep(5)
                stop_line_flag = True
        else:
            drive(angle, 10)

        angle_ex = angle

    rospy.spin()


if __name__ == '__main__':

    start()
