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
from traffic_light_detection import TrafficLine_Detector
from yolov3_trt_ros.src.lane_detector import Lane_Detector

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



# all global variables
tm = cv2.TickMeter()

# center offset of xycar in usb camera
center_offset = 20
# PID values
ie, de, pe = 0, 0, 0
# image shape
Height, Width = 480, 640
# bridge for changing /usb_cam/image_raw topic to OpenCV image type
bridge = CvBridge()

pub = None

lbound = np.array([0, 0, 0], dtype=np.uint8)
rbound = np.array([75, 255, 255], dtype=np.uint8)

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



def PID(error, p_gain, i_gain, d_gain):
    global ie, de, pe
    de = error - pe
    pe = error
    if (-3500 < ie < 3500):
        ie += error
    else:
        ie = 0

    return p_gain*pe + i_gain*ie + d_gain*de



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
        isRED, _ = TrafficLine_Detector.detect(frame[obj_position[1]:obj_position[3],obj_position[0]:obj_position[2]])
        if isRED == 0:
            drive(0, 8)
        else:
            drive(0, 0)
            rospy.sleep(5)
            drive(0, 8)
            rospy.sleep(1)



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

    stop_line_flag = False

    rospy.init_node('auto_drive')
    pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)

    image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, img_callback)
    rospy.Subscriber('/yolov3_trt_ros/detections', BoundingBoxes, callback, queue_size=1)
    rospy.sleep(2)

    lane_detector = Lane_Detector()

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

        stop_sig, gray_bird = lane_detector.stop_line_detect(image)
        
        angle = lane_detector.get_steering_angle(image)

        angle = PID(angle, 0.75, 0.0005, 0.03)

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


if __name__ == '__main__':

    start()
