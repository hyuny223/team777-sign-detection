import math
import cv2
import numpy as np

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

class Lane_Detector:
    tm = cv2.TickMeter()
    Height, Width = 480, 640
    Gap = 70
    Offset = 0
    l_ex, r_ex = 0, 640
    # Set ROI points for bird eye view
    src_pts = np.array([[20, 290], [550, 290], [600, 370],
                    [0, 370]], dtype=np.float32)
    dst_pts = np.array([[0, 0], [639, 0], [639, 479], [0, 479]], dtype=np.float32)
    mat_pers = cv2.getPerspectiveTransform(src_pts, dst_pts)

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
    
    angle_mov = MovingAverage(5)


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

    def getSteeringAngle(self, image):

        lpos, rpos = self.process_image(image)

        center = (lpos + rpos) / 2
        error = -(320 + 0 - center)

        self.tm.stop()
        time_now = self.tm.getTimeSec()
        self.tm.start()

        angle = max(min(50, error), -50)
        self.angle_mov.add_sample(angle)
        angle = self.angle_mov.get_wmm()

        return angle

    def process_image(self, frame):

        # gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calibrate
        gray_cal = self.calibrate_image(gray, self.Height, self.Width)
        # ROI selection (bird eye view)
        roi = cv2.warpPerspective(gray_cal, self.mat_pers, (self.Width, self.Height))
        # blur
        roi = cv2.GaussianBlur(roi, (0, 0), 4)
        # Canny
        roi_canny = cv2.Canny(roi, 40, 50)
        # HoughLinesP
        lines = cv2.HoughLinesP(roi_canny, 1.0, math.pi/180,
                                30, minLineLength=30, maxLineGap=10)
        # divide_left_right
        if lines is None:
            return self.l_ex, self.r_ex
        left_lines, right_lines = self.divide_left_right(lines)

        # get center of lines
        frame, lpos = self.get_line_pos(roi, left_lines, left=True)
        frame, rpos = self.get_line_pos(roi, right_lines, right=True)

        if (lpos == 0 and rpos == 640):
            lpos = self.l_ex
            rpos = self.r_ex
        elif (lpos == 0):
            lpos = rpos - 500
        elif (rpos == 640):
            rpos = lpos + 500

        self.l_ex, self.r_ex = lpos, rpos
        return lpos, rpos

    def calibrate_image(self, frame, Height, Width):

        tf_image = cv2.undistort(frame, self.mtx, self.dist, None, self.cal_mtx)
        x, y, w, h = self.cal_roi
        tf_image = tf_image[y:y+h, x:x+w]
        return cv2.resize(tf_image, (Width, Height))

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
    
    def get_line_pos(self, img, lines, left=False, right=False):
        m, b = self.get_line_params(lines)

        if m == 0 and b == 0:
            if left:
                pos = 0
            if right:
                pos = Width
        else:
            y = self.Gap / 2
            pos = (y - b) / m

            b += self.Offset
            x1 = (self.Height - b) / float(m)
            x2 = ((self.Height/2) - b) / float(m)

            cv2.line(img, (int(x1), self.Height), (int(x2), (self.Height/2)), 0, 3)

        return img, int(pos)

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

    def get_steering_angle(self, image):

        lpos, rpos = self.process_image(image)

        center = (lpos + rpos) / 2
        error = -(320 + 0 - center)

        self.tm.stop()
        time_now = self.tm.getTimeSec()
        self.tm.start()

        angle = max(min(50, error), -50)
        self.angle_mov.add_sample(angle)
        angle = self.angle_mov.get_wmm()

        return angle