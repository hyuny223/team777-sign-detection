import cv2
import glob
import sys

class bounding_box:
    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

def color_detect_by_histogram(frame, box_info):
    
    
    sign_len = len(frame)
    traffic_height = abs(box_info.xmax-box_info.xmin)
    hist_list = [0] * sign_len

    for y in range(box_info.ymax, box_info.ymin):
        for x in range(box_info.xmax, box_info.xmin):
            if frame[y, x] != 0:
                hist_list[y] += 1

    max_idx = hist_list.index(max(hist_list))
    print(max_idx)

    # if max_idx < sign_len/3:
    #     print("red")
    #     return False
    # elif max_idx < sign_len*2/3:
    #     print("yello")
    #     return False
    # else:
    #     print("green")
    #     return True


def onChange(pos):
    pass


def demo_color():
    file_list = glob.glob("./resources/*")
    cv2.namedWindow("src")
    cv2.createTrackbar("hl: ", "src", 0, 255, onChange)
    cv2.createTrackbar("sl: ", "src", 0, 255, onChange)
    cv2.createTrackbar("vl: ", "src", 0, 255, onChange)
    cv2.createTrackbar("hu: ", "src", 0, 255, onChange)
    cv2.createTrackbar("su: ", "src", 0, 255, onChange)
    cv2.createTrackbar("vu: ", "src", 0, 255, onChange)
    cv2.setTrackbarPos("hu: ", "src", 255)
    cv2.setTrackbarPos("su: ", "src", 255)
    cv2.setTrackbarPos("vu: ", "src", 255)
    cv2.setTrackbarPos("vl: ", "src", 128)
    for file in file_list:
        frame = cv2.imread(file, cv2.IMREAD_COLOR)
        frame = cv2.resize(frame, None, None, 0.5, 0.5, cv2.INTER_AREA)
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        while True:
            tmp = cv2.waitKey(1)
            if tmp == 27:
                sys.exit(1)
            elif tmp == 110:
                break
            hl = cv2.getTrackbarPos("hl: ", "src")
            sl = cv2.getTrackbarPos("sl: ", "src")
            vl = cv2.getTrackbarPos("vl: ", "src")
            hu = cv2.getTrackbarPos("hu: ", "src")
            su = cv2.getTrackbarPos("su: ", "src")
            vu = cv2.getTrackbarPos("vu: ", "src")
            frame_bin = cv2.inRange(frame_hsv, (hl, sl, vl), (hu, su, vu))
            cv2.imshow("frame_bin", frame_bin)


def demo_color2():
    file_list = glob.glob("./resources/*")
    frame = cv2.imread(file_list[0], cv2.IMREAD_COLOR)
    frame = cv2.resize(frame, None, None, 0.5, 0.5, cv2.INTER_AREA)
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    while True:
        frame_bin = cv2.inRange(frame_hsv, (0, 0, 225), (255, 255, 255))
        frame_roi = frame_bin[147:295, 534:620]

        sign_len = len(frame_roi)
        hist_list = [0] * sign_len

        for y in range(len(frame_roi)):
            for x in range(len(frame_roi[0])):
                if frame_roi[y, x] != 0:
                    hist_list[y] += 1
        max_idx = hist_list.index(max(hist_list))

        if max_idx < sign_len/3:
            print("red")
        elif max_idx < sign_len*2/3:
            print("yello")
        else:
            print("green")
        cv2.imshow("frame_bin", frame_roi)
        if cv2.waitKey() == 27:
            sys.exit(1)

def mouse_callback(event, x, y, flags, param): 
    print("mouse event, x:", x ," y:", y)

if __name__ == "__main__":
    # demo_color()
    # cv2.namedWindow("src")
    file_list = glob.glob("./resources/*")
    print("hello")
    
    frame = cv2.imread(file_list[0], cv2.IMREAD_GRAYSCALE)
    frame = cv2.resize(frame, None, None, 0.5, 0.5, cv2.INTER_AREA)

    cv2.imshow("frame_bin", frame)

    cv2.setMouseCallback("frame_bin",mouse_callback)
    bbox = bounding_box(106, 51, 143, 62)
    color_detect_by_histogram(frame, bbox)
    cv2.waitKey(0)
    
