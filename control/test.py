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


def mouse_callback(event, x, y, flags, param): 
    print("mouse event, x:", x ," y:", y)

if __name__ == "__main__":
    # demo_color()
    # cv2.namedWindow("src")
    file_list = glob.glob("./resources/*")
    print("hello")
    
    frame = cv2.imread(file_list[0], cv2.IMREAD_GRAYSCALE)
    #frame = cv2.resize(frame, None, None, 0.5, 0.5, cv2.INTER_AREA)

    cv2.imshow("frame_bin", frame)

    cv2.setMouseCallback("frame_bin",mouse_callback)
    # bbox = bounding_box(106, 51, 143, 62)
    # color_detect_by_histogram(frame, bbox)
    cv2.waitKey(0)