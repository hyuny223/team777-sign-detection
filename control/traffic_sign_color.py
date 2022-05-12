import cv2
import glob
import sys


def color_detect_by_histogram(frame):
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
        return 1
    elif max_idx < sign_len*2/3:
        print("yello")
        return 1
    else:
        print("green")
        return 0


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


if __name__ == "__main__":
    # demo_color()
    demo_color2()
