import cv2
import numpy as np


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
    gray_roi = gray_bird[30:136, 130:171]
    gray_roi = cv2.GaussianBlur(gray_roi, (3, 3), 1)
    gray_roi_bin = cv2.inRange(gray_roi, 0, 150)

    total_area = (135-30)*(170-130)
    line_area = np.count_nonzero(gray_roi_bin)
    print(line_area, total_area)
    if (float(line_area)/total_area) > 0.1:
        cv2.rectangle(gray_bird, (130, 30), (170, 135), (255), 3)
        return ["stop", 0], gray_bird
    else:
        cv2.rectangle(gray_bird, (130, 30), (170, 135), (0), 3)
        return ["go", 1], gray_bird


def demo():
    cap = cv2.VideoCapture("video_2.avi")

    skip_frame = 500
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if skip_frame > 0:
            skip_frame -= 1
            continue

        sig, gray_bird = stop_line_detect(frame)
        cv2.imshow("frame", frame)
        cv2.imshow("gray_bird", gray_bird)
        if cv2.waitKey() == 27:
            print("break")
            break


if __name__ == "__main__":
    demo()
