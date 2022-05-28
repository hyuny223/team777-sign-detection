import cv2

class TrafficLine_Detector:


    def detect(frame):
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
