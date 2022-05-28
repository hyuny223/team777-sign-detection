import cv2
import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="video2img")
    parser.add_argument("--video_path", type=str,
                        help="video path", default=None)
    parser.add_argument("--img_path", type=str,
                        help="image directory. for example, ./output",
                        default="./output")
    parser.add_argument("--frame_pass", type=int,
                        help="frame gaps passed per image", default=1)
    if (len(sys.argv) == 1):
        # python file opened without any argument
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    count = 0
    args = parse_args()
    cap = cv2.VideoCapture(args.video_path)
    video_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    if args.frame_pass != 0:
        frame_pass = args.frame_pass
        print("frame pass:", frame_pass,
              "\nExpected frames:", int(video_frames/frame_pass))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_pass != 0:
            count += 1
            continue
        file_name = args.img_path + '/' + str(int(count/frame_pass)) + ".png"
        cv2.imwrite(file_name, frame)
        count += 1
    print("The job is successfully finished!")
