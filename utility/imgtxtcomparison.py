import glob
import sys
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="video2img")
    parser.add_argument("--path1", type=str,
                        help="path1: , ex) ./JPEGImages", default=None)
    parser.add_argument("--ext1", type=str,
                        help="ex) .png", default=None)
    parser.add_argument("--txtpath", type=str,
                        help="txt path ex) ./Annotations", default=None)
    if (len(sys.argv) == 1):
        # python file opened without any argument
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    if args.path1 is None or args.txtpath is None:
        # 2 elements needed
        parser.print_help()
        sys.exit(1)
    return args


if __name__ == "__main__":
    args = parse_args()
    path1_filelist = glob.glob(args.path1 + "/*" + args.ext1)
    path2_filelist = glob.glob(args.txtpath + "/*.txt")

    for file_name in path1_filelist:
        tmp = file_name.replace(args.ext1, ".txt")
        tmp = tmp.replace(args.path1, args.txtpath)
        if tmp in path2_filelist:
            continue
        else:
            with open(tmp, 'a') as f:
                f.write("")
