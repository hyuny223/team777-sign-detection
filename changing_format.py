import glob


file_list = glob.glob("./default/label_2/*.txt")
input_dir = "./default/label_2/"
output_dir = "./output/"


data_table = {
    'left': 0,
    'right': 1,
    'stop': 2,
    'crosswalk': 3,
    'uturn': 4,
    'traffic_light': 5,
    'ignore': 6
}

for file_name in file_list:
    tmp = ""
    with open(file_name, "r") as f:
        lines = f.readlines()
        for line in lines:
            data_split = line.split()
            data_kitti = ""
            data_kitti = data_kitti + str(data_table[data_split[0]])
            xyxy = list(map(float, data_split[4:8]))
            xywh = [round((xyxy[0]+xyxy[2])/2/640, 6), round((xyxy[1]+xyxy[3])
                                                             / 2/480, 6), round((xyxy[2]-xyxy[0])/640, 6), round((xyxy[3]-xyxy[1])/480, 6)]

            for datum in xywh:
                data_kitti = data_kitti + ' ' + str(datum)
                output_file = output_dir+file_name.replace(input_dir, "")
            tmp = tmp + data_kitti + "\n"
        with open(output_file, "a") as f2:
            f2.write(tmp)
