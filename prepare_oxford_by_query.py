import os
import shutil

oxford_path = '/mnt/sdb-seagate/datasets/oxford5k/oxbuild_images_small/'
oxford_dest = '/mnt/sdb-seagate/datasets/oxford5k/oxbuild_by_queries/'
gt_path = 'gt-oxford/'


files = os.listdir(gt_path)
query_dict = {}
for file in files:
    if file.endswith("_good.txt") or file.endswith("_ok.txt"):

        if file.endswith("_good.txt"):
            key_name = file[:-len("_good.txt")]
        else:
            key_name = file[:-len("_ok.txt")]

        if key_name not in query_dict.keys():
            query_dict[key_name] = []

        with open(gt_path + file, 'r') as list_positives:
            for line in list_positives.readlines():
                query_dict[key_name] += [line[:-len("\n")]]
    # elif file.endswith("_query.txt"):
    #     key_name = file[:-len("_query.txt")]
    #     if key_name not in query_dict.keys():
    #         query_dict[key_name] = []
    #
    #     with open(gt_path + file, 'r') as list_positives:
    #         query_pic = list_positives.readline().split(" ")[0][len("oxc1_"):]
    #         query_dict[key_name] += [query_pic]

for query in query_dict.keys():
    os.mkdir(oxford_dest + query)

for query in query_dict.keys():
    for pic in query_dict[query]:
        shutil.copy(oxford_path + pic + ".jpg", oxford_dest + "/{}/".format(query) + pic + ".jpg")
