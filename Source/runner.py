import os
import sys
from os import listdir
from inkML2png import run


def visit_folder(absolute_path, path_from_root=""):
    folders_files_list = listdir(absolute_path)
    files_list = filter(lambda x: x.find('.inkml') != -1, folders_files_list)
    folders_list = filter(lambda x: x.find('.') == -1, folders_files_list)

    files_list = list(files_list)
    if len(files_list) > 0:
        os.makedirs(os.path.dirname(f"{output_path}{path_from_root}" + "raw_input\\"), exist_ok=True)


    for folder in folders_list:
        visit_folder(absolute_path + folder + "\\", folder + "\\")
    for i, file in enumerate(files_list):
        data = run(f"{absolute_path}{file}", f"{output_path}{path_from_root}" + "raw_input\\" + f"{i}.png")
        if data is not None:
            os.makedirs(os.path.dirname(f"{output_path}{path_from_root}" + "truth\\"), exist_ok=True)
            with open(f"{output_path}{path_from_root}" + "truth\\" + f"{i}.txt", "w") as f:
                f.write(data)
        else:
            with open(f"{output_path}skipped_files_list.txt", "a") as f:
                f.write(f"{absolute_path}{file}")

output_path = __file__[: __file__.rfind("\\") - 5] + "input\\"

# print(output_path)
visit_folder(sys.argv[1] + "\\")
