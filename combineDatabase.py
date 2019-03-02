# coding=utf-8
import os
import shutil

# 得到库的所有不同手指文件夹的路径
def get_folder_dir(path):
    folder_dir = []
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            # print("image:", os.path.join(root, name))
            folder_dir.append(os.path.join(root, name))
    return folder_dir

# 将不同手指的文件夹复制到另外一个大的文件夹里
def copy_folders_to(folder_dir,dst_folder):
    for f in folder_dir:
        print(f)
        dst_dir = "%s/%s" % (dst_folder, f.split('/')[-1])
        # 如果已经存在，则采用将图片添加进文件夹的方法
        if (os.path.exists(dst_dir)):
            for root, dirs, files in os.walk(f, topdown=False):
                for name in files:
                    shutil.copy(os.path.join(root, name), os.path.join(dst_dir, name))
        else:
            # 否则直接将整个文件夹添加进去
            shutil.copytree(f, dst_dir)

# 一个个地复制
dst_folder='/home/users/wdd/myDatabase/SD_TH_ML_CP_set'

path1 = '/home/users/wdd/myDatabase/Finger_ROI/fvdb_ROI'
folder_dir1=get_folder_dir(path1)
copy_folders_to(folder_dir1, dst_folder)

path2 = '/home/users/wdd/myDatabase/Finger_ROI/MALAY_rawData_ROI'
folder_dir2=get_folder_dir(path2)
copy_folders_to(folder_dir2, dst_folder)

path3 = '/home/users/wdd/myDatabase/Finger_ROI/SDUMLA_rawData_ROI'
folder_dir3=get_folder_dir(path3)
copy_folders_to(folder_dir3, dst_folder)

path4 = '/home/users/wdd/myDatabase/Finger_ROI/SDUMLA_splitValidSet'
folder_dir4=get_folder_dir(path4)
copy_folders_to(folder_dir4, dst_folder)

path5 = '/home/users/wdd/myDatabase/Finger_ROI/TSINGHUA_rawData_ROI'
folder_dir5=get_folder_dir(path5)
copy_folders_to(folder_dir5, dst_folder)




