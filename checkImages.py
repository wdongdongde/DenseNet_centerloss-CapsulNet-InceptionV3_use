# coding=utf-8
#  检查图片是否有问题，如果有则删除
import os
from PIL import Image
import shutil
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

images=[]
path ='/home/wdxia/Finger_ROI_Database/ML_ROI_complete/MALAY_rawData_ROI'
# path = '/home/wdxia/Finger_ROI_Database/SD_ROI_complete/SDUMLA_rawData_ROI'
def get_image_files(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            # print("image:", os.path.join(root, name))
            images.append(os.path.join(root, name))

def remove_bad_images():
    get_image_files(path)
    s = images
    i = 0
    for image in s:
        try:
            image_file = image
            img = Image.open(image_file)
            img.verify()
        except IOError:
            i = i+1
            print(image_file)
            shutil.move(image_file,'/home/wdxia/Finger_ROI_Database/err_imgs/ML')
            # print(os.path.join('/home/users/wdd/myDatabase/err_imgs/ML', image_file.split('/')[-1]))
            os.rename(os.path.join('/home/wdxia/Finger_ROI_Database/err_imgs/ML', image_file.split('/')[-1]),
                      os.path.join('/home/wdxia/Finger_ROI_Database/err_imgs/ML', image_file.split('/')[-1]).split('.')[-2]+'_'+str(i)+'.jpg')
            # print(os.path.join('/home/users/wdd/myDatabase/err_imgs/ML', image_file.split('/')[-1]).split('.')[-2]+'_'+str(i)+'.jpg')


remove_bad_images()



