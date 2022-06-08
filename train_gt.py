# -*- coding: UTF-8 -*-
import os



def list_txt(img_path,list_txt_path):
    datanames = os.listdir(img_path)
    print(datanames)
    #for i in datanames:
        #fw = open(list_txt_path, 'a', encoding="utf-8")
        #fw.write(i+'\n')
        #fw.close()
        
 


def rename(path):

    # path = 'G:/jzy/code/DB-master/datasets/icdar2017/train_gts/'

    filelist=os.listdir(path)  # 该文件夹下所有的文件（包括文件夹)
    for files in filelist:  # 遍历所有文件?
        # print(files)
        Olddir = os.path.join(path, files)    # 原来的文件路径

        if os.path.isdir(Olddir):  # 判断是否为文件夹 如果是文件夹则跳�?
            continue
        filename = os.path.splitext(files)[0]  # 将文件名和扩展名分开
        # print("------------", filename)

        filetype = os.path.splitext(files)[1]  # 文件扩展?
        # print("=================", filetype)

        if filename.find('gt_') >= 0:  # 如果文件名中含有gt_

            newdir = os.path.join(path, filename.split('gt_')[1]+filetype)

            # 取gt_前面的字符，若需要取后面的字符则使用filename.split('gt_')[1]

        if not os.path.isfile(newdir):

            os.rename(Olddir, newdir)
        
        
if __name__ == '__main__':
    path = r'/home/math-tr/data/ICDAR2017/train_gts/'
    # list_txt_path = r'/Data/home/xiashangzi2/DB-master/datasets/ICDAR2017/val_gts/'
    rename(path)
    print("finished!!!")
