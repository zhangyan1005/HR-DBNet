# -*- coding: UTF-8 -*-
import os




def list_txt(img_path,list_txt_path):
    datanames = os.listdir(img_path)
    fw = open(list_txt_path, 'a', encoding="utf-8")
    for i in datanames:
        fw.write(i+'\n')
        #fw.close()



if __name__ == '__main__':
    img_path = r'/home/math-tr/data/MSRA_TD500/test_im/'
    list_txt_path = r'/home/math-tr/data/MSRA_TD500/test_list.txt'
    list_txt(img_path, list_txt_path)
    print("finished!!!")