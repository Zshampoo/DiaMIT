import json
import os

import cv2
import SimpleITK as sitk
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pydicom

def read_dicom(path):
    image = []
    image_path = os.listdir(path)
    image_path.sort()
    for ip in image_path:
        pa = os.path.join(path, ip)
        dcm = pydicom.read_file(pa, force=True).pixel_array
        image.append(dcm)
    image = np.array(image)
    return image

def read_nii(path):
    read_ = sitk.ReadImage(path)
    t2_arr = sitk.GetArrayFromImage(read_)
    return t2_arr

def zscore_process(arr):
    # zscore
    pixel_min = np.min(arr)
    pixel_max = np.max(arr)
    pixel_mean = np.mean(arr)
    pixel_std = np.std(arr)
    try:
        arr1 = (arr - pixel_mean) / pixel_std
    except:
        arr1 = (arr -pixel_mean) / (pixel_std + 1e-6)
    return arr1

def write_pkls(arr, path):
    with open(path, 'wb') as f:
        pickle.dump(arr, f)
        f.close()

def pkl_load(path):
    with open(path, "rb") as f:
        arr = pickle.load(f)
    return arr

def json_load(path):
    with open(path, "r", encoding="utf-8") as f:
        contents = json.load(f)
    return contents

def vis(array):
    array = array.squeeze()
    for x in array:
        plt.imshow(x)
        plt.show()

def resize_image2d(imgarr, shape):
    img_arr = cv2.resize(imgarr, (shape[0], shape[1]))
    img_arr = np.array(img_arr)
    return img_arr


def resize_slice(slices, shape):
    array = []
    for slice in slices:
        array.append(cv2.resize(slice, (shape[0], shape[1])))
    array = np.array(array)
    return array

def select_slice(image, keep_slices):
    slices_image = []
    slices_mask = []
    shape = image.shape[0]
    if shape == keep_slices:
        slices_image = image
    elif shape > keep_slices:
        delete_slices = shape - keep_slices
        if delete_slices % 2 == 0:
            start = int(delete_slices / 2)
            end = int(shape - delete_slices / 2)
        else:
            start = int(delete_slices // 2 + 1)
            end = int(shape - delete_slices // 2)
        slices_image = image[start:end, :, :]
        # print('start:{0},end:{1}'.format(start,end))
    else:
        slices_image = image
        print("slice less than",keep_slices)

    return np.array(slices_image),shape

def txt_load_new(path):
    with open(path, "rb") as f:
        txt = f.readlines()
    result = []
    for t in txt:
        temp = t.decode().replace('\n','')
        temp = temp.replace('\r', '')
        temp = temp.replace(' ', '')
        result.extend(temp)
    resultstr = ""
    for i in range(len(result)):
        resultstr += result[i]
    return resultstr


def txt_load_old(path):
    with open(path, "rb") as f:
        txt = f.readlines()
    result = []
    for t in txt:
        temp = t.decode().replace('\n','')
        temp = temp.replace('\r', '')
        temp = temp.replace(' ', '')
        temp = temp.split(',')
        result.append(temp)
    return  result


def checkno_read(path):
    with open(path, "rb") as f:
        txt = f.readlines()
    result = []
    for t in txt:
        temp = t.decode().replace('\n','')
        result.append(temp)
    return result

def txt_write(path,zhenduan):
    with open(path, 'w', encoding='utf-8') as f:
        for i in range(len(zhenduan)):
            f.write(str(zhenduan[i][0]) + ','+str(zhenduan[i][1]) + ','+str(zhenduan[i][2])+'\n')

def csv_load(path, enc='utf-8'):
    lines = pd.read_csv(path, encoding=enc)
    return lines

def report_split(report):
    sentences_fh = []
    sentences_dh = []

    report = report.replace('.','。')
    report = report.replace(',', '，')
    report = report.replace(';', '；')

    if '。' in report:
        sentences_jh = report.split('。')
    else:
        sentences_jh = report

    if isinstance(sentences_jh, list):
        if '' in sentences_jh:
            sentences_jh.remove('')

    for sentence_jh in sentences_jh:
        if '；' in sentence_jh:
            for x in sentence_jh.split('；'):
                sentences_fh.append(x)
        elif ';' in sentence_jh:
            for x in sentence_jh.split(';'):
                sentences_fh.append(x)
        else:
            sentences_fh.append(sentence_jh)
    if isinstance(sentences_fh, list):
        if '' in sentences_fh:
            sentences_fh.remove('')

    for sentence_fh in sentences_fh:
        if '，' in sentence_fh:
            for x in sentence_fh.split('，'):
                sentences_dh.append(x)
        elif ',' in sentence_fh:
            for x in sentence_fh.split(','):
                sentences_dh.append(x)
        else:
            sentences_dh.append(sentence_fh)

    if isinstance(sentences_dh, list):
        if '' in sentences_dh:
            sentences_dh.remove('')
    return sentences_dh, sentences_fh, sentences_jh

def check_sentences(sentence, descriptions):
    flag = False
    for template in descriptions:
        if template in sentence:
            flag = True
            break
    return flag

def zhenduan_select(zhenduan_origin,zhenduan_list,subname,loc):
    for i in range(len(zhenduan_list)):
        name = zhenduan_list[i][0]
        location = zhenduan_list[i][1]
        zhenduan = zhenduan_list[i][2].replace('\n','')

        if (str(subname) == name) and (str(loc) == location):
            return zhenduan
    return None

def txt_filter(str):
    string = str.replace(' ', '')
    string = string.replace('.', '。')
    string = string.replace(',', '，')
    string = string.replace(';', '；')
    string = string.replace(':', '：')
    string = string.replace('?', '？')
    string = string.replace('!', '！')
    string = string.replace('*', 'x')
    return string

def txt_replace(str, target):
    string = str.replace('.', target)
    string = string.replace('。', target)
    string = string.replace(',', target)
    string = string.replace('，', target)
    string = string.replace(';', target)
    string = string.replace('；', target)
    string = string.replace('?', target)
    string = string.replace('？', target)
    string = string.replace('!', target)
    string = string.replace('！', target)
    return string

def txt_replace_split(str, target):
    string = txt_replace(str, target)
    str_list = string.split(target)[:-1]
    return  str_list