import numpy.random
from Scripts.metrics import *
from Scripts.preprocess import *

def data_loader_braintumor(pklroot, channels, keep_slices = None, shape =(256,256), doc = None):
    doc = pd.read_excel('Brain_tumor.xlsx')
    ages = doc['age'].values.tolist()
    ages, age_min, age_max = agelist_process(ages)

    data = pkl_load(pklroot)
    train_arr = []
    val_arr = []
    test_arr = []
    # img_zs_arr, des, label, use_type, img_arr, subname
    for d in data:
        img_arr = d[0] # (channels, slice, 192, 192)
        if img_arr.shape[0] < 3:
            print(d[4])
            continue
        des = d[1]
        des = des.strip()
        label = d[2]
        usetype = d[3] # train val test
        img_origin = d[4]
        subname = d[-1]

        image_arr, img_ori_arr = [], []
        if channels == 1:
            image_arr = img_arr[-1] # T2
            image_arr = resize_slice(image_arr, shape=shape)
            image_arr, shap = select_slice(image_arr, keep_slices)
            image_arr = image_arr[np.newaxis, :, :, :]
        else:
            for i in range(img_arr.shape[0]):
                # resize
                image = resize_slice(img_arr[i], shape=shape) #（slicenum, H, W）
                img_origin = resize_slice(img_origin[i], shape=shape)
                # slice select
                image, shap = select_slice(image,keep_slices)
                image_arr.append(image)
                image_ori, shap_ = select_slice(img_origin, keep_slices)
                img_ori_arr.append(image_ori)
        image_arr = np.array(image_arr)
        img_ori_arr = np.array(img_ori_arr)

        # age year
        age = doc.loc[doc['ID'] == int(subname.split('_')[-1]), 'age'].values[0]
        age = age_process(age, age_min, age_max)
        sex = doc.loc[doc['ID'] == int(subname.split('_')[-1]), 'sex'].values[0]
        sex = sex_process(sex)

        if usetype == 0:
            train_arr.append([image_arr, des, label, age,sex])
        elif usetype == 1:
            val_arr.append([image_arr, des, label, age,sex])
        elif usetype == 2:
            test_arr.append([image_arr, des, label, age, sex, img_ori_arr, subname])
    return train_arr, val_arr, test_arr


def braintumor_get_data(path,shape,channels, keep_slices = None, data_type = 0):

    paths = ['braintumor_T1T1CT2_resize_zscore_192.pkl']
    Xtrain, Xval, Xtest = data_loader_braintumor(path + '/' + paths[0], channels=channels, keep_slices=keep_slices,shape=shape)

    # train
    # merge & shuffle
    np.random.shuffle(Xtrain)

    # val
    # merge & shuffle
    np.random.shuffle(Xval)

    # test
    # merge & shuffle
    np.random.shuffle(Xtest)

    # split
    train_image = np.array([x[0] for x in Xtrain])
    val_image = np.array([x[0] for x in Xval])
    test_image = np.array([x[0] for x in Xtest])

    train_report = [x[1] for x in Xtrain]
    val_report = [x[1] for x in Xval]
    test_report = [x[1] for x in Xtest]

    train_Y = [x[3] for x in Xtrain]
    val_Y = [x[3] for x in Xval]
    test_Y = [x[3] for x in Xtest]

    train_age = [x[4] for x in Xtrain]
    val_age = [x[4] for x in Xval]
    test_age = [x[4] for x in Xtest]

    train_sex = [x[5] for x in Xtrain]
    val_sex = [x[5] for x in Xval]
    test_sex = [x[5] for x in Xtest]


    subname = [x[-1] for x in Xtest]
    img_origin = [x[-2] for x in Xtest]
    img_origin = np.array(img_origin)
    if data_type == 0:
        return train_image, train_report, train_Y, train_age, train_sex
    elif data_type == 1:
        return val_image, val_report, val_Y, val_age, val_sex
    elif data_type == 2:
        return test_image, test_report, test_Y, test_age, test_sex, img_origin, subname
    else:
        print('is_train value error')

