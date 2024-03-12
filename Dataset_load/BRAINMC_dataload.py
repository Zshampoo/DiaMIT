import numpy.random
from Scripts.metrics import *
from Scripts.preprocess import *

def data_loader_final(pklroot, channels, keep_slices = None, shape =(256,256)):
    # load
    data = pkl_load(pklroot)
    doc = pd.read_excel('Brain_multiclass_w_agesex.xlsx')
    ages = doc['age'].values.tolist()
    ages, age_min, age_max = agelist_process(ages)
    train_arr = []
    val_arr = []
    test_arr = []
    dict_label = {}
    for d in data:
        T2_arr = d[0]
        T1_arr_register = d[1]
        des = d[4]
        des = des.strip()
        label = d[6]
        usetype_random = d[7] # train val test
        img_origin_T2, img_origin_T1_register, img_origin_T1, img_origin_T2_register = d[9], d[10], d[11], d[12]
        time = d[-2]
        checkno = d[-1]

        # resize
        T2_arr, T1_arr_register = resize_slice(T2_arr, shape=shape), resize_slice(T1_arr_register, shape=shape) #（slicenum, H, W）
        img_origin_T2, img_origin_T1_register = resize_slice(img_origin_T2, shape=shape), resize_slice(img_origin_T1_register, shape=shape)
        # slice select
        T2_arr, _ = select_slice(T2_arr,keep_slices)
        T1_arr_register, _ = select_slice(T1_arr_register, keep_slices)
        img_origin_T2, _ = select_slice(img_origin_T2, keep_slices)
        img_origin_T1_register, _ = select_slice(img_origin_T1_register, keep_slices)
        image_ori = np.array(img_origin_T2)
        image_ori = image_ori.astype('float64')

        # age year
        age = doc.loc[doc['checkno']==int(checkno), 'age'].values[0]
        age = age_process(age, age_min, age_max)
        sex = doc.loc[doc['checkno'] == int(checkno), 'sex'].values[0]
        sex = sex_process(sex)

        image = np.concatenate([T2_arr[np.newaxis, :, :, :], T1_arr_register[np.newaxis, :, :, :], T1_arr_register[np.newaxis, :, :, :]],axis=0)

        if usetype_random == 0:
            train_arr.append([image, des, label, time, checkno, age, sex])
        elif usetype_random == 1:
            val_arr.append([image, des, label, time, age, sex, checkno, image_ori])
        elif usetype_random == 2:
            test_arr.append([image, des, label, time, age, sex])
    return train_arr, test_arr, val_arr



def brainmc_get_data(path,shape,channels, keep_slices = None, data_type = 0):

    paths = ['brainmc7000_T2-T1T2-T1-T2T1_zscore.pkl']
    Xtrain, Xval, Xtest = data_loader_final(path + '/' +paths[0], channels=channels, keep_slices=keep_slices, shape=shape)
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

    train_Y = [x[2] for x in Xtrain]
    val_Y = [x[2] for x in Xval]
    test_Y = [x[2] for x in Xtest]

    train_sex = [x[-1] for x in Xtrain]
    val_sex = [x[-1] for x in Xval]
    test_sex = [x[-3] for x in Xtest]

    train_age = [x[-2] for x in Xtrain]
    val_age = [x[-2] for x in Xval]
    test_age = [x[-4] for x in Xtest]

    img_origin = [x[-1] for x in Xtest]
    checkno = [x[-2] for x in Xtest]
    if data_type == 0:
        return train_image, train_report, train_Y, train_age, train_sex
    elif data_type == 1:
        return val_image, val_report, val_Y, val_age, val_sex
    elif data_type == 2:
        return test_image, test_report, test_Y, test_age, test_sex, img_origin, checkno
    else:
        print('is_train value error')


