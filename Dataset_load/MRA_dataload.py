import numpy.random
from Scripts.metrics import *
from Scripts.preprocess import *


def agelist_process_mra(ages):
    ageps = []
    for age in ages:
        if 'Y' in age:
            agep = int(age.replace('Y',''))
            agep = np.float64(agep)
        elif 'M' in age:
            agep = int(age.replace('M',''))
            agep = np.float64(agep / 12)
        elif 'W' in age:
            agep = int(age.replace('W',''))
            agep = np.float64(agep * 7 / 365)
        if agep < 0:
            agep = 0.0
        ageps.append(agep)
    return ageps, np.min(ageps), np.max(ageps)

def age_process_mra(age, min, max):
    agep = -1
    if 'Y' in age:
        agep = float(age.replace('Y',''))
    elif 'M' in age:
        agep = int(age.replace('M',''))
        agep = float(agep / 12)
    elif 'W' in age:
        agep = int(age.replace('W',''))
        agep = float(agep * 7 / 365)
    normal_age = (agep - min) / (max- min + 1e-6)
    return normal_age

def data_loader_final(pklroot, channels, keep_slices = None, shape =(256,256), doc=None):
    # load
    doc = pd.read_excel('Brain_mra.xlsx')
    ages = doc.loc[doc['available'] == 1, 'age'].values.tolist()
    ages, age_min, age_max = agelist_process_mra(ages)

    data = pkl_load(pklroot)
    train_arr = []
    val_arr = []
    test_arr = []
    # image_arr, des, zhenduan, use_type, loc, T2_type, source,img_origin,no
    for d in data:
        img_arr = d[0]
        des = d[1]
        label = d[2]
        usetype = d[3] # train val test

        img_origin = d[-2]
        no = d[-1]

        # resize
        image = resize_slice(img_arr, shape=shape) #（slicenum, H, W）
        img_origin = resize_slice(img_origin, shape=(256,256))
        # slice select
        image, shap = select_slice(image,keep_slices)
        image_ori, shap_ = select_slice(img_origin, keep_slices)


        image = np.concatenate([image[np.newaxis, :, :, :],image[np.newaxis, :, :, :],image[np.newaxis, :, :, :]],axis=0)

        # age year
        age = doc.loc[doc['chechno'] == int(no), 'age'].values[0]
        age = age_process_mra(age, age_min, age_max)
        sex = doc.loc[doc['chechno'] == int(no), 'sex'].values[0]
        sex = sex_process(sex)

        if usetype == 0:
            train_arr.append([image, des,label,age, sex])
        elif usetype == 1:
            val_arr.append([image, des, label,age, sex])
        elif usetype == 2:
            test_arr.append([image, des, label, age, sex, image_ori, no])

    return train_arr, val_arr, test_arr


def mra_get_data(path,shape,channels, keep_slices = None, data_type = 0):

    paths = ['mra_zscore.pkl']
    doc = pd.read_excel('all_mra.xlsx', engine='openpyxl')
    Xtrain, Xval, Xtest = data_loader_final(path + '/' +paths[0], channels=channels, keep_slices=keep_slices, shape=shape, doc = doc)

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

    train_age = [x[3] for x in Xtrain]
    val_age = [x[3] for x in Xval]
    test_age = [x[3] for x in Xtest]

    train_sex = [x[4] for x in Xtrain]
    val_sex = [x[4] for x in Xval]
    test_sex = [x[4] for x in Xtest]



    img_origin = [x[-2] for x in Xtest]
    no = [x[-1] for x in Xtest]
    img_origin = np.array(img_origin)

    if data_type == 0:
        return train_image, train_report, train_Y, train_age, train_sex
    elif data_type == 1:
        return val_image, val_report, val_Y, val_age, val_sex
    elif data_type == 2:
        return test_image, test_report, test_Y,test_age, test_sex, img_origin, no
    else:
        print('is_train value error')

