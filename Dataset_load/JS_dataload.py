import numpy.random
from Scripts.preprocess import *

def agelist_process_spinal(ages):
    ageps = []
    for age in ages:
        agep = float(age)
        if agep < 0:
            agep = 0.0
        ageps.append(agep)
    return ageps, np.min(ageps), np.max(ageps)

def age_process_spinal(age, min, max):
    agep = float(age)
    if agep < 0:
        agep = 0.0
    normal_age = (agep - min) / (max- min + 1e-6)
    return normal_age

def sex_process_spinal(sex):
    if sex == 'male':
        sexp = 1.0
    else:
        sexp = 0.0
    return sexp

def data_loader_final(pklroot, cat, channels, dict_zhenduan = None,keep_slices = None, shape =(256,256)):
    # load
    doc = pd.read_excel('Spinal.xlsx').astype('str')
    ages = doc['age'].values.tolist()
    ages, age_min, age_max = agelist_process_spinal(ages)

    data = pkl_load(pklroot)
    count = 0
    train_arr = []
    val_arr = []
    test_arr = []
    # image_arr, des, diagnostic, use_type, loc, sequence, label/source, img_origin,checkno_ls
    for d in data:
        count += 1
        img_arr = d[0]
        des = d[1]
        des = des.strip()
        usetype = d[3] # train val test

        img_origin = d[-2]
        no = d[-1]

        if img_arr.shape[0]<9:
            print('{0}-{1}'.format(cat,no))
            continue

        # resize
        image = resize_slice(img_arr, shape=shape) #（slicenum, H, W）
        img_origin = resize_slice(img_origin, shape=shape)

        # slice select
        image,shap = select_slice(image,keep_slices)
        img_origin,_ = select_slice(img_origin,keep_slices)

        image = np.concatenate([image[np.newaxis, :, :, :],image[np.newaxis, :, :, :],image[np.newaxis, :, :, :]],axis=0)

        # age year
        no = str(no)
        if ('sub' not in no) and ('M' not in no):
            no = str(int(no))
        if doc.loc[doc['no'] == no, 'age'].shape[0] == 0:
            print(no)
        age = doc.loc[doc['no'] == no, 'age'].values[0]
        age_ = age_process_spinal(age, age_min, age_max)
        sex = doc.loc[doc['no'] == no, 'sex'].values[0]
        sex_ = sex_process_spinal(sex)

        if usetype == 0:
            train_arr.append([image, des, age_, sex_])
        elif usetype == 1:
            val_arr.append([image, des, age_, sex_])
        elif usetype == 2:
            test_arr.append([image, des, age_, sex_, img_origin,no])

    return train_arr, val_arr, test_arr


def JS_get_data(path,shape,channels, keep_slices = None, data_type=0):

    paths = ['sgm_T2_zscore25.pkl', 'jzl_T2_zscore23.pkl', 'xgm_T2_zscore.pkl', 'ms_T2_zscore14.pkl', 'nmo_T2_zscore18.pkl']

    Xtrain_sgm, Xval_sgm, Xtest_sgm = data_loader_final(path + '/' + paths[0], cat='sgm', channels=channels, keep_slices=keep_slices, shape=shape, dict_zhenduan=None)
    Xtrain_jzl, Xval_jzl,Xtest_jzl  = data_loader_final(path + '/' + paths[1], cat='jzl',channels=channels, keep_slices=keep_slices,shape=shape,dict_zhenduan=None)
    Xtrain_xgm, Xval_xgm, Xtest_xgm = data_loader_final(path + '/' + paths[2], cat='xgm', channels=channels,keep_slices=keep_slices, shape=shape, dict_zhenduan=None)
    Xtrain_ms, Xval_ms, Xtest_ms = data_loader_final(path + '/' + paths[3], cat='ms', channels=channels, keep_slices=keep_slices, shape=shape,dict_zhenduan=None)
    Xtrain_nmo, Xval_nmo, Xtest_nmo = data_loader_final(path + '/' + paths[4], cat='nmo',channels=channels, keep_slices=keep_slices, shape=shape,dict_zhenduan=None)

    # sgm 0
    Ytrain_sgm = [[0] for i in range(len(Xtrain_sgm))]
    Yval_sgm = [[0]  for i in range(len(Xval_sgm))]
    Ytest_sgm = [[0]  for i in range(len(Xtest_sgm))]
    # jzl 1
    Ytrain_jzl = [[1] for i in range(len(Xtrain_jzl))]
    Yval_jzl = [[1] for i in range(len(Xval_jzl))]
    Ytest_jzl = [[1] for i in range(len(Xtest_jzl))]
    # sgm 2
    Ytrain_xgm = [[2] for i in range(len(Xtrain_xgm))]
    Yval_xgm = [[2] for i in range(len(Xval_xgm))]
    Ytest_xgm = [[2] for i in range(len(Xtest_xgm))]
    # ms 3
    Ytrain_ms = [[3] for i in range(len(Xtrain_ms))]
    Yval_ms = [[3] for i in range(len(Xval_ms))]
    Ytest_ms = [[3] for i in range(len(Xtest_ms))]
    # nmo 4
    Ytrain_nmo = [[4] for i in range(len(Xtrain_nmo))]
    Yval_nmo = [[4] for i in range(len(Xval_nmo))]
    Ytest_nmo = [[4] for i in range(len(Xtest_nmo))]

    # merge & shuffle
    # train
    Xtrain_sgm.extend(Xtrain_jzl)
    Xtrain_sgm.extend(Xtrain_xgm)
    Xtrain_sgm.extend(Xtrain_ms)
    Xtrain_sgm.extend(Xtrain_nmo)

    Ytrain_sgm.extend(Ytrain_jzl)
    Ytrain_sgm.extend(Ytrain_xgm)
    Ytrain_sgm.extend(Ytrain_ms)
    Ytrain_sgm.extend(Ytrain_nmo)

    train_X = Xtrain_sgm
    train_Y = np.array(Ytrain_sgm)
    train_XY = list(zip(train_X, train_Y))
    np.random.shuffle(train_XY)
    train_X, train_Y = zip(*train_XY)
    train_X = list(train_X)
    train_Y = np.array(train_Y)

    # val
    Xval_sgm.extend(Xval_jzl)
    Xval_sgm.extend(Xval_xgm)
    Xval_sgm.extend(Xval_ms)
    Xval_sgm.extend(Xval_nmo)

    Yval_sgm.extend(Yval_jzl)
    Yval_sgm.extend(Yval_xgm)
    Yval_sgm.extend(Yval_ms)
    Yval_sgm.extend(Yval_nmo)

    val_X = Xval_sgm
    val_Y = np.array(Yval_sgm)
    val_XY = list(zip(val_X, val_Y))
    np.random.shuffle(val_XY)
    val_X, val_Y = zip(*val_XY)
    val_X = list(val_X)
    val_Y = np.array(val_Y)

    # test
    Xtest_sgm.extend(Xtest_jzl)
    Xtest_sgm.extend(Xtest_xgm)
    Xtest_sgm.extend(Xtest_ms)
    Xtest_sgm.extend(Xtest_nmo)

    Ytest_sgm.extend(Ytest_jzl)
    Ytest_sgm.extend(Ytest_xgm)
    Ytest_sgm.extend(Ytest_ms)
    Ytest_sgm.extend(Ytest_nmo)

    test_X = Xtest_sgm
    test_Y = np.array(Ytest_sgm)
    test_XY = list(zip(test_X, test_Y))
    np.random.shuffle(test_XY)
    test_X, test_Y = zip(*test_XY)
    test_X = list(test_X)
    test_Y = np.array(test_Y)

    # split
    train_image = np.array([x[0] for x in train_X])
    val_image = np.array([x[0] for x in val_X])
    test_image = np.array([x[0] for x in test_X])

    train_report = [x[1] for x in train_X]
    val_report = [x[1] for x in val_X]
    test_report = [x[1] for x in test_X]


    train_age = [x[2] for x in train_X]
    train_sex = [x[3] for x in train_X]
    val_age = [x[2] for x in val_X]
    val_sex = [x[3] for x in val_X]
    test_age = [x[2] for x in test_X]
    test_sex = [x[3] for x in test_X]

    test_img_origin = [x[4] for x in test_X]
    test_subname = [x[5] for x in test_X]
    test_img_origin = np.array(test_img_origin)

    if data_type == 0:
        return train_image, train_report,train_Y, train_age,train_sex
    elif data_type == 1:
        return val_image, val_report, val_Y, val_age, val_sex
    elif data_type == 2:
        return test_image, test_report,test_Y, test_age, test_sex, test_img_origin,test_subname
    else:
        print('is_train value error')
