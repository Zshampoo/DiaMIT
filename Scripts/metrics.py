import json
import os
import time
import cv2
import torch
import xlsxwriter
import numpy as np
from numpy import interp
from sklearn.model_selection import KFold
from torch import nn
from matplotlib import pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, average_precision_score, \
    precision_recall_curve, auc

from Scripts.preprocess import  report_split

Punctuations = [',', '，', '.', '。', '、', ';', '；', '“', '”', ':', '：', '!', '！', '?', '？']

def agelist_process(ages):
    ageps = []
    for age in ages:
        if 'year' in age:
            agep = float(age.replace('year',''))
        elif 'month' in age:
            agep = int(age.replace('month',''))
            agep = float(agep / 12)
        elif 'day' in age:
            agep = int(age.replace('day',''))
            agep = float(agep / 365)
        if agep < 0:
            agep = 0.0
        ageps.append(agep)
    return ageps, np.min(ageps), np.max(ageps)

def age_process(age, min, max):
    agep = -1
    if 'year' in age:
        agep = float(age.replace('year',''))
    elif 'month' in age:
        agep = int(age.replace('month',''))
        agep = float(agep / 12)
    elif 'day' in age:
        agep = int(age.replace('day',''))
        agep = float(agep / 365)
    normal_age = (agep - min) / (max- min + 1e-6)
    return normal_age

def sex_process(sex):
    if sex == 'M':
        sexp = 1.0
    else:
        sexp = 0.0
    return sexp

def auc_save(root_path, j, auc, fpr, tpr, threshold, target, predict1 ):
    path = root_path + '/test_' + str(j) + 'auc_' + str(round(auc, 4))
    if not os.path.exists(path):
        os.makedirs(path)
    target = np.array(target)
    predict1 = np.array(predict1)
    data = []
    data.append(target)
    data.append(predict1[:,1])
    with open((path + '/auc.txt'), 'w', encoding='utf-8') as f:
        for j in range(target.shape[0]):
            f.write(str(target[j]))
            for i in range(predict1.shape[-1]):
                f.write(',' + str(predict1[j, i]))
            f.write('\n')

def auc_mc_save(root_path, j, auc, target, predict1 ):
    path = root_path + '/test_' + str(j) + 'auc_' + str(round(auc, 4))
    if not os.path.exists(path):
        os.makedirs(path)
    target = np.array(target)
    predict1 = np.array(predict1)
    data = []
    data.append(target)
    for i in range(predict1.shape[-1]):
        data.append(predict1[:,i])
    with open((path + '/auc.txt'), 'w', encoding='utf-8') as f:
        for j in range(target.shape[0]):
            f.write(str(np.argmax(target[j])))
            for i in range(predict1.shape[-1]):
                f.write(',' + str(predict1[j,i]))
            f.write('\n')

def acc_binary(pred, label):
    len = pred.shape[0]
    class_num = pred.shape[1]
    count = 0
    class_counts = []
    pred_arr = np.array([np.argmax(i) for i in pred.to('cpu')])
    label_arr = np.array([np.argmax(i) for i in label.to('cpu')])

    for i in range(len):
        if pred_arr[i] == label_arr[i]:
            count += 1

    for j in range(class_num):
        class_counts.append(np.sum(label_arr == j))

    acc = np.float64(count/len)
    # cpu
    acc =np.float64(count / len)

    return acc, count, len, class_counts

def acc_mc(pred, label):
    len = pred.shape[0]
    class_num = pred.shape[1]
    count = 0
    class_counts = []
    pred_arr = np.array([np.argmax(i) for i in pred.to('cpu')])
    label_arr  = np.array([i for i in label.to('cpu')])

    for i in range(len):
        if pred_arr[i]==label_arr[i]:
            count +=1

    for j in range(class_num):
        class_counts.append(np.sum(label_arr == j))

    acc = np.float64(count / len)

    return acc, count, len, class_counts

def fm_binary(pred, label, target_names):
    label = np.array([np.argmax(i) for i in label])
    predict = np.array([np.argmax(i) for i in pred])
    lab = torch.Tensor(label)
    pre = torch.Tensor(predict)
    pre_ = pre.type(torch.IntTensor)
    cf_martix = confusion_matrix(lab, pre_)
    cf_result = classification_report(label, predict, target_names = target_names)

    print(cf_martix)
    print(cf_result)

    return cf_result, cf_martix


def fm_mc(pred, label, target_names):

    pred_arr = np.array([np.argmax(i) for i in pred])
    cf_martix = confusion_matrix(label, pred_arr)
    cf_result = classification_report(label, pred_arr, target_names=target_names)

    print(cf_martix)
    print(cf_result)

    return cf_result, cf_martix

def auc_cal_binary (targ, pred, target_name, is_draw, is_save, ck_path =None):

    predict = torch.Tensor(pred)
    predict_  = (nn.Softmax(dim=-1)(predict)).numpy()
    target_ = [np.argmax(i) for i in targ]
    target_ = np.array(target_)

    colors = ['red', 'yellow', 'green', 'blue']
    auc_ = roc_auc_score(target_, predict_[:,1])
    fpr, tpr, threshold = roc_curve(target_, predict_[:, 1])
    if is_draw:
        plt.figure()
        plt.plot(fpr, tpr, 'k-', color=colors[0], label=u' (AUC = {:.4f}) '.format(auc_))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve of {0} vs {1}'.format(target_name[0], target_name[1]))
        plt.show()
    if is_save:
        ck_path_ = ck_path.split('/')[:-1]
        ckpath_str = ""
        for s in ck_path_: ckpath_str += s + '/'
        with open(ckpath_str+'auc.txt', 'w', encoding='utf-8') as f:
            f.write('target predict:\n')
            for i in range(len(target_)):
                f.write(str(target_[i])+','+ str(predict_[i, 0])+','+ str(predict_[i, 1])+'\n')
            f.write('auc:'+ str(auc_))

    print('auc:', auc_)
    return auc_, fpr, tpr, threshold, target_, predict_

def auc_mc(target, pred, target_name, is_draw, is_save, ck_path =None):
    predict = torch.Tensor(pred)
    predict_ = (nn.Softmax(dim=-1)(predict)).numpy()
    class_num = predict.shape[1]
    classes = [i for i in range(class_num)]
    target_binary = label_binarize(target, classes=classes)
    aucs,fprs, tprs, thresholds = [], [], [],[]
    colors = ['red','yellow','green','blue']

    for j in range(class_num):
        auc_ = roc_auc_score(target_binary[:,j], predict_[:,j], average=None)
        aucs.append(auc_)
        fpr, tpr, threshold = roc_curve(target_binary[:, j], predict_[:, j])
        fprs.append(fpr)
        tprs.append(tpr)
        thresholds.append(threshold)
    aucovr = roc_auc_score(target_binary, predict_, multi_class='ovr')
    aucovo = roc_auc_score(target_binary, predict_, multi_class='ovo')

    # micro
    fpr_dict, tpr_dict, auc_dict = {}, {}, {}
    fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(target_binary.ravel(), predict_.ravel())
    auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])
    #  macro
    all_fpr = np.unique(np.concatenate([fprs[i] for i in range(class_num)]))  # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(class_num): mean_tpr += interp(all_fpr, fprs[i], tprs[i])  # Finally average it and compute AUC
    mean_tpr /= class_num
    fpr_dict["macro"] = all_fpr
    tpr_dict["macro"] = mean_tpr
    auc_dict["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])  # Plot all ROC curves
    if is_draw:
        plt.figure()
        for i in range(class_num):
            plt.plot(fprs[i], tprs[i], 'k-', color=colors[i], label=u'{} (area = {:.4f})'.format(target_name[i], aucs[i]))

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve of {0}'.format(target_name))
        plt.show()
    if is_save:
        ck_path_ = ck_path.split('/')[:-1]
        ckpath_str = ""
        for s in ck_path_: ckpath_str += s + '/'
        with open(ckpath_str + 'auc.txt', 'w', encoding='utf-8') as f:
            f.write('target predict:\n')
            for i in range(len(target_binary)):
                f.write(str(target_binary[i]))
                for j in range(class_num):
                    f.write(','+str(predict_[i, j]))
                f.write('\n')

            f.write('auc:' + str(np.mean(aucs,axis=0)))

    print('avg auc of {0} classes: {1}'.format(class_num, np.mean(aucs,axis=0)))

    return aucs, np.mean(aucs,axis=0), fpr_dict, tpr_dict, auc_dict, thresholds, target_binary, predict_

def auprc_binary (targ, pred, target_name, is_draw):

    predict = torch.Tensor(pred)
    predict_  = (nn.Softmax(dim=-1)(predict)).numpy()
    target_ = [np.argmax(i) for i in targ]
    target_ = np.array(target_)

    colors = ['red','yellow','green','blue','gold','gray','orange','seagreen']
    precision, recall, thresholds = precision_recall_curve(target_, predict_[:, 1])
    auprc = auc(recall, precision)

    if is_draw:
        plt.figure()
        plt.plot(precision, recall, 'k-', color=colors[0], label='AUPRC = {.3f}'.format(auprc))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.show()
    print('auprc:', auprc)
    return auprc

def auprc_mc(target, pred, target_name, is_draw):
    class_num = pred.shape[1]
    classes = [i for i in range(class_num)]
    target_binary = label_binarize(target, classes=classes)

    predict = torch.Tensor(pred)
    predict_ = (nn.Softmax(dim=-1)(predict)).numpy()

    auprcs = []
    colors = ['red','yellow','green','blue','gold','gray','orange','seagreen']

    for j in range(class_num):
        precision, recall, thresholds = precision_recall_curve(target_binary[:,j], predict_[:,j])
        auprc = auc(recall, precision)
        auprcs.append(auprc)
        if is_draw:
            plt.figure()
            plt.plot(precision, recall, 'k-', color=colors[j], label='{} (AUPRC = {.3f})'.format(target_name[j], auprc))

            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.show()
    print('avg auprc of {0} classes: {1}'.format(class_num, np.mean(auprcs,axis=0)))

    return auprcs, np.mean(auprcs,axis=0)

def count_add (count_list, step):
    for i in range(len(count_list)):
        count_list[i] += step[i]
    return count_list

def check_report(sentences):
    report = ''
    for s in sentences:
        report = report + s+'[SEP]'
    return  report

def save_checkpoint(ck_root, md_param, repeat_exp, epoch, state, modelpath):
    model_type = md_param['image_aggregation_type']
    lr = md_param['lr']
    save_type = md_param['save_model_type']

    path = ck_root + '/EX{0}_modeltype_{1}_lr{2}_savetype{3}_time{4}'.format(str(repeat_exp), model_type, lr, save_type, time.time())
    if not os.path.exists(path):
        os.makedirs(path)
    with open((path + '/parameters.txt'), 'w', encoding='utf-8') as f:
        for k in md_param.keys():
            f.write('{0} : {1}\n'.format(k, md_param[k]))
    path_ = os.path.join(path, modelpath)
    torch.save(state, path_)
    return path_

def save_test_results(writepath, zs,xls,contris_all, xrs, slice_attsall, img_origins, subnames, region_atts=None):
    path = writepath + '/TEST_RESULTS_DATA'
    if not os.path.exists(path):
        os.makedirs(path)
    data  = {}
    for i in range(len(zs)):
        no = subnames[i]
        data[no] = {}
        data[no]['zs'] = zs[i]
        data[no]['xls'] = xls[i]
        if len(contris_all) != 0:
            data[no]['contris_all'] = contris_all[i]
        if len(region_atts) != 0:
            data[no]['region_attention'] = region_atts[i]
        if len(region_atts) != 0:
            data[no]['slice_attention'] = slice_attsall[i]
        data[no]['xrs'] = xrs[i]
        data[no]['img_origin'] = img_origins[i]

    jsondata = json.dumps(data)

    with open((path + '/data.json'), 'w', encoding='utf-8') as f:
        f.write(jsondata)

def save_IRENE_results(writepath, zs,xls, xrs, slice_attsall, img_origins, subnames):
    path = writepath + '/TEST_RESULTS_DATA'
    if not os.path.exists(path):
        os.makedirs(path)
    data, att_layers = {}, {}

    layer_num = 12
    for j in range(layer_num):
        # layername = 'layer' + str(j)
        data_layer = []
        for i in range(len(slice_attsall)):
            element = slice_attsall[i][j].cpu().numpy() # (16,12, x, x)
            data_layer.extend(element)
        data_layer = np.array(data_layer).tolist() # (48,12,x,x)
        data_layer_avg = np.average(data_layer, axis=1)
        att_layers[j] = data_layer_avg.tolist()

    for i in range(len(zs)):
        no = subnames[i]
        data[no] = {}
        data[no]['zs'] = zs[i]
        data[no]['xls'] = xls[i]
        data[no]['xrs'] = xrs[i]
        data[no]['layer_atts0'] = att_layers[0][i]
        data[no]['layer_atts1'] = att_layers[1][i]
        data[no]['layer_atts2'] = att_layers[2][i]
        data[no]['layer_atts3'] = att_layers[3][i]
        data[no]['layer_atts4'] = att_layers[4][i]
        data[no]['layer_atts5'] = att_layers[5][i]
        data[no]['layer_atts6'] = att_layers[6][i]
        data[no]['layer_atts7'] = att_layers[7][i]
        data[no]['layer_atts8'] = att_layers[8][i]
        data[no]['layer_atts9'] = att_layers[9][i]
        data[no]['layer_atts10'] = att_layers[10][i]
        data[no]['layer_atts11'] = att_layers[11][i]
        data[no]['img_origin'] = img_origins[i]
    att_layers = {}
    jsondata = json.dumps(data)
    with open((path + '/data_irene.json'), 'w', encoding='utf-8') as f:
        f.write(jsondata)

def save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)

def check_sentences(sentence, descriptions):
    flag = False
    for template in descriptions:
        if template in sentence:
            flag = True
            break
    return flag

def sentence_token_count(reports):
    reports_lens = []
    reports = list(reports)
    for report in reports:
        report_lens = []
        _,sentences_fh,_ = report_split(report)
        lens = [len(x) for x in sentences_fh]
        last = 0
        for i in range(len(lens)):
                now = last +lens[i]+1
                last = now
                report_lens.append(now)
        reports_lens.append(report_lens)
    return  reports_lens

def result_to_file(ckpath, val_epochs, val_accs, val_aucs,test_losses, test_accs, test_auc_mcs,test_aucs,test_cfms, test_cfrs, EPOCHTIMES):
    if not os.path.exists(ckpath):
        os.makedirs(ckpath)
    excle1 = xlsxwriter.Workbook(ckpath+ '/result.xlsx'.format())
    repeat_counts = len(val_epochs)
    for i in range(repeat_counts):
        worksheet = excle1.add_worksheet(name='EX{0}'.format(i))
        
        worksheet.write(1, 1, 'val_epochs')
        worksheet.write(1, 2, val_epochs[i])
        worksheet.write(2, 1, 'val_accs')
        worksheet.write(2, 2, val_accs[i])
        worksheet.write(3, 1, 'val_aucs')
        worksheet.write(3, 2, val_aucs[i])
        worksheet.write(4, 1, 'test_losses')
        worksheet.write(4, 2, test_losses[i])
        worksheet.write(5, 1, 'test_accs')
        worksheet.write(5, 2, test_accs[i])

        worksheet.write(6, 1, 'test_auc_mcs')
        for j1 in range(len(test_auc_mcs[i])):
            worksheet.write(6, j1+2, test_auc_mcs[i][j1])

        worksheet.write(7, 1, 'test_aucs')
        worksheet.write(7, 2, test_aucs[i])

        worksheet.write(8, 1, 'test_cfms')
        for j2 in range(test_cfms[i].shape[0]):
            for k2 in range(test_cfms[i].shape[1]):
                worksheet.write(9+j2, 1+k2, test_cfms[i][j2][k2])

        worksheet.write(10 + test_cfms[i].shape[0] , 1, 'test_cfrs')
        worksheet.write(11 + test_cfms[i].shape[0], 1, test_cfrs[i])


        worksheet.write(12 + test_cfms[i].shape[0], 1, 'EPOCHTIMES')
        worksheet.write(12 +test_cfms[i].shape[0], 2, EPOCHTIMES[i])
    excle1.close()

def dict_append(dict, data):
    if data not in dict.keys():
        dict[data] = 1
    else:
        dict[data] += 1
    return dict

def dict_len_append(dict, data):
    if len(data) not in dict.keys():
        dict[len(data)] = 1
    else:
        dict[len(data)] += 1
    return  dict

def Kfold_split(N, all_arr, fold_index):
    kf = KFold(n_splits=N, shuffle=True)
    train_arrs, val_arrs, test_arrs = [], [], []
    for trainval_index, test_index in kf.split(all_arr):
        trainval_index = trainval_index.tolist()
        test_index = test_index.tolist()
        val_index, train_index = trainval_index[ : len(test_index)], trainval_index[len(test_index):]
        train_arr, val_arr, test_arr = [], [], []
        for ind_train in train_index:
            train_arr.append(all_arr[ind_train])
        for ind_val in val_index:
            val_arr.append(all_arr[ind_val])
        for ind_test in test_index:
            test_arr.append(all_arr[ind_test])

        train_arrs.append(train_arr)
        val_arrs.append(val_arr)
        test_arrs.append(test_arr)
    trainarr, valarr, testarr = train_arrs[fold_index], val_arrs[fold_index], test_arrs[fold_index]
    return trainarr, valarr, testarr

