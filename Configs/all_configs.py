from Models.IRENE.modeling_irene import CONFIGS
img_baselines, txt_baselines, mm_baselines = ['AVG', '3D'], ['CLS'], ['IRENE', 'DIAMIT', 'AVG_dem']

def get_spinal_configs(modeltype = 'MM', subtype = None):

    PADDING = False if modeltype == 'MM' else True

    PREDICT_TYPE = None
    if subtype in img_baselines:
        PREDICT_TYPE = 'img_only'
    elif subtype in txt_baselines:
        PREDICT_TYPE = 'rep_only'
    elif subtype in mm_baselines:
        PREDICT_TYPE = 'img+rep'
    else:
        print('subtype error')

    DEVICE = "cuda:0"

    model_parameters = {
        'MODEL': modeltype,
        'irene_config': CONFIGS["IRENE"],
        'padding': PADDING,
        'epochs': 50,
        'batchsize': 32,
        'dataset': 'spinal',
        'path': 'DATA/SPINAL',
        'keep_slices': 9,
        'W': 224,
        'H': 224,
        'D': 9,
        'train_droplast': False,
        'val_droplast': False,
        'test_droplast': False,
        'checkpoint_path': 'CHECKPOINT/SPINAL',
        'shape': (224, 224),
        'classes': 5,
        'class_names': ['SE', 'SCA', 'SHB', 'MS', 'NMOSD'],
        'report_max_length': 128,
        'image_model_name': "resnet18",
        'report_model_name': 'chinese-roberta-wwm-ext',
        'image_aggregation_type': subtype,
        'mm_dim': 128,
        'bias': False,
        'channels': 3,
        'freeze_layers': [0, 1, 2, 3, 4, 5],
        'predict_type': PREDICT_TYPE,
        'fusion_method': 'proj',
        'label_squeeze': True,
        'draw_roc': True,
        'device': DEVICE,
        'repeat_counts': 5,
        'lr': 1e-5,
        'lr_warmup': 1e-5,
        'warmup_epochs': 0,
        'lr_reduce_factor': 0.5,
        'lr_reduce_patience': 10,
        'weight_decay': 1e-6,
        'val_epoch': 1,
        'save_epoch': -1,
        'save_model_type': 'acc_loss',
        'report_weights_path': None,
        'pretrained_weights_path': None,
        'image_weights_path': None,
    }
    return model_parameters

def get_braintumor_configs(modeltype='MM', subtype=None):
    PADDING = False if modeltype == 'MM' else True

    PREDICT_TYPE = None
    if subtype in img_baselines:
        PREDICT_TYPE = 'img_only'
    elif subtype in txt_baselines:
        PREDICT_TYPE = 'rep_only'
    elif subtype in mm_baselines:
        PREDICT_TYPE = 'img+rep'
    else:
        print('subtype error')

    DEVICE = "cuda:1"

    model_parameters = {
        'MODEL': modeltype,  # 'MM', 'IRENE'
        'irene_config': CONFIGS["IRENE"],
        'padding': PADDING,
        'epochs': 30,
        'batchsize': 32,
        'dataset': 'braintumor',
        'path': 'DATA/BRAINTUMOR',
        'keep_slices': 22,
        'W': 192,
        'H': 192,
        'D': 22,
        'train_droplast': False,
        'val_droplast': False,
        'test_droplast': False,
        'checkpoint_path': 'CHECKPOINT/BRAINTUMOR',
        'shape': (192, 192),
        'classes': 3,
        'class_names': ['GBM', 'PCNSL', 'BM'],
        'report_max_length': 128,
        'kd_max_length': 128,
        'image_model_name': "resnet18",
        'report_model_name': 'chinese-roberta-wwm-ext',
        'image_aggregation_type': subtype,
        'mm_dim': 128,
        'bias': False,
        'channels': 3,
        'freeze_layers': [0, 1, 2, 3, 4, 5],
        'predict_type': PREDICT_TYPE,
        'fusion_method': 'proj',
        'label_squeeze': False,
        'draw_roc': True,
        'device': DEVICE,
        'repeat_counts': 5,
        'lr': 1e-5,
        'lr_warmup': 1e-5,
        'warmup_epochs': 0,
        'lr_reduce_factor': 0.5,
        'lr_reduce_patience': 10,
        'weight_decay': 1e-6,
        'val_epoch': 1,
        'save_epoch': -1,
        'save_model_type': 'acc_loss',
        'report_weights_path': None,
        'pretrained_weights_path': None,
        'image_weights_path': None,
    }
    return model_parameters

def get_brainidh_configs(modeltype='MM', subtype=None):
    PADDING = True if modeltype == 'MM' else False

    PREDICT_TYPE = None
    if subtype in img_baselines:
        PREDICT_TYPE = 'img_only'
    elif subtype in txt_baselines:
        PREDICT_TYPE = 'rep_only'
    elif subtype in mm_baselines:
        PREDICT_TYPE = 'img+rep'
    else:
        print('subtype error')

    DEVICE = "cuda:1"

    model_parameters = {
        'MODEL': modeltype,  # 'MM', 'IRENE'
        'irene_config': CONFIGS["IRENE"],
        'padding': PADDING,
        'epochs': 30,
        'batchsize': 32,
        'dataset': 'brainidh',
        'path': 'DATA/BRAINIDH',
        'keep_slices': 23,
        'W': 128,
        'H': 128,
        'D': 23,
        'train_droplast': False,
        'val_droplast': False,
        'test_droplast': False,
        'checkpoint_path': 'CHECKPOINT/BRAINIDH',
        'shape': (128, 128),
        'classes': 2,
        'class_names': ['WT', 'MUT'],
        'report_max_length': 224,
        'kd_max_length': 128,
        'image_model_name': "resnet18",
        'report_model_name': 'chinese-roberta-wwm-ext',
        'image_aggregation_type': subtype,
        'mm_dim': 128,
        'mhsa_dim': 512,
        'mhsa_heads': 8,
        'dropout_rate': 0,
        'mask_columns': None,
        'bias': False,
        'channels': 3,
        'freeze_layers': [0, 1, 2, 3, 4, 5],
        'predict_type': PREDICT_TYPE,
        'fusion_method': 'proj',
        'label_squeeze': False,
        'draw_roc': True,
        'device': DEVICE,
        'repeat_counts': 5,
        'lr': 1e-5,
        'lr_warmup': 1e-5,
        'warmup_epochs': 0,
        'lr_reduce_factor': 0.5,
        'lr_reduce_patience': 10,

        'weight_decay': 1e-6,
        'val_epoch': 1,
        'save_epoch': -1,
        'save_model_type': 'acc_loss',
        'report_weights_path': None,
        'pretrained_weights_path': None,
        'image_weights_path': None,
    }
    return model_parameters

def get_mra_configs(modeltype = 'MM', subtype = None):

    PADDING = True if modeltype == 'MM' else False

    PREDICT_TYPE = None
    if subtype in img_baselines:
        PREDICT_TYPE = 'img_only'
    elif subtype in txt_baselines:
        PREDICT_TYPE = 'rep_only'
    elif subtype in mm_baselines:
        PREDICT_TYPE = 'img+rep'
    else:
        print('subtype error')

    DEVICE = "cuda:1"

    model_parameters = {
        'MODEL': modeltype,
        'irene_config': CONFIGS["IRENE"],
        'padding': PADDING,
        'epochs': 50,
        'batchsize': 32,
        'dataset': 'mra',
        'path': 'DATA/MRA',
        'keep_slices': 12,
        'W': 192,
        'H': 192,
        'D': 12,
        'train_droplast': False,
        'val_droplast': False,
        'test_droplast': False,
        'checkpoint_path': 'CHECKPOINT/MRA',
        'shape': (192, 192),
        'classes': 5,
        'class_names': ['NC', 'AVM', 'MMD', 'IA', 'ICAS'],
        'report_max_length': 128,
        'kd_max_length': 128,
        'image_model_name': "resnet18",
        'report_model_name': 'chinese-roberta-wwm-ext',
        'image_aggregation_type': subtype,
        'mm_dim': 256,
        'bias': False,
        'channels': 3,
        'freeze_layers': [0, 1, 2, 3, 4, 5],
        'predict_type': PREDICT_TYPE,
        'fusion_method': 'proj',
        'label_squeeze': False,
        'draw_roc': True,
        'device': DEVICE,
        'repeat_counts': 5,
        'lr': 5e-6,
        'lr_warmup': 5e-6,
        'warmup_epochs': 0,
        'lr_reduce_factor': 0.5,
        'lr_reduce_patience': 10,
        'weight_decay': 1e-6,
        'val_epoch': 1,
        'save_epoch': -1,
        'save_model_type': 'acc_loss',
        'report_weights_path': None,
        'pretrained_weights_path': None,
        'image_weights_path': None,
    }
    return model_parameters

def get_brainmc_configs(modeltype = 'MM', subtype = None):

    PADDING = True if modeltype == 'MM' else False

    PREDICT_TYPE = None
    if subtype in img_baselines:
        PREDICT_TYPE = 'img_only'
    elif subtype in txt_baselines:
        PREDICT_TYPE = 'rep_only'
    elif subtype in mm_baselines:
        PREDICT_TYPE = 'img+rep'
    else:
        print('subtype error')

    DEVICE = "cuda:1"

    model_parameters = {
        'MODEL': modeltype,
        'irene_config': CONFIGS["IRENE"],
        'padding': PADDING,
        'epochs': 50,
        'batchsize': 32,
        'dataset': 'brainmc',
        'path': 'DATA/BRAINMC',
        'keep_slices': 23,
        'W': 128,
        'H': 128,
        'D': 23,
        'train_droplast': False,
        'val_droplast': False,
        'test_droplast': False,
        'checkpoint_path': 'CHECKPOINT/BRAINMC',
        'shape': (128, 128),
        'classes': 7,
        'class_names': [ 'NDD', 'TBI', 'CVD', 'TME', 'BT', 'NID', 'NC'],
        'report_max_length': 224,
        'kd_max_length': 128,
        'image_model_name': "resnet18",
        'report_model_name': 'chinese-roberta-wwm-ext',
        'image_aggregation_type': subtype,
        'mm_dim': 256,
        'bias': False,
        'channels': 3,
        'freeze_layers': [0, 1, 2, 3, 4, 5],
        'predict_type': PREDICT_TYPE,
        'fusion_method': 'proj',
        'label_squeeze': False,
        'draw_roc': True,
        'device': DEVICE,
        'repeat_counts': 1,
        'lr': 5e-6,
        'lr_warmup': 5e-6,
        'warmup_epochs': 0,
        'lr_reduce_factor': 0.5,
        'lr_reduce_patience': 10,

        'weight_decay': 1e-6,
        'val_epoch': 1,
        'save_epoch': -1,
        'save_model_type': 'acc_loss',
        'report_weights_path': None,
        'pretrained_weights_path': None,
        'image_weights_path': None,
    }
    return model_parameters