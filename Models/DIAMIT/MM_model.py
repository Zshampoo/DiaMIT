import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import AutoModel,BertModel
import torchvision.models as models
from Models.DIAMIT.Attentions import Attention
from Models.RESNET3D.Resnet3D_from_MED3D import *
from Models.ViT.vit_model import vit_base_patch32_224_in21k as create_model

ROBERTA_PATH = '/ROBERTA/roberta_config'
BERT_PATH = '/BERT'

class MMmodel(nn.Module):
    def __init__(self,
                 image_model_name = 'resnet18',
                 report_model_name = 'chinese-roberta-wwm-ext', # 'bert-base-chinese', 'chinese-roberta-wwm-ext'
                 img_aggregation_type = 'DIAMIT', # 'AVG', 'AVG_DEM' '3D', 'DIAMIT', 'IRENE'
                 img_weight_path = None,
                 input_W = 256, # width of slice
                 input_H = 256, # height of slice
                 input_D = 9, # slice number
                 multimodal_dim = 128,
                 mhsa_dim = 512,
                 mhsa_heads = 8,
                 dropout_rate = 0.1,
                 mask_columns = 2,
                 bias = False,
                 channels = 3,
                 nb_class = 2,
                 freeze_layers = None,
                 predict_type = 'img+rep', # 'img', 'rep', 'img+rep'
                 fusion_method = 'concate', # 'concate', 'proj'
                 device = "cuda:0",
                 w_agesex = False
                 ):

        super(MMmodel, self).__init__()

        self.device = device
        self.predict = predict_type
        self.type = img_aggregation_type
        self.w_agesex = w_agesex
        # init image encoder
        if freeze_layers is None:
            freeze_layers = [0, 1, 2, 3, 4, 5]
        self.Img_model = ImageEncoder(image_model = image_model_name,
                                     aggregation_type = img_aggregation_type,
                                     image_weight_path = img_weight_path,
                                     H = input_H,
                                     W = input_W,
                                     D = input_D,
                                     channels = channels,
                                     mm_dim = multimodal_dim,
                                     mhsa_dim = mhsa_dim,
                                     num_class = nb_class,
                                     num_heads = mhsa_heads,
                                     bias = bias,
                                     dropout_rate = dropout_rate,
                                     mask_columns = mask_columns,
                                     device = self.device
                                     )

        # init report encoder
        self.Rep_model = RepEncoder(rep_model=report_model_name,freeze_layers=freeze_layers,device = self.device)

        # init fusion and prediction model
        self.Predict_model = Classifier(img_outputdim = self.Img_model._get_img_dim(),
                                        rep_output_dim = 768,
                                        multimodal_dim = multimodal_dim,
                                        bias = bias,
                                        num_class=nb_class,
                                        predict_type=predict_type,
                                        fusion_method=fusion_method,
                                        device = self.device)

    def forward(self, xis, xrs_encoded_inputs, xage = None, xsex= None):
        '''
        xis: input image (batchsize, C, Slice, H, W)
        xrs_encoded_inputs: report after tokenizing
        '''
        # Encoding
        xre, xie, xde, slice_scores, region_atts = None, None, None, None, None
        ## REP
        if self.predict == 'img_only':
            xTe = None
        else:
            xcls, xmpe, _ = self.Rep_model(xrs_encoded_inputs)
            xTe = xcls
        ## IMG
        if self.predict == 'rep_only':
            pass
        else:
            xie, slice_scores = self.Img_model(xis, xr_slice = xTe)

        z, contris_ = self.Predict_model(xie, xTe, xage, xsex)

        return z, slice_scores, contris_
    
# Image Encoder
class ImageEncoder(nn.Module):
    def __init__(self,
                 image_model,
                 aggregation_type,
                 image_weight_path,
                 H,
                 W,
                 D,
                 channels,
                 mm_dim,
                 mhsa_dim,
                 num_class,
                 num_heads,
                 bias,
                 dropout_rate,
                 mask_columns,
                 device):
        super(ImageEncoder, self).__init__()
        self.device_img = device
        # init Resnet
        self.aggregation = aggregation_type
        self.H = H
        self.W = W
        self.slice_num = D
        self.channels = channels
        self.mm_dim = mm_dim
        self.mhsa_dim = mhsa_dim
        self.num_class = num_class
        self.num_heads = num_heads
        self.bias = bias
        self.dropout = dropout_rate
        self.mask_columns = mask_columns

        model_, fc_input = self._get_res_basemodel(image_model, aggregation_type, image_weight_path, H, W, D, channels)
        if aggregation_type == '3D':
            self.img_encoder = nn.Sequential(*list(model_.children())[:-1])  # drop FC
        else:
            self.resnet_model_1 = nn.Sequential(*list(model_.children())[:-1]) #  drop FC
            self.img_encoder = nn.Sequential(*list(model_.children())[:-2]) #  drop FC and avgpool
            if not (self.channels == self.img_encoder[0].in_channels):
                self.img_encoder[0] = nn.Conv2d(self.channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.fc_input = fc_input

        # REM
        # RSA_COSINE
        self.Proj_REP_cs = nn.Linear(768, self.mm_dim, bias=self.bias)
        self.Proj_SLICE_cs = nn.Linear(self.fc_input, self.mm_dim, bias=self.bias)


    def _get_img_dim(self):
        return self.fc_input

    def _get_res_basemodel(self, image_model, aggregation_type, image_weights_path, H, W, D, channels):
        # backbone
        if aggregation_type == '3D':
            model = resnet18(sample_input_W = W,
                             sample_input_H = H,
                             sample_input_D = D,
                             channels = channels,
                             shortcut_type = 'A',
                             no_cuda = False,
                             num_seg_classes=1)
            model = model.to(self.device_img)
        else:
            self.resnet_dict = {"resnet18": models.resnet18(pretrained=True),
                                "resnet34": models.resnet34(pretrained=True),
                                "resnet50": models.resnet50(pretrained=True),
                                "resnet101": models.resnet101(pretrained=True)}
            model = self.resnet_dict[image_model]
            if channels == 1:
                model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        print("Image feature extractor: {0}, aggregation type: {1}".format(image_model,aggregation_type))

        net_dict = model.state_dict()

        if image_weights_path != None:
            print('loading pretrained model from{0}'.format(image_weights_path))
            pretrain = torch.load(image_weights_path)
            pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
            # k 是每一层的名称，v是权重数值
            net_dict.update(pretrain_dict)  # 字典 dict2 的键/值对更新到 dict 里。
            model.load_state_dict(net_dict)  # model.load_state_dict()函数把加载的权重复制到模型的权重中去
            print("--------Image pre-train model load successfully --------")
        if image_model == "resnet18" or image_model == "resnet34":
            fc_input = 512
        else:
            fc_input = 2048
        return model, fc_input

    def RSA_cs(self,xpool,xrt):
        # xrt (batchssize, rep_dim)
        # xpool (batchssize, slice, img_dim)
        xrmm = self.Proj_REP_cs(xrt) # (batchsize, mmdim)
        xpmm = self.Proj_SLICE_cs(xpool)# (batchsize, slice, mmdim)

        # cosine
        xrmm_norm = F.normalize(xrmm, dim=-1).unsqueeze(dim=1)
        if xpmm.ndim != xrmm_norm.ndim:
             xpmm = xpmm.unsqueeze(dim=0)
        xpmm_norm = F.normalize(xpmm, dim=-1)
        cos_sim = torch.matmul(xrmm_norm, xpmm_norm.transpose(1, 2))
        cos_sim_norm = F.softmax(cos_sim, dim=-1)
        v_weighted = torch.matmul(cos_sim_norm, xpool)

        v_weighted = v_weighted.squeeze(dim=1)
        slice_atts = cos_sim_norm.squeeze(dim=1)

        return v_weighted,slice_atts

    def forward(self, xis, xr_slice = None, xr_region=None):
        # Encoding
        if self.aggregation == '3D':
            h = self.img_encoder(xis)
            hi = nn.AdaptiveAvgPool3d((1, 1, 1))(h)
            hi = nn.Flatten()(hi)
            return hi, None, None

        v, slice_scores = None, None
        ## 2.5D
        ## first squeeze before encoding （batch, channel, slice ,256, 256）
        xis = xis.transpose(1, 2)
        batchsize = xis.shape[0]
        xis = xis.reshape(batchsize * self.slice_num, self.channels, self.H, self.W)  #(batch*slice, c, h, w)
        hi = self.img_encoder(xis)
        fm_dim = hi.shape[1]
        hi = hi.reshape(batchsize, self.slice_num, fm_dim, hi.shape[-2], hi.shape[-1])  # (batch,slice, 512, 8,8)
        h_ = nn.AdaptiveAvgPool2d((1, 1))(hi)  # hi (batch,slice,512, 1, 1)
        h_squeeze = h_.reshape(batchsize, self.slice_num, fm_dim)

        if self.aggregation == 'AVG' or self.aggregation =='AVG_dem':
            v = torch.mean(h_squeeze,dim=1) # avg on slice
        elif self.aggregation == 'DIAMIT':
            v, slice_scores = self.RSA_cs(h_squeeze, xr_slice)


        return v, slice_scores

# Report Encoder
class RepEncoder(nn.Module):
    def __init__(self, rep_model, freeze_layers, device):
        super(RepEncoder, self).__init__()

        # init roberta
        self.roberta_model = self._get_rep_basemodel(rep_model, freeze_layers)
        self.device_rep = device

    def _get_rep_basemodel(self, rep_model_name, freeze_layers):
        try:
            print("report feature extractor:", rep_model_name)
            if rep_model_name == 'bert-base-chinese':
                model = AutoModel.from_pretrained(BERT_PATH)
            elif rep_model_name == 'chinese-roberta-wwm-ext':
                model = BertModel.from_pretrained(ROBERTA_PATH)
            print("--------Report pre-train model load successfully --------")
        except:
            raise ("Invalid model name. Check the config file and pass a BERT model from transformers lybrary")

        if freeze_layers is not None:
            for layer_idx in freeze_layers:
                for param in list(model.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = False

        return model

    def mean_pooling(self, model_output, attention_mask):
        """
        Mean Pooling - Take attention mask into account for correct averaging
        """
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self,encoded_inputs):
        encoded_inputs = encoded_inputs.to(self.device_rep)
        outputs = self.roberta_model(**encoded_inputs)
        mp_embeddings = self.mean_pooling(outputs, encoded_inputs['attention_mask'])
        cls_embeddings = outputs[1]
        token_embeddings = outputs[0]

        return cls_embeddings, mp_embeddings, token_embeddings

# Fusion and Prediction
class Classifier(nn.Module):
    def __init__(self, img_outputdim, rep_output_dim, multimodal_dim, bias, num_class, device, w_agesex = False, predict_type = 'img+rep', fusion_method = 'concate'):
        super(Classifier, self).__init__()

        self.img_dim = img_outputdim
        self.rep_dim = rep_output_dim
        self.mm_dim = multimodal_dim
        self.bias = bias
        self.num_class = num_class
        self.predict_type = predict_type
        self.fusion_method = fusion_method
        self.device_cls = device

        # PROJECTION MATRICES
        self.proj_img = nn.Linear(self.img_dim,self.mm_dim,bias = self.bias)
        self.proj_rep = nn.Linear(self.rep_dim, self.mm_dim,bias = self.bias)

        # FCs
        ## FC for img_only baselines
        self.FC_img_binary = nn.Sequential(
            nn.Linear(self.img_dim, self.num_class),
            nn.Softmax(dim=-1)
        )
        self.FC_img_mc= nn.Sequential(
            nn.Linear(self.img_dim, self.num_class)
        )
        ## FC for rep_only baselines
        self.FC_rep_bianry = nn.Sequential(
            nn.Linear(self.rep_dim, self.num_class),
            nn.Softmax(dim=-1)
        )
        self.FC_rep_mc = nn.Sequential(
            nn.Linear(self.rep_dim, self.num_class)
        )
        ## FC for multi-modal baselines
        self.FC_mm_proj_binary = nn.Sequential(
            nn.Linear(self.mm_dim + self.mm_dim, self.num_class),
            nn.Softmax(dim=-1)
        )
        #
        self.FC_mm_proj_mc = nn.Sequential(
            nn.Linear(self.mm_dim + self.mm_dim, self.num_class),
        )

        self.sa_embeddings = nn.Linear(2, self.mm_dim, bias = True)

    def XAI_contribution(self, z, cls_layer, mmdim):
        zi, zr, zd = z[:,:mmdim], z[:,mmdim:mmdim*2], z[:,mmdim*2:]
        Wi, Wr, Wd= cls_layer.weight.data[:,:mmdim], cls_layer.weight.data[:,mmdim:mmdim*2], cls_layer.weight.data[:,mmdim*2:] # (2,128)
        b = cls_layer.bias.data.unsqueeze(dim=0).repeat(zi.shape[0],1) # (2,)
        logit_i_b3, logit_r_b3, logit_d_b3 = torch.matmul(zi, Wi.transpose(0,1)) + b/3, torch.matmul(zr, Wr.transpose(0,1)) + b/3, torch.matmul(zd, Wd.transpose(0,1)) + b/3
        contris = torch.cat([logit_i_b3.unsqueeze(dim=-1), logit_r_b3.unsqueeze(dim=-1), logit_d_b3.unsqueeze(dim=-1)],dim=-1)  # (32,5,3)
        contris_ = nn.Softmax(dim=-1).to(self.device_cls)(contris)
        return contris_

    def forward(self, zie, zre, zage=None, zsex=None):
        # if zie!= None:
        #     if zie.ndim == 1:
        #         zie = zie.unsqueeze(dim=0)
        # if zre!= None:
        #     if zre.ndim == 1:
        #         zre = zre.unsqueeze(dim=0)
        z, contris_ = None, None

        # binary cls
        if self.num_class == 2:
            if self.predict_type == 'img_only':
                z = self.FC_img_binary(zie)
            elif self.predict_type == 'rep_only':
                z = self.FC_rep_bianry(zre)
            elif self.predict_type == 'img+rep':
                if self.fusion_method == 'concate':
                    z_ = torch.cat([zie,zre],dim=-1)
                    z = self.FC_mm_binary(z_)
                elif self.fusion_method == 'proj':
                    zim = self.proj_img(zie)
                    zrm = self.proj_rep(zre)
                    # sexage
                    zsex = zsex.unsqueeze(dim=-1)
                    zage = zage.unsqueeze(dim=-1)
                    zsa = torch.cat([zsex, zage], dim=-1)
                    zsa_ = self.sa_embeddings(zsa)
                    z_ = torch.cat([zim, zrm, zsa_], dim=-1).squeeze(dim=1)
                    contris_ = self.XAI_contribution(z_, self.FC_mm_proj_binary[0], mmdim=self.mm_dim)
                    z = self.FC_mm_proj_binary(z_)
                else:
                    print('wrong value of concat_type')
            else:
                print('wrong value of fc_type')
        else:
            if self.predict_type == 'img_only':
                z = self.FC_img_mc(zie)
            elif self.predict_type == 'rep_only':
                z = self.FC_rep_mc(zre)
            elif self.predict_type == 'img+rep':
                if self.fusion_method == 'concate':
                    z_ = torch.cat([zie,zre],dim=-1)
                    z = self.FC_mm_mc(z_)
                elif self.fusion_method == 'proj':
                    zim = self.proj_img(zie) # (32, 512) - > (32, mmdim)
                    zrm = self.proj_rep(zre) # (32, 768) - > (32, mmdim)
                    # sexage
                    zsex = zsex.unsqueeze(dim=-1)
                    zage = zage.unsqueeze(dim=-1)
                    zsa = torch.cat([zsex, zage], dim=-1)
                    zsa_ = self.sa_embeddings(zsa)
                    z_ = torch.cat([zim, zrm, zsa_], dim=-1).squeeze(dim=1)
                    contris_ = self.XAI_contribution(z_, self.FC_mm_proj_mc[0], mmdim=self.mm_dim)
                    z = self.FC_mm_proj_mc(z_)
                else:
                    print('wrong value of concat_type')
            else:
                print('wrong value of fc_type')
        return z, contris_




