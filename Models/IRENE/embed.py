# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import BCEWithLogitsLoss,CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from transformers import BertModel, AutoModel

import Models.IRENE.configs as configs
from Models.IRENE.attention import Attention
import pdb

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, device, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)
        tk_lim = config.cc_len
        num_lab = config.lab_len
        self.device = device

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        # extra
        self.bertmodel = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
        # self.bertmodel = AutoModel.from_pretrained("bert-base-uncased")

        self.cc_embeddings = Linear(768, config.hidden_size)  
        self.lab_embeddings = Linear(1, config.hidden_size)  
        self.sex_embeddings = Linear(1, config.hidden_size)  
        self.age_embeddings = Linear(1, config.hidden_size)  
        
        self.position_embeddings = nn.Parameter(torch.zeros(1, 1+n_patches, config.hidden_size))
        self.pe_cc = nn.Parameter(torch.zeros(1, tk_lim, config.hidden_size))
        self.pe_lab = nn.Parameter(torch.zeros(1, num_lab, config.hidden_size))
        self.pe_sex = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.pe_age = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])
        self.dropout_cc = Dropout(config.transformer["dropout_rate"])
        self.dropout_lab = Dropout(config.transformer["dropout_rate"])
        self.dropout_sex = Dropout(config.transformer["dropout_rate"])
        self.dropout_age = Dropout(config.transformer["dropout_rate"])

    def mean_pooling(self, model_output, attention_mask):
        """
        Mean Pooling - Take attention mask into account for correct averaging
        """
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    # def forward(self, x, cc, lab, sex, age):
    def forward(self, x, cc, age, sex):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.hybrid:
            x = self.hybrid_model(x)
        # for multislice x (batch, channel, slice, H, W)
        if x.ndim == 5:
            C,S,H,W = x.shape[1],x.shape[2],x.shape[3],x.shape[4]
        else:
            C, H, W = x.shape[1], x.shape[2], x.shape[3]
            S = 1
        x = x.transpose(1, 2).reshape(B * S, C, H, W)
        x = self.patch_embeddings(x) # 16*16 --> CNN --> 1*1 (B*S, 768, 16,16)
        # bert encoding
        cc = cc.to(self.device)
        bertout = self.bertmodel(**cc)
        cc_bert = bertout[0]
        # mp_embeddings = self.mean_pooling(bertout, cc['attention_mask'])
        cc = self.cc_embeddings(cc_bert)
        # if not (lab == None) :
        #     lab = self.lab_embeddings(lab)
        # if sex is not None:
        #     sex = self.sex_embeddings(sex)
        # if age is not None:
        #     age = self.age_embeddings(age)

        x = x.flatten(2) # (384,768,16,16) --> (384,768,256)
        x = x.transpose(-1, -2) # （384,256,768）
        x = x.reshape(B ,S, x.shape[-2],x.shape[-1])# （32,23,256,768）
        x = torch.mean(x, dim=1) # （32,256,768）
        x = torch.cat((cls_tokens, x), dim=1) # （32,257,768）

        embeddings = x + self.position_embeddings
        cc_embeddings = cc + self.pe_cc
        # lab_embeddings = lab + self.pe_lab
        if sex is not None:
            sex = sex.unsqueeze(dim=-1).unsqueeze(dim=-1)
            sex = sex.type(torch.FloatTensor)
            sex = sex.to(self.device)
            sex = self.sex_embeddings(sex)
            sex_embeddings = sex + self.pe_sex
            sex_embeddings = self.dropout_sex(sex_embeddings)
        if age is not None:
            age = age.unsqueeze(dim=-1).unsqueeze(dim=-1)
            age = age.type(torch.FloatTensor)
            age = age.to(self.device)
            age = self.age_embeddings(age)
            age_embeddings = age + self.pe_age
            age_embeddings = self.dropout_age(age_embeddings)

        embeddings = self.dropout(embeddings)
        cc_embeddings = self.dropout_cc(cc_embeddings)
        # lab_embeddings = self.dropout_lab(lab_embeddings)
        # sex_embeddings = self.dropout_sex(sex_embeddings)
        # age_embeddings = self.dropout_age(age_embeddings)
        if (sex is not None) and (age is not None):
            # return embeddings, cc_embeddings, lab_embeddings, sex_embeddings, age_embeddings
            return embeddings, cc_embeddings, sex_embeddings, age_embeddings
        else:
            # (B, 256+1, 768); (B, reportmaxlength, 768)
            return embeddings, cc_embeddings


