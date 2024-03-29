import numpy as np
import torch
from torch import nn
import math
import torch.nn.functional as F

class DotProductAttention(nn.Module):
    def __init__(self, dropout=None, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = None
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
    def forward(self, queries, keys, values, expand = False):
        d = queries.shape[-1]
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(d)
        self.attention_weights = nn.functional.softmax(scores, dim=-1)
        if self.dropout is not None:
            att_weights = self.dropout(self.attention_weights)
            return torch.matmul(att_weights, values),att_weights
        else:
            return torch.matmul(self.attention_weights, values), self.attention_weights

# version 2
class Attention(nn.Module):
    def __init__(self, temperature=None, attn_dropout=0.1, cuda_device = "cuda:0"):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.device = cuda_device

    def forward(self, q, k, v, mask=None):
        if self.temperature == None:
            self.temperature = math.sqrt(q.shape[-1])
        attn = torch.matmul(q / self.temperature, k.transpose(-2, -1))
        if mask is not None:
            mask = torch.Tensor(mask).reshape(1, 1, 1, -1).to(self.device)
            mask_ = mask.repeat(attn.shape[0],attn.shape[1],attn.shape[2],1)
            attn  = attn.masked_fill(mask_ == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class CrossSliceAttention(nn.Module):
    def __init__(self,input_channels):
        super(CrossSliceAttention,self).__init__()
        self.linear_q=nn.Conv2d(in_channels=input_channels,out_channels=input_channels,kernel_size=(1,1),bias=False)
        self.linear_k=nn.Conv2d(in_channels=input_channels,out_channels=input_channels,kernel_size=(1,1),bias=False)
        self.linear_v=nn.Conv2d(in_channels=input_channels,out_channels=input_channels,kernel_size=(1,1),bias=False)

    def forward(self,pooled_features,features):
        q=self.linear_q(pooled_features)
        q=q.view(q.size(0),-1)
        k=self.linear_k(pooled_features)
        k=k.view(k.size(0),-1)
        v=self.linear_v(features)
        x=torch.matmul(q,k.permute(1,0))/np.sqrt(q.size(1))
        x=torch.softmax(x,dim=1) # (l,l)
        out=torch.zeros_like(v) #(l,c,w,h)
        for i in range(x.size(0)):
            temp=x[i,:].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            out[i,:,:,:]=torch.sum(temp*v,dim=0).clone()
        return out,x


class MultiHeadedCrossSliceAttentionModule(nn.Module):
    def __init__(self,input_channels,heads=3,pool_kernel_size=(4,4),input_size=(128,128),batch_size=20,pool_method='avgpool'):
        super(MultiHeadedCrossSliceAttentionModule,self).__init__()
        self.attentions=[]
        self.linear1=nn.Conv2d(in_channels=heads*input_channels,out_channels=input_channels,kernel_size=(1,1))
        self.norm1=nn.LayerNorm([batch_size,input_channels,input_size[0],input_size[1]])
        self.linear2=nn.Conv2d(in_channels=input_channels,out_channels=input_channels,kernel_size=(1,1))
        self.norm2=nn.LayerNorm([batch_size,input_channels,input_size[0],input_size[1]])

        if pool_method=="maxpool":
            self.pool=nn.MaxPool2d(kernel_size=pool_kernel_size)
        elif pool_method=="avgpool":
            self.pool=nn.AvgPool2d(kernel_size=pool_kernel_size)
        else:
            assert (False)  # not implemented yet

        for i in range(heads):
            self.attentions.append(CrossSliceAttention(input_channels))
        self.attentions=nn.Sequential(*self.attentions)

    def forward(self,pooled_features,features):

        for i in range(len(self.attentions)):
            x_=self.attentions[i](pooled_features,features)
            if i==0:
                x=x_
            else:
                x=torch.cat((x,x_),dim=1)
        out=self.linear1(x)
        x=F.gelu(out)+features
        out_=self.norm1(x)
        out=self.linear2(out_)
        x=F.gelu(out)+out_
        out=self.norm2(x)
        pooled_out=self.pool(out)
        return pooled_out,out