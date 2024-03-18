# @author       : Bingyu Xin   
# @Institute    : CS@Rutgers
import torch
from torch import nn
from model.BasicModule import conv_block
from model.BasicModule import UNetRes
import torch.nn.functional as F
import numpy as np
import ipdb

def geometric_approximation_1(s):  
    ba, chan = s.shape
    dtype = s.dtype

    I = torch.ones([ba, chan], device=s.device).type(dtype)
    temp = 1e-8 * I
    s = s + temp
    I = torch.diag_embed(I)
    # I = torch.eye(s.shape[0], device=s.device).type(dtype)
    p = s.unsqueeze(-1) / s.unsqueeze(-2) - I
    p = torch.where(p < 1., p, 1. / p) 

    a1 = s.unsqueeze(-1).repeat(1, 1, chan).permute(0, 2,1)
    # a1 = s.repeat(s.shape[0], 1).t()
    a1_t = a1.permute(0, 2,1)
    lamiPluslamj = 1. / ((s.unsqueeze(-1) + s.unsqueeze(-2)))  # do not need to sub I,because have been * a1

    a1 = 1. / torch.where(a1 >= a1_t, a1, - a1_t)
    # a1 *= torch.ones(s.shape[0], s.shape[0], device=s.device).type(dtype) - I
    a1 *= torch.ones_like(I, device=s.device).type(dtype) - I
    p_app = torch.ones_like(p)
    p_hat = torch.ones_like(p)
    for i in range(9):
        p_hat = p_hat * p
        p_app += p_hat
    a1 = lamiPluslamj * a1 * p_app

    return a1
class svdv2_1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M):
        try:
            U, S, V = torch.svd(M,some=True,compute_uv=True)
        except:#avoid cond() too large
            print(M.max())
            print(M.min())
            ipdb.set_trace()
            U, S, V = torch.svd(M+1e-3*M.mean()*torch.rand_like(M),some=True,compute_uv=True)
        dtype = M.dtype
        S[S <= torch.finfo(dtype).eps] = torch.finfo(dtype).eps
        ctx.save_for_backward(M, U, S,V)
        return U,S,V

    @staticmethod
    def backward(ctx, dL_du, dL_ds,dL_dv):
        M, U,S,V = ctx.saved_tensors
        k= geometric_approximation_1(S)
        k[k == float('inf')] = k[k != float('inf')].max()
        k[k == float('-inf')] = k[k != float('-inf')].min()
        k[k != k] = k.max()
        K_t=k.permute(0,2,1)
        diag_s=torch.diag_embed(S).type(torch.complex64)
        VT=torch.permute(V,[0,2,1])
        tt=2*torch.matmul(diag_s,K_t*torch.matmul(VT,dL_dv))
        grad_input=tt+torch.diag_embed(dL_ds)
        US=torch.matmul(U, grad_input)
        grad_input = torch.matmul(US, VT)
        return grad_input

    
class lowranknet(nn.Module):
    def __init__(self, buffer_size=1, n_iter=8, n_convs=6, n_filters=64, block_type='cnn', norm='ortho'):
        '''
        HQS-Net from paper " Learned Half-Quadratic Splitting Network for MR Image Reconstruction "
        ( https://openreview.net/pdf?id=h7rXUbALijU ) ( https://github.com/hellopipu/HQS-Net )
        :param buffer_size:  buffer_size m
        :param n_iter:  iterations n
        :param n_convs: convolutions in each reconstruction block
        :param n_filters: output channel for convolutions
        :param block_type: 'cnn' or 'unet
        :param norm: 'ortho' norm for fft
        '''

        super().__init__()
        self.norm = norm
        self.m = buffer_size
        self.n_iter = n_iter
        ## the initialization of mu may influence the final accuracy
        self.mu = nn.Parameter(1 * torch.ones((1, 1))) # 1
        self.mu1 = nn.Parameter(1 * torch.ones((1, 1))) 
        self.mu2 = nn.Parameter(1 * torch.ones((1, 1))) 
        
        self.thres_coef = nn.Parameter(torch.Tensor([5]))  #5
        self.block_type = block_type
        if self.block_type == 'cnn':
            rec_blocks = []
            for i in range(self.n_iter):
                rec_blocks.append(
                    conv_block('hqs-net', channel_in=  (2*(self.m + 1)-2), n_convs=n_convs,
                               n_filters=n_filters))  # self.m +
            self.rec_blocks = nn.ModuleList(rec_blocks)
        elif self.block_type == 'unet':
            self.rec_blocks = UNetRes(in_nc=2 * (self.m + 1), out_nc=2 * self.m, nc=[64, 128, 256, 512], nb=4,
                                      act_mode='R',
                                      downsample_mode="strideconv", upsample_mode="convtranspose")
    
    ##这个函数将图像从图像空间转换到 k-空间（频率空间）。
    ##使用快速傅立叶变换 (FFT) 将图像从图像域转换为 k-空间，并与给定的掩码相乘。
    ## img = x; mask = M; fft = F, so k = y.
    def _forward_operation(self, img, mask):

        k = torch.fft.fft2(torch.view_as_complex(img.permute(0, 2, 3, 1).contiguous()),
                           norm=self.norm)       ##torch.view_as_complex: 这个函数用于将一个实数张量转换为复数张量;将图像从 [batch_size, channels, height, width] 转换为 [batch_size, height, width, channels];傅里叶变换和逆傅里叶变换应该使用正交化标准化
        k = torch.view_as_real(k).permute(0, 3, 1, 2).contiguous()   #傅里叶变换的结果从复数张量转换为一个包含实部和虚部的实数张量。[batch_size, height, width, channels] 改变回 [batch_size, channels, height, width]。
        k = mask * k
        return k    

    ##这个函数执行的操作与前向操作相反：它将 k-空间数据转换回图像空间。k=y
    ##使用逆快速傅立叶变换 (IFFT)。F_H
    def _backward_operation(self, k, mask):
        k = mask * k
        #print('k_shape', k.shape) (16,2,192,160)
        img = torch.fft.ifft2(torch.view_as_complex(k.permute(0, 2, 3, 1).contiguous()), norm=self.norm)
        #print('img', img.shape)  (16,192,160)
        img = torch.view_as_real(img).permute(0, 3, 1, 2).contiguous()
        #print('img1', img.shape)  (16,2,192,160)
        return img
    
    #计算 k-空间中的残差，然后将其逆变换回图像空间
    def update_opration(self, f_1, f_2, k, mask):
        h_1 = k - self.mu1*self._forward_operation(f_1, mask) - self.mu2*self._forward_operation(f_2, mask)
        update = f_1 + f_2 + self.mu * self._backward_operation(h_1, mask)
        return update
    


    def lowrank(self,x):
        # 使用 view_as_complex 合成复数矩阵
        complex_matrix = torch.view_as_complex(x.permute(0, 2, 3, 1).contiguous())
        # 对复数矩阵进行 SVD
        U, S, Vh = svdv2_1.apply(complex_matrix)

        # 处理 S
        #thres = torch.sigmoid(self.thres_coef) * S[:, 0]
        #thres = 5 * S[:, 0]
        #thres = torch.unsqueeze(thres,-1)
       # S = torch.relu(S -  thres)
        S = torch.sign(S)*torch.relu(torch.abs(S)-0.01)
        # 将 S 扩展为对角矩阵并转换为复数形式
        S_expanded = torch.diag_embed(S).type(torch.complex64)

        # 计算 US
        US = torch.matmul(U, S_expanded)

        # 重构复数矩阵
        M = torch.matmul(US, Vh.transpose(-2, -1))

        # 分离实部和虚部，以适配原始 x_rec 的形状
        x_rec = torch.view_as_real(M).permute(0, 3, 1, 2).contiguous()

        return x_rec

    
    def forward(self, img, k, mask): 
        '''
        :param img: zero-filled images, (batch,2,h,w)
        :param k:   corresponding undersampled k-space data , (batch,2,h,w)
        :param mask: uncentered sampling mask , (batch,2,h,w)
        :return: reconstructed img
        '''

        ## initialize buffer f : the concatenation of m copies of the complex-valued zero-filled images
        f = torch.cat([img] * self.m, 1).to(img.device)##(batch,10,h,w)
        #f1 = torch.cat([img] * self.m, 1).to(img.device)
        #f1 = torch.randn_like(f)*0.01
        f1 = torch.zeros_like(f)
        #print(f.shape)  (16,2,192,160)
        ## n reconstruction blocks
        for i in range(self.n_iter):
            f_1 = f[:, 0:2].clone() ##创建了一个切片张量的副本；(batch, 2, h, w)
            #f_2 = torch.rand_like(f1[:, 0:2].clone())*0.01
            f_2 = f1.clone()
            updated_f_1 = self.update_opration(f_1, f_2, k, mask)#(16,2,196,160)
            if self.block_type == 'cnn':
                f =  f_1+self.rec_blocks[i](updated_f_1)  #(16,2,196,160) f + self.rec_blocks[i](torch.cat([f, updated_f_1], 1)) 
                #f = torch.mul(torch.sign(updated_f_1), F.relu(torch.abs(updated_f_1) - self.soft_thr))
                f1 = self.lowrank(updated_f_1)
            elif self.block_type == 'unet':
                f = f + self.rec_blocks(torch.cat([f, updated_f_1], 1))
        return  updated_f_1[:, 0:2]
