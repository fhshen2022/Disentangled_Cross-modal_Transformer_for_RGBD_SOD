# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 15:51:22 2023

@author: Clover
"""

import torch.nn as nn
import torch
from .token_performer import Token_performer
from .Transformer import saliency_token_inference, token_TransformerEncoder
import math



class Decoder_DFP(nn.Module):
    '''
    Decoder with Disentangled Feature Pyramid 
    '''
    def __init__(self, embed_dim=384, token_dim=64, depth=2, img_size=224):

        super(Decoder_DFP, self).__init__()
        self.consistent_token = nn.Parameter(torch.zeros(1, 1, embed_dim//2))
        self.consistent_infer = token_inference(dim = embed_dim//2,num_heads=1)
        self.complementary_token = nn.Parameter(torch.zeros(1, 1, embed_dim//2))
        self.complementary_infer = token_inference(dim = embed_dim//2,num_heads=1)
        self.depth_consistent_token = nn.Parameter(torch.zeros(1, 1, embed_dim//2))
        self.depth_consistent_infer = token_inference(dim = embed_dim//2,num_heads=1)
        self.depth_complementary_token = nn.Parameter(torch.zeros(1, 1, embed_dim//2))
        self.depth_complementary_infer = token_inference(dim = embed_dim//2,num_heads=1)
        self.mutualinfo = Mutual_info_reg(input_channels=96, channels=token_dim, latent_size=6, hw=14*14)
        self.mutualinfo2 = Mutual_info_reg(input_channels=32, channels=token_dim, latent_size=6, hw=28*28)
        self.mutualinfo3 = Mutual_info_reg(input_channels=32, channels=token_dim, latent_size=6, hw=56*56)
        self.saliency_token = nn.Parameter(torch.zeros(1, 1, embed_dim//2))
        
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, token_dim),
        )

        
        self.img_size = img_size
        
        # token upsampling and multi-level token fusion
        self.decoder1 = decoder_module(dim=384//2, token_dim=token_dim, img_size=img_size, ratio=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True)
        self.decoder2 = decoder_module(dim=384//2, token_dim=token_dim, img_size=img_size, ratio=4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True)
        self.decoder3 = decoder_module(dim=384//2, token_dim=token_dim,  img_size=img_size, ratio=1, kernel_size=(7, 7), stride=(4, 4), padding=(2, 2), fuse=False)
        
        # token based multi-task predictions
        self.token_pre_1_8 = token_trans(in_dim=token_dim, embed_dim=embed_dim//2, depth=depth//2, num_heads=1)
        self.token_pre_1_4 = token_trans(in_dim=token_dim, embed_dim=embed_dim//2, depth=depth//2, num_heads=1)
        self.token_pre_1_16 = token_trans(in_dim=embed_dim, embed_dim=embed_dim//2, depth=depth//2, num_heads=1)
        
        self.rgb_comp_pre_1_8 = token_trans(in_dim=token_dim//2, embed_dim=embed_dim//2, depth=depth//2, num_heads=1)
        self.depth_comp_pre_1_8 = token_trans(in_dim=token_dim//2, embed_dim=embed_dim//2, depth=depth//2, num_heads=1)
        self.rgb_cons_pre_1_8 = token_trans(in_dim=token_dim//2, embed_dim=embed_dim//2, depth=depth//2, num_heads=1)
        self.depth_cons_pre_1_8 = token_trans(in_dim=token_dim//2, embed_dim=embed_dim//2, depth=depth//2, num_heads=1)
        #self.rgb_cat_1_8 = nn.Linear(64,32)
        #self.depth_cat_1_8 = nn.Linear(64,32)
        
        
        self.rgb_comp_pre_1_4 = token_trans(in_dim=token_dim//2, embed_dim=embed_dim//2, depth=depth//2, num_heads=1)
        self.depth_comp_pre_1_4 = token_trans(in_dim=token_dim//2, embed_dim=embed_dim//2, depth=depth//2, num_heads=1)
        self.rgb_cons_pre_1_4 = token_trans(in_dim=token_dim//2, embed_dim=embed_dim//2, depth=depth//2, num_heads=1)
        self.depth_cons_pre_1_4 = token_trans(in_dim=token_dim//2, embed_dim=embed_dim//2, depth=depth//2, num_heads=1)
        #self.rgb_cat_1_4 = nn.Linear(64,32)
        #self.depth_cat_1_4 = nn.Linear(64,32)
        

        # predict saliency maps
        self.pre_1_16 = nn.Linear(token_dim, 1)
        self.pre_1_8 = nn.Linear(token_dim, 1)
        self.pre_1_4 = nn.Linear(token_dim, 1)
        self.pre_1_1 = nn.Linear(token_dim, 1)
        
        self.rgb_pre_1_16 = nn.Linear(96,1)
        self.depth_pre_1_16 = nn.Linear(96,1)
        self.rgb_pre_1_8 = nn.Linear(32,1)
        self.depth_pre_1_8 = nn.Linear(32,1)
        self.rgb_pre_1_4 = nn.Linear(32,1)
        self.depth_pre_1_4 = nn.Linear(32,1)
        
        
        self.norm_c = nn.LayerNorm(embed_dim)
        self.mlp_c = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, token_dim),
        )
        
        self.proj_1_16 = nn.Linear(384,192)
        #self.proj_1_16 = nn.Linear(384*2,192*2)
        #self.depth_proj_1_16 = nn.Linear(384,192)


        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.xavier_uniform_(m.weight),
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif classname.find('Linear') != -1:
                nn.init.xavier_uniform_(m.weight),
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif classname.find('BatchNorm') != -1:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        
    def forward(self, rgb_fea_1_16, rgb_fea_1_8, rgb_fea_1_4, depth_fea_1_16, depth_fea_1_8, depth_fea_1_4):
        b,hw,c = rgb_fea_1_16.shape
        
        
        consistent_tokens = self.consistent_token.expand(b,-1,-1) #[b,1,192]
        complementary_tokens =self.complementary_token.expand(b,-1,-1)
        rgb_fea_1_16_cons = torch.cat((consistent_tokens,rgb_fea_1_16[:,:,:c//2]), dim=1)
        rgb_fea_1_16_cons = self.consistent_infer(rgb_fea_1_16_cons)#[b,14*14,384//4]
        rgb_fea_1_16_comp =  torch.cat((complementary_tokens,rgb_fea_1_16[:,:,c//2:]), dim=1)
        rgb_fea_1_16_comp = self.complementary_infer(rgb_fea_1_16_comp)
        
        depth_consistent_tokens = self.depth_consistent_token.expand(b,-1,-1)
        depth_complementary_tokens =self.depth_complementary_token.expand(b,-1,-1)
        depth_fea_1_16_cons = torch.cat((depth_consistent_tokens,depth_fea_1_16[:,:,:c//2]), dim=1)
        depth_fea_1_16_cons = self.depth_consistent_infer(depth_fea_1_16_cons)
        depth_fea_1_16_comp =  torch.cat((depth_complementary_tokens,depth_fea_1_16[:,:,c//2:]), dim=1)
        depth_fea_1_16_comp = self.depth_complementary_infer(depth_fea_1_16_comp)
        rgb_pre_1_16 = self.rgb_pre_1_16(rgb_fea_1_16_cons).transpose(1,2).reshape(b,1,14,14)
        depth_pre_1_16 = self.depth_pre_1_16(depth_fea_1_16_cons).transpose(1,2).reshape(b,1,14,14)
        loss1 = self.mutualinfo(rgb_fea_1_16_comp,depth_fea_1_16_comp)#-self.consis_mutualinfo(rgb_fea_1_16_cons,depth_fea_1_16_comp)
        
        rgb_fea_1_16 = torch.cat((rgb_fea_1_16_cons,rgb_fea_1_16_comp), dim=-1)
        depth_fea_1_16 = torch.cat((depth_fea_1_16_cons,depth_fea_1_16_comp),dim=-1)
        
        fea_1_16 = torch.cat((rgb_fea_1_16,depth_fea_1_16),dim=-1) #[b,14*14,384]
        #fea_1_16 = self.proj_1_16(torch.cat((rgb_fea_1_16,depth_fea_1_16),dim=-1))
        
        
        saliency_tokens = self.saliency_token.expand(b,-1,-1)
        saliency_fea_1_16, token_fea_1_16, saliency_tokens = self.token_pre_1_16(fea_1_16,saliency_tokens)
        saliency_fea_1_16 = self.mlp(self.norm(saliency_fea_1_16)) #[b,14*14,64]
        
        # saliency_fea_1_16 [B, 14*14, 64]
        
        mask_1_16 = self.pre_1_16(saliency_fea_1_16)
        mask_1_16 = mask_1_16.transpose(1, 2).reshape(b, 1, self.img_size // 16, self.img_size // 16) 
        
        #fea_1_8 = self.decoder1(saliency_fea_1_16,rgb_fea_1_8,depth_fea_1_8)
        # 1/16 -> 1/8
        # reverse T2T and fuse low-level feature
        
        rgb_fea_cons_1_8,  _, consistent_tokens = self.rgb_cons_pre_1_8(rgb_fea_1_8[:,:,:32], consistent_tokens)
        depth_fea_cons_1_8,  _, depth_consistent_tokens = self.depth_cons_pre_1_8(depth_fea_1_8[:,:,:32], depth_consistent_tokens)
        rgb_fea_comp_1_8,  _, complementary_tokens = self.rgb_comp_pre_1_8(rgb_fea_1_8[:,:,32:], complementary_tokens)
        depth_fea_comp_1_8,  _, depth_complementary_tokens = self.depth_comp_pre_1_8(depth_fea_1_8[:,:,32:], depth_complementary_tokens)
        rgb_fea_1_8 = torch.cat((rgb_fea_cons_1_8,rgb_fea_comp_1_8),dim=-1)#[b,28*28,32]
        depth_fea_1_8 = torch.cat((depth_fea_cons_1_8,depth_fea_comp_1_8),dim=-1)
        
        rgb_pre_1_8 = self.rgb_pre_1_8(rgb_fea_cons_1_8).transpose(1,2).reshape(b,1,28,28)
        depth_pre_1_8 = self.depth_pre_1_8(depth_fea_cons_1_8).transpose(1,2).reshape(b,1,28,28)
        loss2 = self.mutualinfo2(rgb_fea_comp_1_8,depth_fea_comp_1_8)#-self.consis_mutualinfo2(rgb_fea_cons_1_8,depth_fea_cons_1_8)
        
        fea_1_8 = self.decoder1(token_fea_1_16[:, 1:, :], rgb_fea_1_8,depth_fea_1_8)
        #fea_1_8 = self.decoder1(saliency_fea_1_16, rgb_fea_1_8,depth_fea_1_8)

        # token prediction
        saliency_fea_1_8,  token_fea_1_8, saliency_tokens = self.token_pre_1_8(fea_1_8, saliency_tokens)
       
        # predict saliency and contour maps
        mask_1_8 = self.pre_1_8(saliency_fea_1_8)
        mask_1_8 = mask_1_8.transpose(1, 2).reshape(b, 1, self.img_size // 8, self.img_size // 8)

        # 1/8 -> 1/4
        
        rgb_fea_cons_1_4,  _, consistent_tokens = self.rgb_cons_pre_1_4(rgb_fea_1_4[:,:,:32], consistent_tokens)
        depth_fea_cons_1_4,  _, depth_consistent_tokens = self.depth_cons_pre_1_4(depth_fea_1_4[:,:,:32], depth_consistent_tokens)
        rgb_fea_comp_1_4,  _, complementary_tokens = self.rgb_comp_pre_1_4(rgb_fea_1_4[:,:,32:], complementary_tokens)
        depth_fea_comp_1_4,  _, depth_complementary_tokens = self.depth_comp_pre_1_4(depth_fea_1_4[:,:,32:], depth_complementary_tokens)
        rgb_fea_1_4 = torch.cat((rgb_fea_cons_1_4,rgb_fea_comp_1_4),dim=-1)#[b,28*28,32]
        depth_fea_1_4 = torch.cat((depth_fea_cons_1_4,depth_fea_comp_1_4),dim=-1)
        
        rgb_pre_1_4 = self.rgb_pre_1_4(rgb_fea_cons_1_4).transpose(1,2).reshape(b,1,56,56)
        depth_pre_1_4 = self.depth_pre_1_4(depth_fea_cons_1_4).transpose(1,2).reshape(b,1,56,56)
        loss3 = self.mutualinfo3(rgb_fea_comp_1_4,depth_fea_comp_1_4)#-self.consis_mutualinfo3(rgb_fea_cons_1_4,depth_fea_cons_1_4)
        
        fea_1_4 = self.decoder2(token_fea_1_8[:, 1:, :], rgb_fea_1_4,depth_fea_1_4)

        # token prediction
        saliency_fea_1_4, token_fea_1_4, saliency_tokens = self.token_pre_1_4(fea_1_4, saliency_tokens)
       
        
        # predict saliency maps and contour maps
        mask_1_4 = self.pre_1_4(saliency_fea_1_4)
        mask_1_4 = mask_1_4.transpose(1, 2).reshape(b, 1, self.img_size // 4, self.img_size // 4)

        # 1/4 -> 1
        saliency_fea_1_1 = self.decoder3(saliency_fea_1_4)

        mask_1_1 = self.pre_1_1(saliency_fea_1_1)
        mask_1_1 = mask_1_1.transpose(1, 2).reshape(b, 1, self.img_size // 1, self.img_size // 1)
        
        
        
        return [mask_1_16, mask_1_8, mask_1_4, mask_1_1], [loss1,loss2,loss3], [rgb_pre_1_16,depth_pre_1_16,rgb_pre_1_8,depth_pre_1_8,rgb_pre_1_4,depth_pre_1_4]






class token_trans(nn.Module):
    def __init__(self, in_dim=64, embed_dim=384, depth=14, num_heads=6, mlp_ratio=3.):
        super(token_trans, self).__init__()

        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.encoderlayer = token_TransformerEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.saliency_token_pre = saliency_token_inference(dim=embed_dim, num_heads=1)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp2 = nn.Sequential(
            nn.Linear(embed_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
        )

    def forward(self, fea, saliency_tokens):
        B, _, _ = fea.shape

        # project to 384 dim
        fea = self.mlp(self.norm(fea))
        fea = torch.cat((saliency_tokens, fea), dim=1)
        fea = self.encoderlayer(fea)
        saliency_tokens = fea[:, 0, :].unsqueeze(1)
        saliency_fea = self.saliency_token_pre(fea)

        # reproject back to 64 dim
        saliency_fea = self.mlp2(self.norm2(saliency_fea))

        return saliency_fea,  fea, saliency_tokens


class token_inference(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, 384//4, bias=qkv_bias)
        self.k = nn.Linear(dim, 384//4, bias=qkv_bias)
        self.v = nn.Linear(dim, 384//4, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim//2, 384//4)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sigmoid = nn.Sigmoid()
        
        self.proj_fea = nn.Linear(dim, 384//4, bias=qkv_bias)

    def forward(self, fea):
        B, N, C = fea.shape
        x = self.norm(fea)
        T_s, F_s = x[:, 0, :].unsqueeze(1), x[:, 1:, :]
        # T_s [B, 1, 384]  F_s [B, 14*14, 384]
        q = self.q(F_s).reshape(B, (N-1), self.num_heads, C // self.num_heads//2).permute(0, 2, 1, 3)
        k = self.k(T_s).reshape(B, 1, self.num_heads, C // self.num_heads//2).permute(0, 2, 1, 3)
        v = self.v(T_s).reshape(B, 1, self.num_heads, C // self.num_heads//2).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = self.sigmoid(attn)
        attn = self.attn_drop(attn)

        infer_fea = (attn @ v).transpose(1, 2).reshape(B, N-1, C//2)
        infer_fea = self.proj(infer_fea)
        infer_fea = self.proj_drop(infer_fea)

        infer_fea = infer_fea + self.proj_fea(fea[:, 1:, :])
        return infer_fea

from torch.distributions import Normal, Independent, kl
from torch.autograd import Variable
cos_sim = torch.nn.CosineSimilarity(dim=1,eps=1e-8)
CE = torch.nn.BCELoss(reduction='sum')

class Mutual_info_reg(nn.Module):
    '''
    Calculate the mutual information or cosine similarity
    '''
    def __init__(self, input_channels=192, channels=192, latent_size=6, hw=14*14):
        super(Mutual_info_reg, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.layer2 = nn.Conv2d(input_channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.layer3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.layer4 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

        self.channel = channels
        self.hw = hw

        self.fc1_rgb1 = nn.Linear(channels * 1 * hw, latent_size)
        self.fc2_rgb1 = nn.Linear(channels * 1 * hw, latent_size)
        self.fc1_depth1 = nn.Linear(channels * 1 * hw, latent_size)
        self.fc2_depth1 = nn.Linear(channels * 1 * hw, latent_size)

        self.leakyrelu = nn.LeakyReLU()
        self.tanh = torch.nn.Tanh()

    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        return kl_div

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        #eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, rgb_feat, depth_feat):
        #print(rgb_feat.shape)
        b,HW,c = rgb_feat.shape
        h = int(math.sqrt(HW))
        rgb_feat = rgb_feat.transpose(1,2).reshape(b,c,h,h)
        depth_feat = depth_feat.transpose(1,2).reshape(b,c,h,h)
        rgb_feat = self.layer3(self.leakyrelu(self.bn1(self.layer1(rgb_feat))))
        depth_feat = self.layer4(self.leakyrelu(self.bn2(self.layer2(depth_feat))))
        # print(rgb_feat.size())
        # print(depth_feat.size())
        
        rgb_feat = rgb_feat.contiguous().view(-1, self.channel * 1 * HW)
        depth_feat = depth_feat.contiguous().view(-1, self.channel * 1 * HW)

        mu_rgb = self.fc1_rgb1(rgb_feat)
        logvar_rgb = self.fc2_rgb1(rgb_feat)
        mu_depth = self.fc1_depth1(depth_feat)
        logvar_depth = self.fc2_depth1(depth_feat)
        

        mu_depth = self.tanh(mu_depth)
        mu_rgb = self.tanh(mu_rgb)
        logvar_depth = self.tanh(logvar_depth)
        logvar_rgb = self.tanh(logvar_rgb)
        z_rgb = self.reparametrize(mu_rgb, logvar_rgb)
        z_depth = self.reparametrize(mu_depth, logvar_depth)
        '''
        #mutual information
        dist_rgb = Independent(Normal(loc=mu_rgb, scale=torch.exp(logvar_rgb)), 1)
        dist_depth = Independent(Normal(loc=mu_depth, scale=torch.exp(logvar_depth)), 1)
        bi_di_kld = torch.mean(self.kl_divergence(dist_rgb, dist_depth)) + torch.mean(
            self.kl_divergence(dist_depth, dist_rgb))
        z_rgb_norm = torch.sigmoid(z_rgb)
        z_depth_norm = torch.sigmoid(z_depth)
        ce_rgb_depth = CE(z_rgb_norm,z_depth_norm.detach())
        ce_depth_rgb = CE(z_depth_norm, z_rgb_norm.detach())
        #latent_loss = ce_rgb_depth+ce_depth_rgb-bi_di_kld
        '''
        
        #cosine similarity
        latent_loss = torch.abs(cos_sim(z_rgb,z_depth)).sum()

        return latent_loss#, z_rgb, z_depth


class decoder_module(nn.Module):
    def __init__(self, dim=384, token_dim=64, img_size=224, ratio=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True):
        super(decoder_module, self).__init__()

        self.project = nn.Linear(token_dim, token_dim * kernel_size[0] * kernel_size[1])
        self.upsample = nn.Fold(output_size=(img_size // ratio,  img_size // ratio), kernel_size=kernel_size, stride=stride, padding=padding)
        self.fuse = fuse
        self.proj2 =  nn.Sequential(
                                nn.Linear(token_dim , token_dim ),
                                nn.ReLU(),
                                nn.Linear(token_dim,64)
                                 )
        if self.fuse:
            self.concatFuse = nn.Sequential(
                nn.Linear(token_dim+32, token_dim),
                nn.GELU(),
                nn.Linear(token_dim, 64),
            )
            self.concatRgbDepth = nn.Sequential(
                nn.Linear(token_dim*2, token_dim),
                nn.GELU(),
                nn.Linear(token_dim, 32),
            )
            self.att = Token_performer(dim=64, in_dim=64, kernel_ratio=0.5)

            # project input feature to 64 dim
            self.norm = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, token_dim),
                nn.GELU(),
                nn.Linear(token_dim, token_dim),
            )
            

    def forward(self, dec_fea, enc_fea=None, enc_depth=None):

        if self.fuse:
            # from 384 to 64
            dec_fea = self.mlp(self.norm(dec_fea))

        # [1] token upsampling by the proposed reverse T2T module
        dec_fea = self.project(dec_fea)
        dec_fea = self.upsample(dec_fea.transpose(1, 2))
        B, C, H, W = dec_fea.shape
        dec_fea = dec_fea.view(B, C, -1).transpose(1, 2)
        if self.fuse:
            # [2] fuse encoder fea and decoder fea
            enc_fea = self.concatRgbDepth(torch.cat([enc_fea,enc_depth],dim=2))
            dec_fea = self.concatFuse(torch.cat([dec_fea, enc_fea], dim=2))
            dec_fea = self.att(dec_fea)
            return dec_fea
        else:
            dec_fea = self.proj2(dec_fea)
        return dec_fea