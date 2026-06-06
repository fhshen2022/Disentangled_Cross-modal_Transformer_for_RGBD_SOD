import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch.nn as nn
from .t2t_vit import T2t_vit_t_14
from .Transformer import Transformer
from .Decoder_DFP import Decoder_DFP
#import seaborn as sns;sns.set()





class ImageDepthNet(nn.Module):
    def __init__(self, args):
        super(ImageDepthNet, self).__init__()
        # VST Encoder
        self.rgb_backbone = T2t_vit_t_14(pretrained=True, args=args)
        self.depth_backbone = T2t_vit_t_14(pretrained=True, args=args)
        # VST Convertor
        self.transformer = Transformer(embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.)
        # VST Decoder
        self.decoder = Decoder_DFP(embed_dim=384, token_dim=64, depth=2, img_size=args.img_size)


    def forward(self, image_Input, depth_Input):

        B, _, _, _ = image_Input.shape
        # VST Encoder
        #1_16 [B*(14*14)*384], 1_8 [B*(28*28)*64], 1_4 [B*(56*56)*64]
        rgb_fea_1_16, rgb_fea_1_8, rgb_fea_1_4 = self.rgb_backbone(image_Input)#att[0:14]
        depth_fea_1_16, depth_fea_1_8, depth_fea_1_4 = self.depth_backbone(depth_Input)#att[14:28]
        
        #Converter
        rgb_fea_1_16, depth_fea_1_16 = self.transformer(rgb_fea_1_16, depth_fea_1_16)
        
        # VST Decoder
        outputs,comple_loss,consis_pre = self.decoder(rgb_fea_1_16, rgb_fea_1_8, rgb_fea_1_4,depth_fea_1_16, depth_fea_1_8, depth_fea_1_4)
        
        #return outputs
        return outputs,comple_loss,consis_pre#,comple_loss2,comple_loss3
