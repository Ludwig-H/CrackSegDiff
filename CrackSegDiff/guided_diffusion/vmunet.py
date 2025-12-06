from .vmamba import VSSM
import torch
from torch import nn


class VMUNet(nn.Module):
    def __init__(self, 
                 input_channels=3, 
                 num_classes=2,
                 depths=[2, 2, 9, 2], 
                 depths_decoder=[2, 9, 2, 2],
                 drop_path_rate=0.2,
                 load_ckpt_path=None,
                ):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes

        self.vmunet = VSSM(in_chans=input_channels,
                           num_classes=num_classes,
                           depths=depths,
                           depths_decoder=depths_decoder,
                           drop_path_rate=drop_path_rate,
                        )
        
        if self.load_ckpt_path is not None:
            self.load_from()
    
    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits, skip_list = self.vmunet(x)
        if self.num_classes == 1: return torch.sigmoid(logits), skip_list
        else: return logits
    
    def load_from(self):
        if self.load_ckpt_path is not None:
            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_dict = modelCheckpoint['model']
            # Filter keys that exist and have matching shapes
            new_dict = {
                k: v for k, v in pretrained_dict.items() 
                if k in model_dict.keys() and v.shape == model_dict[k].shape
            }
            # Handle mismatch for patch_embed.proj.weight explicitly (e.g. 3 channels -> 9 channels)
            if 'patch_embed.proj.weight' in pretrained_dict and 'patch_embed.proj.weight' in model_dict:
                 v = pretrained_dict['patch_embed.proj.weight']
                 m_v = model_dict['patch_embed.proj.weight']
                 if v.shape != m_v.shape:
                     print(f"Adapting patch_embed.proj.weight from {v.shape} to {m_v.shape}")
                     # Assume [Out, In, H, W]. If In matches 3 and we need 9, repeat or pad.
                     if v.shape[1] == 3 and m_v.shape[1] == 9:
                         new_w = v.repeat(1, 3, 1, 1) # Repeat 3 times to fill 9 channels
                         new_dict['patch_embed.proj.weight'] = new_w

            model_dict.update(new_dict)
            # 打印出来，更新了多少的参数
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(pretrained_dict), len(new_dict)))
            self.vmunet.load_state_dict(model_dict)

            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("encoder loaded finished!")

            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_odict = modelCheckpoint['model']
            pretrained_dict = {}
            for k, v in pretrained_odict.items():
                if 'layers.0' in k: 
                    new_k = k.replace('layers.0', 'layers_up.3')
                    pretrained_dict[new_k] = v
                elif 'layers.1' in k: 
                    new_k = k.replace('layers.1', 'layers_up.2')
                    pretrained_dict[new_k] = v
                elif 'layers.2' in k: 
                    new_k = k.replace('layers.2', 'layers_up.1')
                    pretrained_dict[new_k] = v
                elif 'layers.3' in k: 
                    new_k = k.replace('layers.3', 'layers_up.0')
                    pretrained_dict[new_k] = v
            # Filter keys that exist and have matching shapes
            new_dict = {
                k: v for k, v in pretrained_dict.items() 
                if k in model_dict.keys() and v.shape == model_dict[k].shape
            }
            model_dict.update(new_dict)
            # 打印出来，更新了多少的参数
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(pretrained_dict), len(new_dict)))
            self.vmunet.load_state_dict(model_dict)
            
            # 找到没有加载的键(keys)
            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("decoder loaded finished!")