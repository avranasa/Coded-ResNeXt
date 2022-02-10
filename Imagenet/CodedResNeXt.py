import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from NN_modules import ResNeXt_block
from timm.models.registry import register_model
from timm.models.helpers import load_checkpoint

class CodedResNeXt(nn.Module):
    def __init__(self, **kwargs):
        super( CodedResNeXt, self).__init__()
        self.num_classes = 1000
        self.architecture = kwargs.get('Architecture')
        self.default_cfg = {'architecture':self.architecture,
                           'control':kwargs.get('Control')}
        self.num_codedBlocks = 0
        #create network
        self.net_list, self.net_types = self.MakeListOfModules(**kwargs)


    #===============
    #Creating the NN
    #===============   
    def MakeListOfModules(self,**kwargs):
        List_Modules = nn.ModuleList()
        List_types = []
        res_block_counter = 0
        for Mod_seq in self.architecture:
            if isinstance(Mod_seq, list):                       
                Name_block = 'ResNeXt_block_'+str(res_block_counter)
                List_types.append(Name_block)
                res_block_counter += 1
                N_channels_in, N_channels_out, Bottleneck_d, stride, coding_scheme, dropout_prob = Mod_seq
                new_block = ResNeXt_block(N_channels_in, N_channels_out, Bottleneck_d, stride, coding_scheme, dropout_prob, **kwargs)  
                List_Modules.append( new_block )
                if new_block.ratio_active < 1:
                    self.num_codedBlocks += 1
                #print(Name_block, ' has ', sum(p.numel() for p in new_block.parameters() if p.requires_grad)/1e6, 'M params')
            else:     
                List_Modules.append(self.MakeModule(Mod_seq))                   
                List_types.append(Mod_seq)
        return List_Modules, List_types


    def MakeModule(self, name_mod, Nch_in=None, Nch_out=None):    
        if 'conv' in name_mod:
            stride = 1
            if 'str' in name_mod:
                name_mod, stride = name_mod.split('_str')
                stride = int(stride)
            if 'out' in name_mod:
                name_mod, Nch_out = name_mod.split('_out')
                name_mod, Nch_in = name_mod.split('_in')
                Nch_out = int(Nch_out)
                Nch_in = int(Nch_in)
            k = int(name_mod.split('conv',1)[-1])
            conv_module = nn.Conv2d( Nch_in, Nch_out, kernel_size=(k,k), stride=stride, padding=int((k-1)/2), bias=False)  
            nn.init.kaiming_normal_(conv_module.weight, mode='fan_out', nonlinearity='relu')
            assert k%2 == 1, "The kernel size has to be [k,k] with k an odd number"
            return conv_module
        elif 'linear' in name_mod:
            if '_out' in name_mod:
                name_mod, Nneurons_out = name_mod.split('_out')
            elif Nch_out is not None:
                Nneurons_out = Nch_out
            else:
                Nneurons_out = self.num_classes
            name_mod, Nneurons_in = name_mod.split('_in')
            return nn.Linear(int(Nneurons_in), int(Nneurons_out), bias=False)
        elif 'relu' in name_mod:
            return nn.ReLU(inplace=True) 
        elif 'bn2D' in name_mod:
            if 'in' in name_mod:
                name_mod, Nch_in = name_mod.split('_in')
                Nch_in = int(Nch_in)
            return nn.BatchNorm2d(Nch_in, momentum=0.1, affine=True) 
        elif 'maxPool' in name_mod:
            stride, pad = None, 0 
            if 'pad' in name_mod:
                name_mod, pad = name_mod.split('_pad')
                pad = int(pad)
            if 'str' in name_mod:
                name_mod, stride = name_mod.split('_str')
                stride = int(stride)
            k = int(name_mod.split('maxPool',1)[-1])
            return nn.MaxPool2d(kernel_size=k, stride=stride, padding=pad)
        elif 'adaptiveAvgPool' in name_mod:
            return nn.AdaptiveAvgPool2d(1)
        elif 'avgPoolWithMean' in name_mod:
            return avgPoolWithMean()
        elif 'avgPool' in name_mod:
            k = int(name_mod.split('avgPool',1)[-1])
            return nn.AvgPool2d(kernel_size=(k,k))
        elif 'flatten' in name_mod:
            return nn.Flatten(start_dim=1)
        else:
            print(name_mod,' not in List_Modules available')
            raise ValueError

              
    #=========
    # Forward
    #========= 
    def forward(self, x, targets, mask_subNNs_scheme = None): 
        # The mask_subNNs_scheme is used only at test mode (when there is also no dropSubNN). Its possible values:
        #     - a tupple (i, k, 'remove_active'): Then from the i-th ResNeXt block (starting the counting from 0)
        #       k randomly chosen subNNs out of the set of 'active' for the classes will be removed. The set of
        #       active is determined by seeing from 'targets' the class of the input image and choosing the 
        #       corresponding subset from the coding scheme.
        #     - a tupple (i, k, 'remove_inactive'): Then instead of 'active' is given then k subNNs will be randomly 
        #       removed from the set of corresponding inactives ones. 
        #     - a tupple ('all', 'remove_inactive'): Used to turn the NN into a binary classifier. Only the 
        #       the subNNs dedicated to the corresponding classes of targets are not masked.
        #     - None. No masking is applied
        # In the training mode it is asserted to be None. 
        assert not( self.training and mask_subNNs_scheme is not None)
        DecodeEnergies = []
        Losses_disentangle = []
        mask_dp_info = None
        loss_disentangle_total = torch.zeros(size=(1,), device=x.device)       
        
        for type_layer, layer in zip(self.net_types, self.net_list):             
            if 'ResNeXt_block' not in type_layer: 
                x = layer(x)
            else:
                #The variable "passed_the_layer" is useful only when mask_subNNs_scheme is not None. 
                #In that case it prohibits passing through one block twice.
                passed_the_layer = False
                if mask_subNNs_scheme is not None:
                    if 'ResNeXt_block_'+str(mask_subNNs_scheme[0]) == type_layer or mask_subNNs_scheme[0]=='all':
                        x, loss_disentangle, decodingEnergies, mask_dp_info = layer(x, targets, mask_subNNs_scheme, mask_dp_info)
                        passed_the_layer = True
                if not passed_the_layer:
                    x, loss_disentangle, decodingEnergies, mask_dp_info = layer(x, targets, mask_dp_info=mask_dp_info)

                #store the losses and predictions from decoding energies
                if loss_disentangle is not None:
                    loss_disentangle_total = loss_disentangle_total + loss_disentangle
                    Losses_disentangle.append(loss_disentangle)
                if decodingEnergies is not None:
                    DecodeEnergies.append( decodingEnergies )

        return  [x, loss_disentangle_total, Losses_disentangle, DecodeEnergies]


class avgPoolWithMean(nn.Module):
    def __init__(self,):
        super(avgPoolWithMean, self).__init__()

    def forward(self, x):
        return x.mean((2, 3))


@register_model 
def create_CodedResNeXt(pretrained=False, checkpoint_path='', **kwargs):

    model = CodedResNeXt(**kwargs)
    if checkpoint_path:
        load_checkpoint(model, checkpoint_path)

    return model

