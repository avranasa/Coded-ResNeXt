import torch
import torch.nn as nn
import pdb
import math
import random
import torch.nn.functional as F
from collections import OrderedDict
from subNNs_codes import *
import sys


class ResNeXt_block(nn.Module):
    def __init__(self, N_channels_in, N_channels_out, Bottleneck_d, stride, coding_scheme_name, dropout_prob, args):
        super(ResNeXt_block, self).__init__()

        # Creating the meta-data needed to built the modules of the ResNeXt block
        self.LossDisentangle_type = args['LossDisentangle_type']  
        self.EnergyNormalization = args['Energy_normalization']
        self.SameCodeSameMask = args['Same_code_Same_mask']
        self.dropOut_prob = dropout_prob
        if 'power' in self.LossDisentangle_type:
            List_str = self.LossDisentangle_type.split('_')            
            self.p_dis = float(List_str[-2].replace('power',''))
            self.thr_dis = float(List_str[-1].replace('threshold',''))
        self.N_channels_in, self.N_channels_out, self.Bottleneck_d = N_channels_in, N_channels_out, Bottleneck_d


        #Create the requisites for coding
        self.coding_scheme_name = coding_scheme_name
        if args['Problem'] == 'Cifar10':
            coding_scheme = SUBNNS_CODING_CIFAR10[self.coding_scheme_name]
        elif args['Problem'] == 'Cifar100':
            coding_scheme = SUBNNS_CODING_CIFAR100[self.coding_scheme_name]
        self.N_classes = len(coding_scheme)
        self.N_subNNs = len(coding_scheme[0])
        if args['Control']:
            #changing the  coding to N/N
            coding_scheme = ['1'*self.N_subNNs for cl in range(self.N_classes)]
        Indxs_perClasses = self.Transform_Strings_2_Indxs_perClass(coding_scheme)#In ResNeXt paper N_subNNs is referred as cardinality 
        self.register_buffer('Mask_perClass', self.Make_Masks_perClass_OneLayerSubNNs(self.Bottleneck_d, self.N_subNNs, Indxs_perClasses) )        
        

        self.N_subNNs_active = torch.sum( self.Mask_perClass[0] ) #shape=[N_classes, N_subNNs]
        N_channels_internally = self.N_subNNs*self.Bottleneck_d   
        for cl in range(self.N_classes):
            # Assert the assumption that all class have the same number of active subNNs as the rest
            # i.e. that each codeword has the same number of 1.
            assert torch.sum( self.Mask_perClass[0] ) == self.N_subNNs_active
        self.ratio_active = self.N_subNNs_active / self.N_subNNs
        # Sanity check
        assert (self.ratio_active == 1) or (self.N_subNNs_active >= 1 and self.N_subNNs_active<self.N_subNNs)         
        

        #Building the neural network modules of the main branch
        BatchNorm_mom = 0.1
        self.conv_reduce = nn.Conv2d(N_channels_in, N_channels_internally, kernel_size=1, bias=False)
        self.bn_reduce = nn.BatchNorm2d(N_channels_internally, momentum=BatchNorm_mom)
        self.conv_internal = nn.Conv2d(N_channels_internally, N_channels_internally, kernel_size=3, stride=stride, padding=1, groups=self.N_subNNs, bias=False )
        self.bn_internal = nn.BatchNorm2d(N_channels_internally, momentum=BatchNorm_mom)   
        if self.ratio_active < 1.0:
            self.conv_expand = nn.Conv2d(N_channels_internally, self.N_subNNs*N_channels_out, kernel_size=1, groups=self.N_subNNs, bias=False)            
        else:
            self.conv_expand = nn.Conv2d(N_channels_internally, N_channels_out, kernel_size=1, bias=False)
        self.bn_expand = nn.BatchNorm2d(N_channels_out, momentum=BatchNorm_mom)

            
        #Building the neural network modules of the skip connection  
        assert stride in [1,2]   
        if stride != 1 or N_channels_in != N_channels_out:
            self.skip_connection = nn.Sequential(OrderedDict([
                ('conv_skip', nn.Conv2d(N_channels_in, N_channels_out, kernel_size=1, stride=stride, bias=False)),
                ('bn_skip',  nn.BatchNorm2d(N_channels_out, momentum=BatchNorm_mom))
            ]))   
        else:
            self.skip_connection = nn.Identity()

        
        #Initializations:
        nn.init.kaiming_normal_(self.conv_reduce.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv_internal.weight, mode='fan_out', nonlinearity='relu')      
        nn.init.kaiming_normal_(self.conv_expand.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.bn_expand.weight)#zeroing the last bn's gamma
        if stride != 1 or N_channels_in != N_channels_out:
            nn.init.kaiming_normal_(self.skip_connection._modules['conv_skip'].weight, mode='fan_out', nonlinearity='relu')


    def Transform_Strings_2_Indxs_perClass(self, Indxs_perClasses):
        #Returns "Indxs_perClasses_oneLayer"  which is a list of lenngth=number_of_classes. This list contains
        #lists of len == Number_of_active_for_class_subNNs containing the indexes of subNNs that are active for each class
        N_classes = len(Indxs_perClasses)
        N_subNNs = len(Indxs_perClasses[0])
        Indxs_perClasses_oneLayer = [[] for i in range(N_classes)]
        for cl in range(N_classes):
            for i in range(N_subNNs):
                if Indxs_perClasses[cl][i] == '1':
                    Indxs_perClasses_oneLayer[cl].append(i)   
        return Indxs_perClasses_oneLayer


    def Make_Masks_perClass_OneLayerSubNNs(self, N_ChannelsPerSubNN, N_subNNs, subNNs_indx_perClass):
        #Transforming subNNs_indx_perClass to Tensors with shape=[N_classes, N_subNNs]. 
        N_classes = len(subNNs_indx_perClass)
        Masks_PerClass = [torch.zeros(size=(N_subNNs,),dtype=bool) for _ in range(N_classes) ]         
        for cl in range(N_classes):
            indx = subNNs_indx_perClass[cl]
            Masks_PerClass[cl][indx] = True    
        #list of length=Number_of_Layers, with tensor of shape=[N_classes, N_subNNs]:
        Masks_PerClass = torch.stack([Masks_PerClass[cl] for cl in range(N_classes)])            
        return Masks_PerClass


    def mask_all_inactive(self, x, targets):
        #Used in test time to check the binary classifiers
        if self.ratio_active == 1:
            return x #there are no inactive subNNs
        return self.mask_k_activeORinactive_subNNs(x, targets, int(self.N_subNNs-self.N_subNNs_active), 'remove_inactive')


    def mask_k_activeORinactive_subNNs(self, x, targets, k, mode):
        # Used in test time to check the binary classifiers and to verify that each subNN is working only for the
        #desired subset of classes which it is assigned.
        # It randomly chooses k subNNs out of the set of the active or inactive subNNs
        #depending on the mode and zero their output
        BatchSize, N_subNNs, Ch_perSubNN, H, W = x.shape    
        assert mode in ['remove_active', 'remove_inactive'] 
        mask = torch.ones(size=(BatchSize,self.N_subNNs), dtype=bool).to( x.device)
        auxilary_ind = torch.repeat_interleave(torch.arange(BatchSize).to(x.device), k)
        if mode == 'remove_active': 
            assert k <= self.N_subNNs_active, 'You are trying to remove more active subNNs than there exists'
            ind_drop = torch.multinomial(1.0*self.Mask_perClass[targets], num_samples=k)
        elif mode == 'remove_inactive':
            assert k <= self.N_subNNs-self.N_subNNs_active, 'You are trying to remove more inactive subNNs than there exists'
            ind_drop = torch.multinomial(1.0*(~self.Mask_perClass[targets]), num_samples=k)
        mask[auxilary_ind, ind_drop.flatten()] = False
        return x * mask.view(BatchSize, N_subNNs,1,1,1)
        

    def mask_grads(self, x, mask):      
        if not self.training or self.ratio_active == 1:
            return x 
        mask = mask.view(x.shape[0], self.N_subNNs, 1 ,1, 1)
        return mask*x + torch.logical_not(mask)*x.detach()

    
    def loss_disentangle(self, mask, energy_per_subNN):    
        #assert torch.all( (energy_per_subNN.sum(dim=1)-self.N_subNNs).abs() <5e-2  )  #if dropout_prob = 0       
        if 'power' in self.LossDisentangle_type:  
            if self.thr_dis == 0.0 and self.p_dis%2 == 0:
                #My most common choises. So I do them here without any unnecessary operation. 
                dist = energy_per_subNN * self.ratio_active - 1.0*mask
                return torch.mean( dist**self.p_dis )
            elif self.thr_dis == 0.0 and self.p_dis == 1:
                dist = energy_per_subNN * self.ratio_active - 1.0*mask
                return dist.abs().mean()
            else:
                #inactive pushed to zero & active to one
                E_truncated = F.relu( (energy_per_subNN * self.ratio_active - 1.0*mask).abs() - self.thr_dis ) 
                loss_dis = torch.mean( E_truncated**self.p_dis )
                return loss_dis
        else:
            return None


    def decodeFromEnergies(self, energy_per_subNN):
        E = energy_per_subNN.unsqueeze(dim=1)#dim = [BatchSize,1,N_subNNs]    
        E = E / E.sum(dim=2,keepdim=True) * self.N_subNNs_active
        M = 1.0*self.Mask_perClass.unsqueeze(dim=0)#dim = [1,N_classes,N_subNNs]
        # Compute the distance of signal E to all codewords of mask M.        
        dist = ( (E-M)**2 ).sum(dim=2).sqrt()#dim = [BatchSize,N_classes] #/self.N_subNNs
        
        #Choose the minimum distance. Used Softmin (even though unnecessare) to be in the spirit of classification 
        confidence = F.softmax( -dist, dim=1)# the minus so as afterwards to take the softmin
        return confidence

    
    def dropout_subNNs(self, x, maskActiveSubNNs, mask_info=None):
        # Shape of x: [BatchSize, self.N_subNNs*N_channels_perSubNN, H, W]
        # Shape of maskActiveSubNNs: [BatchSize, N_subNNs]
        if not self.training or self.dropOut_prob == 0:   return x, None

        if not self.SameCodeSameMask:
            return F.dropout3d(x, p=self.dropOut_prob, training=self.training, inplace=True), None            
        else:
            if  mask_info is None or self.coding_scheme_name != mask_info['code_name']:
                #if mask_info is None then it is the first block with dropout. The second clause checks if this blocks coding scheme has changed 
                #compared to the previous one
                mask = torch.bernoulli( (1-self.dropOut_prob)*torch.ones_like(maskActiveSubNNs) ).view(x.shape[0],self.N_subNNs,1,1,1)/ (1-self.dropOut_prob)
            else:
                mask = mask_info['mask']
            return mask*x, {'code_name':self.coding_scheme_name,  'mask':mask}


    def interpretability_operations(self, x, targets, mask_subNNs_scheme, mask_dp_info=None):
        maskActiveSubNNs = self.Mask_perClass[targets]
        BatchSize, C, H, W = x.shape 
        x = x.view(BatchSize,  self.N_subNNs, -1, H, W)

        #Masking for the interpretability plots. Used mainly for in test time.
        if mask_subNNs_scheme is not None:
            if mask_subNNs_scheme[0] == 'all':
                x = self.mask_all_inactive(x, targets)
            else:
                x = self.mask_k_activeORinactive_subNNs(x, targets, mask_subNNs_scheme[1], mask_subNNs_scheme[2])

        #Computing the energies per SubNN and applying energy normalization
        energy_per_subNN = torch.mean(x**2, dim=[2,3,4]) #dim = [BatchSize,N_subNNs]
        if self.EnergyNormalization:
            rms = torch.sqrt(  energy_per_subNN.mean(dim=1)) #+1e-8 #dim = [BatchSize,]
            x = x/rms.view(BatchSize,1,1,1,1)  #each active subNN is pushed to have mean energy of 1
            energy_per_subNN = energy_per_subNN / rms.pow(2).view(BatchSize,1)   

        #Rest of interpretability operations  
        loss_disentangle = self.loss_disentangle(maskActiveSubNNs, energy_per_subNN)
        decodingEnergies = self.decodeFromEnergies(energy_per_subNN.detach())  

        #DropSubNNs
        x, mask_dp_info = self.dropout_subNNs(x, maskActiveSubNNs, mask_dp_info)
        
        return x.sum(dim=1), loss_disentangle, decodingEnergies, mask_dp_info


    def forward(self, x, targets, mask_subNNs_scheme=None, mask_dp_info=None):
        residual = self.skip_connection(x)
        x = self.conv_reduce(x)
        x = F.relu(self.bn_reduce(x), inplace=True)
        x = self.conv_internal(x) 
        x = F.relu(self.bn_internal(x), inplace=True)
        x = self.conv_expand(x)   
        if self.ratio_active < 1:    
            x, loss_disentangle, decodingEnergies, mask_dp_info = self.interpretability_operations(x, targets, mask_subNNs_scheme, mask_dp_info) 
        else:
            loss_disentangle, decodingEnergies = None, None   
        x = self.bn_expand(x)
        x = F.relu( x + residual, inplace=True)
        
        return x, loss_disentangle, decodingEnergies, mask_dp_info
        

