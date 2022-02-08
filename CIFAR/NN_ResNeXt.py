import torch
from NN_modules import ResNeXt_block
import torch.optim as optim
from torchvision import transforms
import pytorch_lightning as pl
from torch.utils.data import  DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import CosineAnnealingLR
from timm.data.auto_augment import rand_augment_transform



class Net_ResNext(pl.LightningModule):
    def __init__(self, args):
        super(Net_ResNext, self).__init__()
        self.Problem = args['Problem']
        if self.Problem == 'Cifar10':
            self.N_classes = 10
        elif self.Problem == 'Cifar100':
            self.N_classes = 100     

        self.architecture = args['architecture']  
        self.disentangle_loss_coef= args['LossDisentanglement_coef'] if not args['Control'] else 0
        self.lr, self.mom, self.weight_d, self.nstv = args['SGD_lr_momentum']
        self.N_epochs = args['N_epochs']
        #create network
        self.net_list, self.net_types = self.MakeListOfModules(args)


    #===============
    #Creating the NN
    #===============   
    def MakeListOfModules(self,args):
        List_Modules = nn.ModuleList()
        List_types = []
        res_block_counter = 0
        for Mod_seq in self.architecture:
            if isinstance(Mod_seq, list):                
                Name_block = 'ResNeXt_block_'+str(res_block_counter)
                List_types.append(Name_block)
                res_block_counter += 1
                N_channels_in, N_channels_out, Bottleneck_d, stride, coding_scheme, dropout_prob = Mod_seq
                new_block = ResNeXt_block(N_channels_in, N_channels_out, Bottleneck_d, stride, coding_scheme, dropout_prob, args)  
                List_Modules.append( new_block )
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
                Nneurons_out = self.N_classes
            name_mod, Nneurons_in = name_mod.split('_in')
            return nn.Linear(int(Nneurons_in), int(Nneurons_out), bias=False) #SOS CHANGED the bias to False. Not that I saw any difference
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
        # If mask_subNNs_scheme is used in test/validation mode (when there is also no dropSubNN):
        #     - a tupple (i, k, 'remove_active') then from the i-th ResNeXt block (starting the counting from 0)
        #       k randomly chosen subNNs out of the set of 'active' for the classes will be removed. The set of
        #       active is determined by seeing from 'targets' the class of the input image and choosing the 
        #       corresponding subset from the coding scheme.
        #     - a tupple (i, k, 'remove_inactive') instead of 'active' is given then k subNNs will be randomly 
        #       removed from the set of corresponding inactives ones. 
        #     - a tupple ('all', 'remove_inactive') then you turn the NN into a binary classifier as only the 
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
                    Losses_disentangle.append(loss_disentangle.item())
                if decodingEnergies is not None:
                    DecodeEnergies.append( decodingEnergies )
        
        preds = F.log_softmax(x, dim=1) 
        return [preds, loss_disentangle_total, Losses_disentangle, DecodeEnergies, x]


    #====================
    # Training/Test step
    #====================
    def Compute_classification_loss(self, preds, targets):
        return F.nll_loss(preds, targets, reduction='mean') 

    
    def training_step(self, batch, batch_idx):      
        x, targets = batch          
        preds, loss_disentangle_total, _, _, _ = self.forward(x, targets)           
        loss_class = self.Compute_classification_loss(preds, targets)        
        loss = self.disentangle_loss_coef*loss_disentangle_total + loss_class
        train_acc = preds.argmax(dim=1).eq(targets).sum().item()/x.shape[0]
        self.log("train/acc", train_acc, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/loss_classification", loss_class.item(), on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/loss_disentangle_total", loss_disentangle_total.item(), on_step=True, on_epoch=True, sync_dist=True)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        x, targets = batch           
        preds, loss_disentangle_total, Losses_disentangle, predsDecodeEnergies, _ = self.forward(x, targets)
        test_acc = preds.argmax(dim=1).eq(targets).sum().item()/x.shape[0]
        loss_class = self.Compute_classification_loss(preds, targets)
        self.log("val/acc", test_acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/loss_classification", loss_class.item(), on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/loss_disentangle_total", loss_disentangle_total.item(), on_step=False, on_epoch=True, sync_dist=True)
        for i, loss_dis_block in enumerate( Losses_disentangle ): 
            self.log("val/loss_dis_block_"+str(i), loss_dis_block, on_step=False, on_epoch=True, sync_dist=True)
        for i, preds_energies_block in enumerate( predsDecodeEnergies ):
            test_decodeEnergy_acc = preds_energies_block.argmax(dim=1).eq(targets).sum().item()/x.shape[0]
            self.log("val/acc_decodeEnergy_block_"+str(i), test_decodeEnergy_acc, on_step=False, on_epoch=True, sync_dist=True) 
 
            
    def configure_optimizers(self,):
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.mom, weight_decay=self.weight_d, nesterov=self.nstv)  
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.N_epochs, eta_min=1e-5, verbose=True)
        for epoch_pass in range(self.trainer.current_epoch):
            lr_scheduler.step()
        return ([optimizer], [lr_scheduler])
         





class DataModule(pl.LightningDataModule):
    def __init__(self, args, data_dir= "./"):
        super().__init__()
        self.data_dir = data_dir
        self.Num_workers_loader = args['N_workers_dataloader']
        self.Problem = args['Problem']
        self.train_batchSize = args['train_batchSize']
        self.test_batchSize = args['test_batchSize']
        if self.Problem == 'Cifar10':
            MEAN_CIFAR = (0.4914, 0.4822, 0.4465)
            STD_CIFAR =(0.2023, 0.1994, 0.2010)# (0.2470, 0.2435, 0.2616) 
            self.N_classes = 10            
        elif self.Problem == 'Cifar100':
            MEAN_CIFAR =  (0.5071, 0.4867, 0.4408)
            STD_CIFAR = (0.2673, 0.2564, 0.2762)
            self.N_classes = 100    
        
        #For validation: 
        self.transform_val = transforms.Compose([
                        transforms.ToTensor(), 
                        transforms.Normalize( MEAN_CIFAR, STD_CIFAR)])        

        #For Training
        if args['no-augmentation']:
            self.transform_train = self.transform_val
        else:
            List_of_transforms = []
            if args['timm-AutoAugment'] is not None:
                print('--> Using rand augment with: ',  args['timm-AutoAugment'])
                aa_params = dict( translate_const = int(32 * 0.45),
                                    img_mean      = tuple([min(255, round(255 * x)) for x in MEAN_CIFAR]),   )
                List_of_transforms += [rand_augment_transform( args['timm-AutoAugment'], aa_params)]
            
            List_of_transforms += [transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(), 
                                    transforms.Normalize( MEAN_CIFAR, STD_CIFAR)]
            self.transform_train = transforms.Compose(List_of_transforms)
                                

    def prepare_data(self):
        if self.Problem == 'Cifar10':
            CIFAR10(root=self.data_dir, train= True, download=True, transform=self.transform_train)
            CIFAR10(root=self.data_dir, train=False, download=True, transform=self.transform_val)
        elif self.Problem == 'Cifar100':
            CIFAR100(root=self.data_dir, train= True, download=True, transform=self.transform_train)
            CIFAR100(root=self.data_dir, train=False, download=True, transform=self.transform_val)

                
    def train_dataloader(self):
        if self.Problem == 'Cifar10':
            dataset_train = CIFAR10(self.data_dir, train=True, transform=self.transform_train)
        elif self.Problem == 'Cifar100':
            dataset_train = CIFAR100(self.data_dir, train=True, transform=self.transform_train)
        loader = DataLoader(dataset_train, batch_size=self.train_batchSize, num_workers=self.Num_workers_loader, shuffle=True, drop_last=True)
        return loader


    def val_dataloader(self): 
        if self.Problem == 'Cifar10':
            dataset_val = CIFAR10(self.data_dir, train=False, transform=self.transform_val)
        elif self.Problem == 'Cifar100':
            dataset_val = CIFAR100(self.data_dir, train=False, transform=self.transform_val)
        loader = DataLoader(dataset_val, batch_size=self.test_batchSize, num_workers=self.Num_workers_loader )
        return loader 






