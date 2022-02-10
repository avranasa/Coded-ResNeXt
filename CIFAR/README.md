In all our experiments we used google's COLAB notebook and implemented with Pytorch Lightning. The file "Coded_ResNeXt_CIFAR.ipynb" contains the
notebook. Click it and above left there is the option of clicking the box [Open in Colab] which will open a COLAB notebook
where the models for CIFAR-10 and CIFAR-100 can be trained. The accelarator used was TPU. We also used the High-RAM option (not necessary for CIFAR experiments) but 
(at the moment) is available only for those with Pro or Pro+ account.

It is easy to choose the architecture, the dataset and the hyperparameters after accessing the notebook. For example the hyperparameters
can be set from the cell:
```python
ARGS = {
        #~~~~~~Problem's settings~~~~~~
        'Problem': 'Cifar10',#Choose between: 'Cifar10', 'Cifar100'  
        'Control': False, #If True then the original ResNeXt is trained.

        #~~~~~~Architectural & interpretability choices~~~~~~
        'Energy_normalization': True,
        'Same_code_Same_mask': True, #If True then two consecutive ResNeXt blocks that have the same coding scheme will
                                #also have the same dropout mask applied to them. Therefore out of N consecutive ResNeXt blocks
                                #with the same coding scheme it will be the first one dropSubNN_probability that counts.
   
        #~~~~~~Losses choices~~~~~~
        'LossDisentangle_type':'power4_threshold0.0', #Loss = diff(E_subNN, target_Energy, threshold)^power.
                                            #The diff function is: max{ |Energy_subNN-target_Energy|-threshold, 0}
        'LossDisentanglement_coef': 6, #Coefficient the loss_disentangle is multiplied with (Denoted $\mu$ in the paper)
                                     #Generally it doesn't affect the overall performance but improves the binary classifiers.
        
        #~~~~~~RandAugment~~~~~~
        'no-augmentation': False, #If true then no data augmentation will be used
        'timm-AutoAugment': 'rand-m2-n1',#The notation is from "https://fastai.github.io/timmdocs/RandAugment"
                                         #For CIFAR100:'rand-m2-n1' and for CIFAR10 'rand-m4-n3'
            
        #~~~~~~Optimization's algorithm choices~~~~~~
        'train_batchSize': 64,#1 step in 8 core training is computing 8 gradients and syncing them, so effectively the size is 8*64=512
        'test_batchSize': 64,
        'N_epochs': 300, 
        'SGD_lr_momentum': [0.1, 0.9, 5e-4, True],#(initial_lr, momentum, weight decay, Nesterov)
        'N_workers_dataloader': 4,
        }
```
