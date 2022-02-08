import pytorch_lightning as pl
import os, sys, timm
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from NN_modules import ResNeXt_block
from NN_ResNeXt  import Net_ResNext, DataModule

'''
For the architectures described below the notation is the following:
    'conv3_in3_out64' means a convolution layer with kernel_size (3,3) input's number of channels is 3 and output's is 64
    'avgPool8' is an average pool with kernel_size (8,8)
    'linear_in1024' is the last linear has the number output features equal to the number of classes and input equal to 1024
    A Coded-ResNeXt block is described as: [ Number_of_Input_Channels, Number_of_output_Channels, Bottleneck_width, 
                                            Stride_Of_the_second_convolutional_layer, Coding_scheme_ration, Probability_dropSubNN ]'
'''
d = 11 
dp_prob = 0.1
ARCHITECTURE_CIFAR_10 = [#stem. Resolution (32x32)
                        'conv3_in3_out64', 'bn2D_in64', 'relu',
                        #stage 1
                        [ 64, 256, d, 1, '10/10', 0.0 ], #Index Of Block: 0,  Resolution (32x32)
                        [ 256, 256, d, 1, '10/10', 0.0 ],
                        [ 256, 256, d, 1, '10/10', 0.0 ],                         
                        #stage 2
                        [ 256, 512, 2*d, 2, '5/10' , dp_prob ], #Index Of Block: 3,  Resolution (16x16)
                        [ 512, 512, 2*d, 1, '5/10' , dp_prob ], 
                        [ 512, 512, 2*d, 1, '5/10' , dp_prob ], 
                        #stage 3
                        [ 512, 1024, 4*d, 2, '3/10', dp_prob ], #Index Of Block: 6,  Resolution (8x8)
                        [ 1024, 1024, 4*d, 1, '3/10', dp_prob ], 
                        [ 1024, 1024, 4*d, 1, '3/10', dp_prob ],          
                        #Last layers
                        'avgPool8', 'flatten','linear_in1024' 
                        ]

d=6 
dp_prob = 0.1 
ARCHITECTURE_CIFAR_100 = [   #stem
                        'conv3_in3_out64', 'bn2D_in64', 'relu',
                        #stage 1
                        [ 64, 256, d, 1, '20/20', 0.0 ], #Index Of Block: 0,  Resolution (32x32)
                        [ 256, 256, d, 1, '20/20',0.0 ],
                        [ 256, 256, d, 1, '20/20',0.0 ],                         
                        #stage 2
                        [ 256, 512, 2*d, 2, '8/20',dp_prob ], #Index Of Block: 3,  Resolution (16x16)
                        [ 512, 512, 2*d, 1, '8/20',dp_prob ],  
                        [ 512, 512, 2*d, 1, '8/20',dp_prob ],  
                        #stage 3
                        [ 512, 1024, 4*d, 2, '4/20',dp_prob ], #Index Of Block: 6,  Resolution (16x16)
                        [ 1024, 1024, 4*d, 1, '4/20',dp_prob ],  
                        [ 1024, 1024, 4*d, 1, '4/20',dp_prob ],             
                        #Last layers
                        'avgPool8', 'flatten','linear_in1024' 
                        ]


ARGS = {
        #Problem's settings
        'Problem': 'Cifar100',#Choose between: 'Cifar10', 'Cifar100'  
        'Control': False, 

        #Architectural & interpretability choices:
        'architecture': ARCHITECTURE_CIFAR_100, #Must match the 'Problem'
        'Energy_normalization': True,
        'Same_code_Same_mask': True, #If True then two consecutive ResNeXt blocks that have the same coding scheme will
                                #also have the same dropout mask applied to them. Therefore out of N consecutive ResNeXt blocks
                                #with the same coding scheme it will be the first one dropSubNN_probability that counts.
                                #Generally it doesn't affect the overall performance but improves the binary classifiers.
        
        #RandAugment
        'no-augmentation': False,
        'timm-AutoAugment': 'rand-m2-n1',#CIFAR100:'rand-m2-n1'      CIFAR10: 'rand-m4-n3'
            
        #Optimization's algorithm choices:
        'train_batchSize': 64,#1 step in 8 core training is computing 8 gradients and syncing them, so it is analogous to training on 1 core with 8x the batch
        'test_batchSize': 64,
        'N_epochs': 300, 
        'SGD_lr_momentum': [0.1, 0.9, 5e-4, True],#(initial_lr, momentum, weight decay, Nesterov)
        'N_workers_dataloader': 1,

        #Losses choices:
        'LossDisentangle_type':'power4_threshold0.0', #Loss = diff(E_subNN, target_Energy, threshold)^power.
                                            #The diff function is: max{ |Energy_subNN-target_Energy|-threshold, 0}
        'LossDisentanglement_coef': 4, #Coefficient the loss_disentangle is multiplied with (Denoted $\mu$ in the paper)
        
        #Experiment Name
        'experiment_name_comet': 'Experiment_Name', 
        }


SAVING_PATH_CHECKPOINTS = "Path/To/Save/CheckPoints"#CHANGE



checkpoint_callback = ModelCheckpoint(every_n_epochs=5, filename=ARGS['experiment_name_comet'],  dirpath=SAVING_PATH_CHECKPOINTS)
dm = DataModule(ARGS)
my_model = Net_ResNext(ARGS)
trainer = Trainer( max_epochs=ARGS['N_epochs'], num_sanity_val_steps=0, tpu_cores=8, precision='bf16', callbacks=[checkpoint_callback])
trainer.fit(my_model, dm)



    
