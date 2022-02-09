import torch  
import matplotlib.pyplot as plt
import random
from Helpers_InteretabilityFunctions import Compute_Threshold_for_Best_F1


def test_print_acc_removing_subNNs(indx_block, List_N_subNNs_to_remove, model, device, test_loader):
    #the indexing of the blocks start from 0 and for ResNeXt-29 goes till 8
    model.eval()
    with torch.no_grad():
        for k in List_N_subNNs_to_remove:
            N_tested_samples = 0
            N_correct_remove_act = 0
            N_correct_remove_inact = 0
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                N_tested_samples += data.shape[0]
                output, _, _, _, _ = model(data, targets,  mask_subNNs_scheme=(indx_block, k, 'remove_inactive'))    
                N_correct_remove_inact += output.argmax(dim=1).eq(targets).sum().item()  

                output, _, _, _, _ = model(data, targets,  mask_subNNs_scheme=(indx_block, k, 'remove_active'))    
                N_correct_remove_act += output.argmax(dim=1).eq(targets).sum().item()  
            avg_acc_remove_inact = N_correct_remove_inact/N_tested_samples
            avg_acc_remove_act =  N_correct_remove_act/N_tested_samples
            print('Accuracy if {} subNNs are removed from inactive: {:2.2%}   and from active: {:2.2%}'.format(k, avg_acc_remove_inact, avg_acc_remove_act))
      

def test_partNN_asBinaryClassifier(model, device,  class_of_model_part, test_loader, train_loader=None, out_of_distr_loader=None):
    # Takes a model that is for binary classification and tests its performance. The binary classifier is for the 
    # class "class_of_model_part". If out_of_distr_loader is given then the performance of the binary classifier is measured
    # also if negative samples come from the out_of_distr_loader. 
    MAX_N_SAMPLES_TRAIN = 50000 # If the train_loader is given then to compute the threshold for the binary classifier
                                # MAX_N_SAMPLES_TRAIN samples will be randomly chosen from training set. The threshold 
                                # that maximizes the F1-measure in that training subset will be the one used also for
                                # testing the binary classifier in the validation set.
                                #Put 50000 for CIFAR10 and if desired less for CIFAR100
    POSITIVES_PERCENTAGE = None #The number of Negative samples we consider is equal to (1/POSITIVES_PERCENTAGE - 1)*N_positives.
                               #So for CIFAR-100 which has 100 validation images used as positives for a class, we will randomly
                               #select (1/0.1 - 1)*100= 900 images as negatives.
                               #Put None for CIFAR10
    fig, axs = plt.subplots(1, 1, sharey=False, tight_layout=False, figsize = (20,8))
    Results = dict()
    model.eval()
    with torch.no_grad():
        print('For class {0}\n\n'.format(class_of_model_part))
        N_bins = 40                  

        #Run the model for all the loaders provided:
        if train_loader is not None: Modes_loader=[ ('train', train_loader), ]
        Modes_loader.append( ('test',test_loader) )
        if out_of_distr_loader is not None: Modes_loader.append( ('out_of_distr', out_of_distr_loader) )
        train_thr = None
        for mode_loader, loader in Modes_loader:
            print('\n========================================')
            print('Forward passes for the loader:  ',mode_loader)            
            print('==========================================')     
            
            if mode_loader in ['train', 'test']:
                ActualPos = []#the binary classifier has to predict True
            ActualNeg = []# and here False

            N_samples = 0
            for data, targets in loader:
                data, targets = data.to(device), targets.to(device)
                bin_class = class_of_model_part*torch.ones_like(targets)
                _, _, _, _, output_preSoftmax = model(data, bin_class,  mask_subNNs_scheme=('all', 'remove_inactive'))      
                output_preSoftmax = output_preSoftmax.to('cpu')
                targets = targets.to('cpu')
                for class_target , output in zip(targets, output_preSoftmax):
                    output_bc = output[class_of_model_part].item() 
                    if mode_loader == 'out_of_distr':
                        ActualNeg.append( output_bc )
                    else:
                        if class_target.item() == class_of_model_part:
                            ActualPos.append( output_bc )
                        else:
                            ActualNeg.append( output_bc )
                N_samples += data.shape[0]
                if mode_loader == 'train' and N_samples > MAX_N_SAMPLES_TRAIN:
                    break

            #Computing the performance of the thresholds and showing the performance of the binary classifier
            if mode_loader == 'train':
                train_thr = Compute_Threshold_for_Best_F1(ActualPos, ActualNeg, POSITIVES_PERCENTAGE)
            elif mode_loader == 'test':
                Results['test'] = (ActualPos, ActualNeg)
                test_thr = Compute_Threshold_for_Best_F1(ActualPos, ActualNeg, POSITIVES_PERCENTAGE)
                axs.hist(ActualPos, density=True, bins=N_bins, alpha=0.5)
                axs.hist(ActualNeg, density=True, bins=N_bins, alpha=0.5)
                _, precision_optimal, recall_optimal, _ = F1score_Presicion_Recall_BallancedAcc(ActualPos, ActualNeg, test_thr, POSITIVES_PERCENTAGE)
                _, precision, recall,_ = F1score_Presicion_Recall_BallancedAcc(ActualPos, ActualNeg, train_thr,  POSITIVES_PERCENTAGE)
                print('Precision and recall for optimal threshold ( {0:.4f}, {1:.4f})'.format(precision_optimal, recall_optimal))
                print('Precision and recall using threshold estimated from training dataset ( {0:.4f}, {1:.4f})'.format(precision, recall))       
            elif mode_loader == 'out_of_distr':
                Results['out_of_distr'] = ActualNeg
                axs.hist(ActualNeg, density=True, bins=N_bins, alpha=0.5)
                _, precision, recall,_ = F1score_Presicion_Recall_BallancedAcc(ActualPos, ActualNeg, train_thr, POSITIVES_PERCENTAGE)
                print('----USING OUT OF DISTRIBUTION NEGATIVES----')
                print('Precision and recall using threshold estimated from training dataset ( {0:.4f}, {1:.4f})'.format(precision,recall)) 
        plt.show()
    return Results
