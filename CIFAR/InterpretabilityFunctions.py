import itertools 
import pdb
import torch  
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import statistics

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
      

def F1score_Presicion_Recall_BallancedAcc(pos_list, neg_list, thr, positives_perc=None):
    #Reducing the number of Negatives if positive_percentage is not None
    N_pos = len(pos_list)
    if positives_perc is not None:
        #when mode_loader == 'out_of_distr', N_pos has the previous value taken from when mode_loader == 'test'
        N_neg = int( (1/positives_perc-1)*N_pos )
        pdb.set_trace()
        assert N_neg <= len(neg_list)
        neg_list_reduced = random.sample(neg_list, N_neg)
    else:
        neg_list_reduced = neg_list
        N_neg = len(neg_list_reduced)
    pos, neg = torch.tensor(pos_list), torch.tensor(neg_list_reduced)

    P, N = pos.shape[0], neg.shape[0]
    assert P == N_pos and N == N_neg
    TruePos = (pos > thr).sum()
    TrueNeg = (neg < thr).sum()
    FalsePos = N - TrueNeg
    FalseNeg = P - TruePos
    F1score = 2*TruePos / (2*TruePos + FalseNeg + FalsePos)
    Precision = TruePos/(TruePos + FalsePos)
    Recall = TruePos / P
    BalancedAcc = 0.5* (TruePos / P + TrueNeg/N)
    return F1score, Precision, Recall, BalancedAcc


def Compute_Threshold_for_Best_F1(ActualPos, ActualNeg, positives_perc=None):
    #This function is doing an 1D search by testing the score as the threshold
    #is increasing and finally chooses the threshold with biggest score.
    MetricToMaximize = 0 # 0:F1score, 1:Precision, 2:Recall, 3:BalancedAcc
    MetricsName = ['F1-score', 'Precision', 'Recall', 'BalancedAcc']
    N_ITER = 10 #if a random subset of Negatives will be selected then we repeat the
                #process N_ITER times.
    if positives_perc is None: N_ITER = 1        
    Thr_best_list = [] #Gets the best found threshold per random realization
    for _ in range(N_ITER):
        bestScore = -1 #All scores are positive and the bigger they are the better.
        thr_best=None
        min_check_value = statistics.median(ActualPos)
        max_check_value = statistics.median(ActualNeg)
        d_value = (max_check_value-min_check_value)/100
        for thr in np.arange (min_check_value, max_check_value, d_value):
            Scores = F1score_Presicion_Recall_BallancedAcc(ActualPos, ActualNeg, thr, positives_perc)
            score = Scores[MetricToMaximize]
            if score > bestScore:
                bestScore = score
                thr_best = thr
        Thr_best_list.append(thr_best)
    thr_final = sum(Thr_best_list)/N_ITER
    
    #Printing Results    
    Scores = F1score_Presicion_Recall_BallancedAcc(ActualPos, ActualNeg, thr_final, positives_perc)
    print('The metric which it tried to maximize: ', MetricsName[MetricToMaximize] )
    print('and the threshold found was: {:0.3f}'.format( thr_final ))
    for i, metric_name in enumerate(MetricsName):
        print(metric_name, ': {:0.3f}%'.format(100*Scores[i]))
       
    return thr_final


def test_partNN_asBinaryClassifier(model, device,  class_of_model_part, test_loader, train_loader=None, out_of_distr_loader=None):
    #Takes a model that is for binary classification and tests its performance.
    MAX_N_SAMPLES_TRAIN = 50000 #Put 50000 for CIFAR10 and if desired less for CIFAR100
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
            print('========================================')     
            
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


def print_binaryThreshold_and_balancedAcc(Positives, Negatives):
    #This function can be used instead of "Compute_Threshold_for_Best_F1"
    #Instead of 1-D search it does a smarter search, i.e. a binary search.
    #It also optimizes for balanced Accuracy.
    #Assuming that as threshold increases the balancedAcc increases and then decreases 
    #We are starting from the right side of the mode
    N_ITER = 10
    pos, neg = torch.tensor(Positives), torch.tensor(Negatives)
    N_pos, N_neg = pos.shape[0], neg.shape[0]
    best_thr_balancedAcc = pos.mean()
    TruePos = (pos > best_thr_balancedAcc).sum()
    TrueNeg = (neg < best_thr_balancedAcc).sum()
    best_balancedAcc = 0.5*(TruePos/N_pos + TrueNeg/N_neg)
    D_thr = 10
    for _ in range(N_ITER):
        try_thr_right = best_thr_balancedAcc + D_thr
        try_thr_left = best_thr_balancedAcc - D_thr
        D_thr = D_thr/2
        TruePos = (pos > try_thr_right).sum()
        TrueNeg = (neg < try_thr_right).sum()
        balancedAcc_right = 0.5*(TruePos/N_pos + TrueNeg/N_neg)
        TruePos = (pos > try_thr_left).sum()
        TrueNeg = (neg < try_thr_left).sum()
        balancedAcc_left = 0.5*(TruePos/N_pos + TrueNeg/N_neg)
        if balancedAcc_right > best_balancedAcc and balancedAcc_right > balancedAcc_left:
            best_balancedAcc = balancedAcc_right
            best_thr_balancedAcc = try_thr_right
        elif balancedAcc_left > best_balancedAcc and balancedAcc_left > balancedAcc_right:
            best_balancedAcc = balancedAcc_left
            best_thr_balancedAcc = try_thr_left
    print('Best balanced accuracy is {0:.2f} and the threshold {1:.2f}'.format(best_balancedAcc,best_thr_balancedAcc))   
