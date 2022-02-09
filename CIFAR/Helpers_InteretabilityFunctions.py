import itertools 
import torch  
import numpy as np
import random
import statistics

def Compute_Threshold_for_Best_F1(ActualPos, ActualNeg, positives_perc=None):
    '''
    This function is doing an 1D search by testing the score as the threshold
    is increasing and finally chooses the threshold with biggest score.
    If positive_perc is not None and between (0,1) then from all the negative samples
    randomly a subset will be chosen so as the number of positive samples divided by 
    the total number of samples to be equal to positive_perc. In CIFAR-100 we 
    used positive_perc=0.1 so as to match the ratio of CIFAR-10 (which is by default 1/10).
    '''
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

  
def F1score_Presicion_Recall_BallancedAcc(pos_list, neg_list, thr, positives_perc=None):
    #Reducing the number of Negatives if positive_perc is not None
    N_pos = len(pos_list)
    if positives_perc is not None:
        #when mode_loader == 'out_of_distr', N_pos has the previous value taken from when mode_loader == 'test'
        N_neg = int( (1/positives_perc-1)*N_pos )
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


def print_binaryThreshold_and_balancedAcc(Positives, Negatives):
    '''
    This function can be used instead of "Compute_Threshold_for_Best_F1".
    Instead of 1-D search it does a smarter search, i.e. a binary search.
    It optimizes for balanced Accuracy.
    Assuming that as threshold increases the balancedAcc increases and then decreases 
    We are starting from the right side of the mode.
    (Note that "Compute_Threshold_for_Best_F1" was the one used for the paper)
    '''
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
