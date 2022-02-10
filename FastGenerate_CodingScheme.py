'''
A much faster greedy method which in all experiments gave of same quality results as the slow 
with (at least) exponential complexity different methods tried.  
'''
#!apt-get install libgmp-dev libmpfr-dev libmpc-dev
#!apt-get install python3-gmpy2
#!pip3 install gmpy
from typing import Counter
import numpy as np
import sys
from tqdm import tqdm
import random
from itertools import combinations, count
import gmpy2
import operator as op
from functools import reduce
import pdb
import pickle

def nCr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  # or / in Python 2


def snoob(x):
    # This function returns next higher number with same number of set bits as x.
    # Taken from "https://www.geeksforgeeks.org/next-higher-number-with-same-number-of-set-bits/"
    
    #IMPORTANT TO GET CODEWORDS IN SORTED MANNER...if not the legit_words list the 
    #algorithm first finds are of much smaller cardinality... an analogy is if you want in an 
    #fixed area to fill maximum number of balls it is better to progressively position the balls 
    #one close to the others than randomly placing them.

    next_x = 0
    if(x):
        rightOne = x & -(x)
        nextHigherOneBit = x + int(rightOne)
        rightOnesPattern = x ^ int(nextHigherOneBit)
        rightOnesPattern = (int(rightOnesPattern) /
                            int(rightOne))
        rightOnesPattern = int(rightOnesPattern) >> 2
        next_x = nextHigherOneBit | rightOnesPattern
    return next_x


def check_solution(sol):
    C = len(sol)
    H = np.zeros(shape=(C,C), dtype=int)
    for i in range(C):
        for j in range(i+1,C):
            H[i,j]= gmpy2.hamdist(sol[i], sol[j])
    print('printing all hamming distances')
    np.set_printoptions(edgeitems=8, linewidth=200)
    print(H)
    maxDist = np.amax(H)
    NumNonZero =int( C*(C-1)/2  )
    for d in range(D,maxDist+1):
        Num_with_dist_d = np.count_nonzero(H == d)
        print("Percentage of pairs with distance {0}: {1:.2f}".format(d,100*Num_with_dist_d/NumNonZero))


def FindOneGreedySolution(legit_words, SizeSol):
    global N, D
    
    list_avail_words = legit_words.copy()
    random.shuffle(list_avail_words)#so as every time FindOneGreedySolution() is called new solution will be retrieved
    
    #Pick word by word by greedily trying to keep the sums of the rows balanced    
    GreedySol = [list_avail_words.pop(0)]
    RowSum = (((GreedySol[0] & (1 << np.arange(N)))) > 0).astype(int)
    while len( GreedySol) < SizeSol:
        bestScoreTillNow = RowSum.max()-RowSum.min() + 2 #bound of the worst possible score
        for i, word in enumerate(list_avail_words):
            candidateRowSum = RowSum + (((word & (1 << np.arange(N)))) > 0).astype(int)
            score = candidateRowSum.max()-candidateRowSum.min()
            if score <bestScoreTillNow:
                bestScoreTillNow = score
                bestIndTillNow = i    
            elif score == bestScoreTillNow :
                #then the codeword prefered is the one with fewer pairs of minimum hamming distance
                N_pairs_minHamDist_old, N_pairs_minHamDist_new = 0, 0
                for word_already_in in GreedySol:
                    if gmpy2.hamdist(word, word_already_in) == D:
                        N_pairs_minHamDist_new += 1
                    if gmpy2.hamdist(list_avail_words[bestIndTillNow], word_already_in) == D:
                        N_pairs_minHamDist_old += 1                        
                if N_pairs_minHamDist_old > N_pairs_minHamDist_new:
                    bestIndTillNow = i
        bestWord = list_avail_words.pop(bestIndTillNow)   
        RowSum = RowSum + (((bestWord & (1 << np.arange(N)))) > 0).astype(int)   
        GreedySol.append(bestWord)
    return GreedySol


def ReturnBestSolution(legit_words):
    global N, C, MaxIter 
    MaxComb = 10000    
    bestSol = None
    BestMaxClasses = C
    BestMinClasses = 0
    #Check here exactly for how many classes each subNN works for
    NumberComb = nCr(len(legit_words),C)
    #print('Solution within {0} available. So combinations {1:e}'.format(len(legit_words),NumberComb))
    if NumberComb < MaxComb:
        iterator_for_words = combinations(legit_words, C)
    else:
        iterator_for_words = range(MaxIter)
    for iii in tqdm(iterator_for_words):
        if isinstance(iii,int):
            #When: NumberComb >= MaxComb
            words = FindOneGreedySolution(legit_words, C)
        else:
            #When: NumberComb < MaxComb
            words = iii
        IntMat = np.array([w for w in words ])
        BinaryMat = (((IntMat[:,None] & (1 << np.arange(N)))) > 0).astype(int)
        ClassesPerSubNN = np.sum(BinaryMat, axis=0)
        MaxClasses = ClassesPerSubNN.max()
        MinClasses = ClassesPerSubNN.min()

        if BestMaxClasses-BestMinClasses > MaxClasses-MinClasses:
            BestMinClasses = MinClasses
            BestMaxClasses = MaxClasses
            bestSol = words
            print("New best Min and max column sum: ", MinClasses, MaxClasses)

        if MaxClasses == MinClasses:
            #best Solution supposedly found
            break
        print("Min and max column sum found in this iteration", MinClasses, MaxClasses)
                    
    IntMat = np.array([w for w in bestSol ])
    BinaryMat = (((IntMat[:,None] & (1 << np.arange(N)))) > 0).astype(int)
    print('new solution with min row sum ', BestMinClasses, 'and max ', BestMaxClasses)
    #print(BinaryMat)
    print([format(w, f"0{N}b") for w in bestSol])
    check_solution(bestSol)



N = 10  # Number of SubNNs
Nact = 3 # Number of active SubNNs per Class
C = 100 #Number of classes (that many codewords we need)
D = 4  # with minimum distance D (it must D>=4 since by default keeping Nact constant
       # gives minimum distance D=2 and D=3 is impossible...). In general it has to be
       # an even number
BigEnoughSubset = 10*C #If that many legit_words have been found then it is assumed that they are enough to find 
    #a good solution. Of course it must BigEnoughSet>=C.
BigEnoughSet = 2*BigEnoughSubset # How big to be the set from which to choose the subset. 
MaxIter = 500 #That many times it will try in a greedy way to find a good coding scheme
Way = 2 #for the ways to do it (changes the order). I think way 1 better when Nact>N/2. (maybe way 2 is better when Nact<N/2).

if __name__ == '__main__':
    
    # store number of blocklength N with Nact ones
    Num_possible_words = nCr(N,Nact)
    if Num_possible_words>1e9:
        sys.exit('Too many combinations to consider')
    legit_words = [] #it will contain the set of codewords in which each pair has more than D distance
                    #and from which the coding scheme the algorithm will try to find
    
     
    portion_checked = 0.0
    counter = 0
    if Way == 1:
        #1st way to do it. We first check the codewords which when converted to integer their value is the smaller.
        MaxValidCodeword = 0#this is the biggest binary number with N bits and Nact of them being 1
        for i in range(N-Nact,N): MaxValidCodeword += 2**i
        w = 0 
        for i in range(Nact): w += 2**i        
        while w<=MaxValidCodeword:
            counter += 1
            if counter/Num_possible_words > portion_checked+0.02:
                portion_checked = counter / Num_possible_words
                print("Already included ", len(legit_words), " and checked ", portion_checked*100, "%")
                
            if len(legit_words)>BigEnoughSet: break
            Include = True
            for w_prev_included in legit_words:
                if gmpy2.hamdist(w, w_prev_included) < D:
                    Include = False
                    break
            if Include: 
                legit_words.append(w)
            w = snoob(w)
    elif Way == 2:
        #2nd way to do it. It starts by putting all "Nact" ones in the right-most positions and 
        #it moves the left-most one by  one position left-wise.
        PowerOf2 = [2**i for i in range(N)]
        for Components in combinations(PowerOf2,Nact):
            counter += 1
            if counter/Num_possible_words > portion_checked+0.02:
                portion_checked = counter / Num_possible_words
                print("Already included ", len(legit_words), " and checked ", portion_checked*100, "%")
            if len(legit_words)>BigEnoughSet: break
            w = sum(Components)
            Include = True
            for w_prev_included in legit_words:
                if gmpy2.hamdist(w, w_prev_included) < D:
                    Include = False
                    break
            if Include: 
                legit_words.append(w)
    '''
    if len(legit_words)>C: 
        with open("legit_words_N"+str(N)+"_Nact"+str(Nact)+"_D"+str(D)+".txt", "wb") as fp:
            pickle.dump(legit_words, fp)
   
    with open("legit_words_N"+str(N)+"_Nact"+str(Nact)+"_D"+str(D)+".txt", "rb") as fp:
        legit_words = pickle.load(fp)
    '''
    
    print('Found ', len(legit_words),' possible codewords to choose from')
    if len(legit_words)<C: 
        print('no good solution found. Decrease minimum Hamming distance')
    elif len(legit_words)>=BigEnoughSubset:
        ReturnBestSolution(random.sample(legit_words,BigEnoughSubset))
    else: 
        ReturnBestSolution(legit_words)
    
