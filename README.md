# Coded-ResNeXt
Code used in the  Coded ResNeXt paper (Coded_ResNeXt.pdf). There are two folders, CIFAR and Imagenet, containing our code for the corresponding datasets. 
To train our models we used the accelarators of Google COLAB. Therefore we include also in each folder a jupiter notebook which gives the possibility to be open by Google COLAB. More information for each dataset on the README.md file of each folder.


The python file FastGenerate_CodingScheme.py is the code used to produce the binary coding scheme. If the number of subNNs is N=10 and the binary codewords have length N=10 it is ok to use an exhaustive search for retrieving a good coding scheme but for N=20 (for CIFAR-100) it is already impossible to search exhaustively. The python file implements the algorithm presented in the appendix to find in a fast way good coding schemes.
