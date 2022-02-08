In the main.py file there is a dictionary named ARGS where the dataset (CIFAR-10 or CIFAR-100) can be chosen, the architecture
and some other relevant hyperparameters. Please fill in the directory SAVING_PATH_CHECKPOINTS where checkpoints and logs will be saved. 

The file InterpretatabilityFunctions.py contains functions that can be used to generate the data or plots that appear in the paper.

For google Colab we used the following installations:
!pip install timm
!pip uninstall -y torch
!pip install torch==1.8.2+cpu torchvision==0.9.2+cpu -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
!pip install  torchtext==0.9.1 -f https://download.pytorch.org/whl/cu101/torch_stable.html
!pip install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.8-cp37-cp37m-linux_x86_64.whl
!pip install --quiet pytorch-lightning
