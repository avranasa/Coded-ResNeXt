In all our experiments we used google's COLAB notebook with a TPU accelarator.  We also used the High-RAM option but (at the moment) is available only for those with Pro or Pro+ account. The file Coded_ResNeXt_Imagenet.ipynb contains the notebook showing how to train a Coded-ResNeXt 50 for imagenet and also how to use it for extracting binary classifier or run the experiment of removing subNNs. Click  on the notebook and  on the upper left corner there is the option of clicking the box [Open in Colab] which will open a COLAB notebook where it is possible to train the models. In each cell there are comments explaining the procedure and the arguments.


NOTE: \
  -It is necessary to provide a path (preferably in a google drive) where the training/validation samples of imagenet are saved. \
  -An epoch takes around 45 minutes.  COLAB Pro account allows for 12 hours retaining a session and Pro+ for 24 hours. So training for 150 epochs, needs to save a checkpoint per epoch in a permanent location (google drive again is the solution) and after the session is terminated to restart it and continue the training from the saved checkpoint.
  
 The training script is based on  https://github.com/rwightman/pytorch-image-models/tree/bits_and_tpu .
