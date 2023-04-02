# ZSL-Speech-Recognition

Zero-Shot Learning is the formulation of a machine learning problem when models are trained without examples. This means that one data set is used during model training, and another, previously unknown to the model, is used during testing. 

My generative models (VAE, GAN) create signal characteristics determined by semantic attributes of a certain class.
 
Stack: PyTorch, NumPy, cuDNN library and CUDA technology are used. Information from hidden layers of NVIDIA's pre-trained Jasper acoustic model is used to extract features and generate vector representations of sounds and texts. The Universal Sentense Encoder (USE) model for English is used to vectorize text annotations. ALso Spotify's Annoying library is used to store lists of words and sentences in the form of matrices. 
