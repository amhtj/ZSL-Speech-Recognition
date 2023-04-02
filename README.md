# ZSL-Speech-Recognition

Zero-Shot Learning is the formulation of a machine learning problem when models are trained without examples. This means that one data set is used during model training, and another, previously unknown to the model, is used during testing. 

My generative models (VAE, GAN) create signal characteristics determined by semantic attributes of a certain class.

As an example of the results, the confusion matrix for the GAN model, tested on the Google Speech Command dataset. Basically, words like "left", "down", "stop" were incorrectly recognized. The percentage of errors on WER for GAN on the test showed 26%, SEP - 46%: 

<img src="/confusion_matrix_GAN.png" alt="Alt text">

The percentage of errors for VAE on the LibriSpeech dataset was 21.89% WER and 53.5 SER on the test. An example of the result of the work of VAE:

<img src="/result_LS_VAE.png" alt="Alt text">

The results are very positive, since the testing was conducted on classes that did not participate in the learning process. The models had only their semantic attributes.
 
Libraries and technologies: 
PyTorch, NumPy, cuDNN library and CUDA technology are used. Information from hidden layers of NVIDIA's pre-trained Jasper acoustic model is used to extract features and generate vector representations of sounds and texts. 
The Universal Sentense Encoder (USE) model for English is used to vectorize text annotations. Also Spotify's Annoying library is used to store lists of words and sentences in the form of matrices. 

Datasets:
Google Speech Command Dataset: https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html
LibriSpeech: https://www.openslr.org/12
