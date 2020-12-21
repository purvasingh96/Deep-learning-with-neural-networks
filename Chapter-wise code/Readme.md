<p align="center"><img width="40%" src="logo/Pytorch_logo.png" /></p>

--------------------------------------------------------------------------------

# Deep Learning with PyTorch

## Downloading data-sets for Colab notebooks
### Change dir and download zip file
In order to download data-sets, use the following code -
```python
import os
from pathlib import Path
os.chdir(Path('./sample_data'))
print(os.getcwd())
!wget -p <path_to_download_folder> -N <dataset_download_link>
```
For example - 
```python
!wget -p  Path('./sample_data') -N  https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
```
### Unzipping the zip file
```python
from zipfile import ZipFile
import os
print(os.getcwd())
zf = ZipFile('text8.zip', 'r')
zf.extractall('./')
zf.close()
```


## PyTorch Projects

1. [Recap of Numpy and Matrices](./Code%20-%20PyTorch/0.%20Recap%20Numpy%20and%20Matrices)
    * [Quiz on Numpy](./Code%20-%20PyTorch/0.%20Recap%20Numpy%20and%20Matrices/NumPy_Quiz.py)
    * [Scalars, Vectors, Matrices, Tensors](./Code%20-%20PyTorch/0.%20Recap%20Numpy%20and%20Matrices/Scalars,_Vectors,_Matricies_and_Tensors.ipynb)
    
2. [Introduction to PyTorch](./Code%20-%20PyTorch/1.%20Intro%20to%20PyTorch)
    * [Deep Learning with PyTorch - 60 minute blitz](./Code%20-%20PyTorch/1.%20Intro%20to%20PyTorch/01.%20Deep_Learning_with_PyTorch_A_60_Minute_Blitz_.ipynb)
    * [Verify PyTorch Installation](./Code%20-%20PyTorch/1.%20Intro%20to%20PyTorch/01.verify_pytorch_installation.ipynb)
    * [Autograd Automatic Differentiation](./Code%20-%20PyTorch/1.%20Intro%20to%20PyTorch/02.%20Autograd_Automatic_Differentiation.ipynb)
    * [Single Layer Neural Network](./Code%20-%20PyTorch/1.%20Intro%20to%20PyTorch/02.single_layer_neural_network.ipynb)
    * [Neural Networks](./Code%20-%20PyTorch/1.%20Intro%20to%20PyTorch/03.%20Neural_networks.ipynb)
    * [Multi-layer Neural Networks](./Code%20-%20PyTorch/1.%20Intro%20to%20PyTorch/03.mutilayer_neural_network.ipynb)
    * [Implementing Softmax Function](./Code%20-%20PyTorch/1.%20Intro%20to%20PyTorch/04.implementing_softmax.ipynb)
    * [Training an Image Classifier](./Code%20-%20PyTorch/1.%20Intro%20to%20PyTorch/04_Training_an_image_classifier.ipynb)
    * [Implementing ReLU Activation Function via PyTorch](./Code%20-%20PyTorch/1.%20Intro%20to%20PyTorch/05.ReLU_using_pytorch.ipynb)
    * [Playing with TensorBoard](./Code%20-%20PyTorch/1.%20Intro%20to%20PyTorch/05_Playing_with_TensorBoard.ipynb)
    * [Training Neural Network via PyTorch](./Code%20-%20PyTorch/1.%20Intro%20to%20PyTorch/06.training_neural_network_via_pytorch.ipynb)
    * [Validation via PyTorch](./Code%20-%20PyTorch/1.%20Intro%20to%20PyTorch/07.%20Validating_using_pytorch.ipynb)
    * [Regularization via PyTorch](./Code%20-%20PyTorch/1.%20Intro%20to%20PyTorch/08.%20Regularization_using_pytorch.ipynb)
    * [Loading Image Data via PyTorch](./Code%20-%20PyTorch/1.%20Intro%20to%20PyTorch/09.%20loading_image_data_via_pytorch.ipynb)
    * [Transfer Learning via PyTorch](./Code%20-%20PyTorch/1.%20Intro%20to%20PyTorch/10.%20Transfer_learning_via_pytorch.ipynb)
 
3. [Convolutional Neural Networks](./Code%20-%20PyTorch/2.%20Convolution%20Neural%20Networks)
    * [Basics: Load, Train, Test and Validate your Model](./Code%20-%20PyTorch/2.%20Convolution%20Neural%20Networks/1.%20Basics/Load_train_test_and_validate_your_model.ipynb)
    * [CIFAR Image Classification](./Code%20-%20PyTorch/2.%20Convolution%20Neural%20Networks/2.%20Image%20Classification/CIFAR_image_classifier.ipynb)
    * [Object Detection](./Code%20-%20PyTorch/2.%20Convolution%20Neural%20Networks/3.%20Object%20Detection)
        * [Frontal Face Recognition](./Code%20-%20PyTorch/2.%20Convolution%20Neural%20Networks/3.%20Object%20Detection/frontal_face_recognition.ipynb)
        * [Object Detection](./Code%20-%20PyTorch/2.%20Convolution%20Neural%20Networks/3.%20Object%20Detection/Object_Detection.ipynb)
    * [Transfer Learning](./Code%20-%20PyTorch/2.%20Convolution%20Neural%20Networks/4.%20Transfer%20Learning)
        * [Bees Prediction via Transfer Learning](./Code%20-%20PyTorch/2.%20Convolution%20Neural%20Networks/4.%20Transfer%20Learning/Transfer_Learning_predict_bees.ipynb)
        * [Flower Prediction via Transfer Learning](./Code%20-%20PyTorch/2.%20Convolution%20Neural%20Networks/4.%20Transfer%20Learning/Transfer_Learning_predict_flowers.ipynb)
    * [Style Transfer](./Code%20-%20PyTorch/2.%20Convolution%20Neural%20Networks/5.%20Style%20Transfer)
        * [Style Transfer on an Octopus](./Code%20-%20PyTorch/2.%20Convolution%20Neural%20Networks/5.%20Style%20Transfer/style_transfer_on_octopus.ipynb)
        * [Style Transfer on Purva](./Code%20-%20PyTorch/2.%20Convolution%20Neural%20Networks/5.%20Style%20Transfer/style_transfer_on_purva.ipynb)
    * [Data Augmentation](./Code%20-%20PyTorch/2.%20Convolution%20Neural%20Networks/6.%20Data%20augmentation)
    * [Weight Initialization Strategies](./Code%20-%20PyTorch/2.%20Convolution%20Neural%20Networks/7.%20Weight%20Initialization%20Strategies/Weight_initialization.ipynb)
    * [Autoencoders](./Code%20-%20PyTorch/2.%20Convolution%20Neural%20Networks/8.%20Autoencoders)
        * [Linear Autoencoder](./Code%20-%20PyTorch/2.%20Convolution%20Neural%20Networks/8.%20Autoencoders/linear_autoencoder.ipynb)
        * [Convolutional Autoencoder](./Code%20-%20PyTorch/2.%20Convolution%20Neural%20Networks/8.%20Autoencoders/convolution_autoencoder.ipynb)
    * [Dog Breed Classifier](./Code%20-%20PyTorch/2.%20Convolution%20Neural%20Networks/9.%20Dog%20breed%20classifier)
    
 4. [Recurrent Neural Networks](./Code%20-%20PyTorch/3.%20Recurrent%20Neural%20Networks)
    * [Text Generation using RNNs](./Code%20-%20PyTorch/3.%20Recurrent%20Neural%20Networks/1.%20Text%20generation%20using%20RNNs)
        * [Future Anna Karenina Series](./Code%20-%20PyTorch/3.%20Recurrent%20Neural%20Networks/1.%20Text%20generation%20using%20RNNs/future_anna_karenina.ipynb)
        * [Future Harry Potter Series](./Code%20-%20PyTorch/3.%20Recurrent%20Neural%20Networks/1.%20Text%20generation%20using%20RNNs/future_harry_potter_series.ipynb)
    * [Sentiment Analysis](./Code%20-%20PyTorch/3.%20Recurrent%20Neural%20Networks/2.%20Sentiment%20Analysis/sentiment_analysis.ipynb)
    * [Time Series Prediction](./Code%20-%20PyTorch/3.%20Recurrent%20Neural%20Networks/3.%20Time%20Series%20Prediction)
    * [Word2Vec](./Code%20-%20PyTorch/3.%20Recurrent%20Neural%20Networks/4.%20Word2Vec)
    * [Generation of T.V. Scripts via NLG](./Code%20-%20PyTorch/3.%20Recurrent%20Neural%20Networks/5.%20Generate%20TV%20Scripts)
    * [Attention](./Code%20-%20PyTorch/3.%20Recurrent%20Neural%20Networks/6.%20Attention/Readme.md)
 
 5. [Generative Adversarial Networks (GANs)](./Code%20-%20PyTorch/4.%20Generative%20Adversarial%20Networks%20(GANs))
    * [Overview: Theorey](./Code%20-%20PyTorch/4.%20Generative%20Adversarial%20Networks%20(GANs)/Readme.md)
    * [Generate Hand Written Digits using GANs](./Code%20-%20PyTorch/4.%20Generative%20Adversarial%20Networks%20(GANs)/1.%20Generating%20hand-written%20digits%20using%20GANs/Hand_written_digit_generation_via_GANs.ipynb)
    * [Deep Convolutional GANs](./Code%20-%20PyTorch/4.%20Generative%20Adversarial%20Networks%20(GANs)/2.%20Deep%20Convolution%20GANs/Deep_Convolution_GANs.ipynb)
    * [Cyclic GANs](./Code%20-%20PyTorch/4.%20Generative%20Adversarial%20Networks%20(GANs)/3.%20Cyclic%20GANs/Readme.md)
        * [Image-to-Image Translation via Cyclic GANs](./Code%20-%20PyTorch/4.%20Generative%20Adversarial%20Networks%20(GANs)/3.%20Cyclic%20GANs/Image-to-Image%20Translation%20via%20Cyclic%20GANs/Image_to_image_translation_via_Cyclic_GANs.ipynb)
    * [Generating Faces via DCGAN](./Code%20-%20PyTorch/4.%20Generative%20Adversarial%20Networks%20(GANs)/4.%20Generate%20Faces%20via%20DCGAN/dlnd_face_generation.ipynb)
    
 6. [Deploying Sentiment Analysis Model using Amazon Sagemaker](./Code%20-%20PyTorch/5.%20Deploy%20Models%20to%20PROD%20via%20Amazon%20Sagemaker)
    * [Deploy IMDB Sentiment Analysis Model](./Code%20-%20PyTorch/5.%20Deploy%20Models%20to%20PROD%20via%20Amazon%20Sagemaker/1.%20Deploy%20IMDB%20Sentiment%20Analysis%20Model/IMDB%20Sentiment%20Analysis%20-%20XGBoost%20-%20Web%20App.ipynb)
    * [Deploy Your Own Sentiment Analysis Model](./Code%20-%20PyTorch/5.%20Deploy%20Models%20to%20PROD%20via%20Amazon%20Sagemaker/2.%20Deploy%20your%20own%20sentiment%20analysis%20model/SageMaker%20Project.ipynb)
    
 7. [Natural Language Processing](./Code%20-%20PyTorch/6.%20Natural-Language-Processing)
    * [Naive Bayes Classifier](./Code%20-%20PyTorch/6.%20Natural-Language-Processing/1.%20Naive%20Bayes%20Classifier/Readme.md)
        * [Spam Classifier](./Code%20-%20PyTorch/6.%20Natural-Language-Processing/1.%20Naive%20Bayes%20Classifier/spam_classifier/Bayesian_Inference.ipynb)
        * [Sentiment Analysis](./Code%20-%20PyTorch/6.%20Natural-Language-Processing/1.%20Naive%20Bayes%20Classifier/sentiment_analysis/Sentiment%20Analysis.ipynb)
    * [POS Tagging](./Code%20-%20PyTorch/6.%20Natural-Language-Processing/2.%20Parts%20of%20Speech%20Tagging/Readme.md)
        * [POS Tagging via HMM and Viterbi Algorithm](./Code%20-%20PyTorch/6.%20Natural-Language-Processing/2.%20Parts%20of%20Speech%20Tagging/POS%20Tagging%20with%20HMM%20and%20Viterbi.ipynb)
        * [HMMs for POS Tagging](./Code%20-%20PyTorch/6.%20Natural-Language-Processing/2.%20Parts%20of%20Speech%20Tagging/HMM%20Tagger.ipynb)
    * [Feature Extraction and Embeddings](./Code%20-%20PyTorch/6.%20Natural-Language-Processing/3.%20Feature%20Extraction%20&%20Embeddings/Readme.md)
    * [Topic Modelling](./Code%20-%20PyTorch/6.%20Natural-Language-Processing/4.%20Topic%20Modelling/Readme.md)
    * [Latent Dirichlet Allocation](./Code%20-%20PyTorch/6.%20Natural-Language-Processing/4.%20Topic%20Modelling/Latent_dirichlet_allocation.ipynb)
    * [Sentiment Analysis](./Code%20-%20PyTorch/6.%20Natural-Language-Processing/5.%20Sentiment%20Analysis)
        * [BERT for sentiment analysis of Twits (StockTwits)](./Code%20-%20PyTorch/6.%20Natural-Language-Processing/5.%20Sentiment%20Analysis/bert-for-sentiment-analysis-of-stock-twits.ipynb)
        * [EDA and sentiment analysis of COVID-19 tweets](./Code%20-%20PyTorch/6.%20Natural-Language-Processing/5.%20Sentiment%20Analysis/covid19-tweets-eda-and-sentiment-analysis.ipynb)
        * [EDA and sentiment analysis of Joe Biden's tweets](./Code%20-%20PyTorch/6.%20Natural-Language-Processing/5.%20Sentiment%20Analysis/eda-and-sentiment-analysis-of-joe-biden-tweets.ipynb)
    * [Machine Translation](./Code%20-%20PyTorch/6.%20Natural-Language-Processing/6.%20Machine%20Translation/Readme.md)
        * [NMT via basic linear algebra](./Code%20-%20PyTorch/6.%20Natural-Language-Processing/6.%20Machine%20Translation/NMT-Basic/NMT%20-%20Basic.html)
        * [NMT via encoder-decoder architecture](./Code%20-%20PyTorch/6.%20Natural-Language-Processing/6.%20Machine%20Translation/NMT-Advanced%20(Tensorflow%20Implementation)/machine_translation.ipynb)
    * [Speech Recognition](./Code%20-%20PyTorch/6.%20Natural-Language-Processing/7.%20Speech%20Recognition/vui_notebook.ipynb)
    * [Autocorrect Tool via Minimum Edit Distance](./Code%20-%20PyTorch/6.%20Natural-Language-Processing/11.%20Autocorrect%20Tool/Auto_correct_tool.ipynb)
    * [Autocomplete tool using n-gram language model](./Code%20-%20PyTorch/6.%20Natural-Language-Processing/12.%20Autocomplete%20Tool/Auto%20complete%20tool.ipynb)
    *  [Natural Language Generation](./Code%20-%20PyTorch/6.%20Natural-Language-Processing/8.%20Natural%20Language%20Generation)
        * [Text generation via RNNs and (Bi)LSTMs](./Code%20-%20PyTorch/6.%20Natural-Language-Processing/8.%20Natural%20Language%20Generation/text-generation-via-rnn-and-lstms-pytorch.ipynb)
    * [Question Answering Models](./Code%20-%20PyTorch/6.%20Natural-Language-Processing/9.%20Question%20Answering)
        * [BERT for answering queries related to stocks](./Code%20-%20PyTorch/6.%20Natural-Language-Processing/9.%20Question%20Answering/bert-for-answering-queries-related-to-stocks.ipynb)
    * [Text Classification](./Code%20-%20PyTorch/6.%20Natural-Language-Processing/10.%20Text%20Classification/)
        * [Github bug prediction using BERT](./Code%20-%20PyTorch/6.%20Natural-Language-Processing/10.%20Text%20Classification/github-bug-prediction-via-bert.ipynb)
        * [Predicting DJIA movement using BERT](./Code%20-%20PyTorch/6.%20Natural-Language-Processing/10.%20Text%20Classification/predicting-DJIA-movement-with-BERT.ipynb)
        * [SMS spam classifier](./Code%20-%20PyTorch/6.%20Natural-Language-Processing/10.%20Text%20Classification/sms-spam-classifier.ipynb) 
        