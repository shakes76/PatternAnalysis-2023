<h3 align="center">Siamese Neural Network for AD/NC Classification </h3>


---

<p align="center"> Name: <b>Raghavendra Singh Gulia</b><br>UQ StudentID: <b>s4824575</b>
    <br> 
</p>

## 📝 Table of Contents

- [Introduction](#-introduction-)
- [Code Overview](#-code-overview-)
- [Results](#-results-)
- [Dependencies](#-dependencies-)
- [Conclusion](#-conclusion-)
- [References](#-references-)


## 🧐 Introduction <a name = "introduction"></a>


The Vision Transformer is a powerful deep learning architecture for computer vision tasks. In this project, we develop a Vision Transformer model to classify brain MRI images from the ADNI dataset into two classes: Alzheimer's Disease (AD) and Normal Control (NC). The Vision Transformer is trained to distinguish between these two classes using self-attention mechanisms.
The project is divided into three main components: 
1) Dataset preparation: Loading and preprocessing the ADNI brain MRI images, 
2) Model architecture: Defining the Vision Transformer architecture based on the ViT paper [8,9], 
3) Training and Prediction: Training the model on preprocessed data and evaluating performance on the held-out test set with the target of achieving a minimum accuracy of 0.8.
The Vision Transformer takes advantage of self-attention to capture global relationships between image patches, without relying on local connections like convolutional networks. This allows it to efficiently model long-range dependencies in the brain MRI data to distinguish between AD and normal classes. We implement the model in TensorFlow and evaluate its ability to classify Alzheimer's disease from brain images with the goal of supporting early disease detection.

![Vision Transformer { width="800" height="600" style="display: block; margin: 0 auto" }](recognition/s4824575_ADNI/assets/VisionTransformer.png)

## 👨🏻‍💻 Code Overview <a name = "code_overview"></a>

<p> This project aims to develop a machine learning model to classify MRI brain images as either Alzheimer's disease (AD) or normal control (NC) using data from the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset. A Vision Transformer architecture is implemented and trained on preprocessed MRI images to distinguish between the two classes<p>
<ol>

<b><li>Dataset Preparation</li></b>
<p>The ADNI dataset contains structural MRI brain scans from multiple sites for AD patients and healthy controls. In the dataset preparation phase, images are loaded from directories organized by class label. The dataset is then split into training, validation, and test subsets for model development and evaluation. Preprocessing steps like resizing, normalization, and grayscale conversion ensure consistency across images.
</p>
<b><li>Model Architecture</li></b>
<p>A Vision Transformer model is defined using the TensorFlow Hub module API. It consists of an embedding layer to generate patch embeddings from input images, followed by an encoder stack of multi-head self-attention and feedforward blocks to learn contextual relationships. Classification is performed via dense layers at the end.
</p>
<b><li>Training </li></b>
<p>The model is compiled with an optimizer, loss function and metrics. It is trained on the preprocessed training subset for 10 epochs with validation monitoring to prevent overfitting. The model weights yielding the best validation accuracy are saved for inference. Hyperparameters like learning rate, batch size are tuned.

-	<b>Data Loading:</b> The training data is loaded using the load_data function, is created.
-	<b>Device Selection:</b> The code checks for GPU availability and moves the model to the GPU for accelerated training.
-	<b>Training Loop:</b> The model undergoes training for a predetermined number of epochs. During training, the Vision Transformer learns to distinguish between AD and NC images.
-	<b>Model Saving:</b> After training, the model is saved to a file as model.h5 (not pushed to the repo since the file was big), allowing for future use.
-	<b>Loss Plotting:</b> A plot of training loss is generated to visualize the training progress.
</p>
<b><li>Making Predictions</li></b>
<p>The prediction phase is executed using the <b>predict.py</b> script. The main components of this phase include:
raining curves are plotted to visualize the learning process. The trained model is loaded on test data and the final test accuracy is logged. The project outputs a classification model that meets the performance criteria on unseen MRI examples.
</p>
</ol>



<h3>Data Set:</h3>

![Dataset]
<p>The data used to train this model is the ADNI dataset. The dataset consists of a training set, containing ~10000 images of cognitive normal brain scans (or normal controls) and ~10000 images of brain scans of patients with Alzheimer's disease. The test set contains ~4500 of each type. To train the SNN, the train data is split into 80% for training, and 20% for validation. However, the data in its raw form is not suitable for training an SNN, as we need image pairs and labels. Therefore, we must build image pairs when loading the data. This is done in <b>dataset.py</b>, and is well documented in the code. In short, we take all the image paths, create the pairs and labels corresponding to the pair, turn them into tensorflow datasets, and then shuffle the dataset before splitting into 80% train, 20% validation. No test set is generated from the image pairs, as we only evaluate the final classification model. Further given the large amount of data, 20% is a suitable amount of data to validate on.</p>
<h3>Classification Data:</h3>
<p>To evaluate and train the classification model we must build a new dataset. One that just contains singular images, and a corresponding label for whether the image is AD or CN (0 or 1). We again use an 80/20 train validation split, and use the entire test set to evaluate the model after training.</p>
<h3>Model Architecture:</h3>
<h4>SNN:</h4>

![ADNI] (recognition/s4824575_ADNI/assets/ADNI DATASET.png)




## 🐍 Results <a name = "results"></a>
<p>After running the train.py</p>

![Training Graph] (recognition/s4824575_ADNI/assets/training.png)

<p>The image shows a line graph with two lines - a blue line and an orange line. The blue line plots the training loss over multiple epochs of training, while the orange line plots the validation loss.
We can see that both the training loss and validation loss steadily decrease as the number of epochs increases. This indicates that the model is learning from the training data and is able to generalize this learning to new data, as seen by the decreasing validation loss. If the validation loss was increasing or plateauing while the training loss continued to decrease, it would suggest the model is overfitting to the training data and not generalizing well.
The consistent downward trend of both lines shows that the model is able to learn the underlying patterns or relationships in the training data and apply that knowledge to new examples, without simply memorizing the training data. This is a good sign that the model will be able to perform well on unseen data.
The legend at the bottom clearly distinguishes the blue line as representing training loss and the orange line as validation loss. This helps the viewer to easily understand what each line is plotting.
In summary, this graph shows that the model is learning effectively during training as both losses decrease, and that it is generalizing this learning well to new examples outside the training dataset. The consistent downward trends indicate the training process is progressing well for this model.</p>

<p>After Predict.py bit we get</p>

![Predict_graph] (recognition/s4824575_ADNI/assets/predict.png)

<p>- There is a clear downward trend seen in both the blue and orange lines over time (as represented on the x-axis).
- The blue line, representing the training loss validation, gradually decreases in a smooth and consistent manner. This indicates the model's training loss is consistently improving as more data is passed through it.
- The orange line, representing the validation loss, also decreases smoothly. This suggests the model is learning features from the training data that generalize well to new examples, rather than just overfitting the training set.
- Both loss values start off higher then gradually decrease as time progresses. This downward trend demonstrates the model is learning effectively from the data and its predictions/performance are improving steadily over multiple iterations/predictions.
- The consistent and steady downward slope of both lines without any spikes or sudden changes indicates the model is learning stability without fluctuations in its learning process.
- The decreasing gap between the blue and orange lines shows the model's training and validation losses are converging, meaning it is learning features that apply to both its training and validation sets equally well.
In summary, the key trends shown are two smooth, consistent downward slopes representing improving training and validation losses, indicating stable and effective learning by the model over time.</p>

- Epoch cycle running in terminal

![epoch_cycle] (recognition/s4824575_ADNI/assets/Model running.png)



## 🦥 Dependencies <a name = "dependencies"></a>
<ul>
<li><b>Tensorflow(tf)</b> - 2.14.0</li>
<li><b>numpy</b> - 1.22.0</li>
<li><b>sklearn.model_selection(train_test_split)</b> - 1.0.2</li>
<li><b>tensorflow.keras.preprocessing.image(load_img,img_to_array)</b> </li>

## 🙏🏽 Conclusion <a name = "conclusion"></a>
<p>
In this study, we developed a visual transformer model for classifying Alzheimer's disease using brain imaging data from the ADNI dataset. The model achieved a test accuracy of over 80%, meeting the specified performance threshold.
The transformer architecture was able to learn discriminative features directly from the raw image data that distinguished between normal and AD brain scans. This demonstrates the model's ability to capture subtle differences indicative of Alzheimer's pathology without relying on manual feature engineering.
The high test accuracy shows that the model is generalized well and can accurately predict the class of previously unseen brain scans. This suggests it has learned representations that are robust and transferable to new data samples.
Overall, this project shows that visual transformers are a promising approach for automated Alzheimer's diagnosis from medical images. With further optimization, such deep learning models may help clinicians with early detection and monitoring of the disease. The model developed here could potentially serve as a decision support tool to assist in the evaluation and management of patients.</p>

## 🔗 References <a name = "references"></a>

- [Vision Transformer wiki ](https://en.wikipedia.org/wiki/Vision_transformer)

- [Vision_transformers_a_very_basic_introduction ](https://medium.com/data-and-beyond/vision-transformers-vit-a-very-basic-introduction-6cd29a7e56f3)
