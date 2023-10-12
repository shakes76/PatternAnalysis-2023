# Detection of Alzheimer's Disease with a Vision Transformer on ADNI dataset

## Background
Transformers have typically seen success in natural language processing tasks, however, with the research performed in the paper "An Image is worth 16x16 words", it can be seen that transformers also have the potential to be used in computer vision and image processing tasks.

Transformers are deep learning models which break the input data into "tokens" which in the case of text are small chunks of characters which are frequently seen together, but in the case of images, are small "patches" of pixels which are positioned close together. These tokens are fed into the encoder layer of the transformer which extracts the relationships between the tokens, and the decoder layer of the transformer which generates the output.

Transformers form a more generalised model compared to traditional CNNs, since the usage of patch embeddings allows the model to __**learn**__ the relationships (or attention) between the tokens, without introducing biases (In the case of CNNs, this involves spatial biases from the kernel - which groups pixels within the kernel's range as "areas of interest"). This allows the model to be applied to a wider range of tasks, such as image classification, object detection, and image segmentation.

Transformers typically require large datasets to be able to overcome the lack of biases.


