# Using a Vision Transformer to Classify Alzheimer's Disease

## ADNI Brain MRI Dataset
The ADNI Brain dataset consists of a training and testing folder, with each subfolder AD and NC conatining a collection of patients that are either positive (AD/Alzheimers group) or negative (NC/Healthy group).

![Alt text](images/data_samples.png)

## Vision Transformers
Transformer models are models which uses an 'attention mechanism' to 'remember' previous parts of a sentence, to be able to create complex and coherent sentences and paragraphs. Vision transformers take the same concept but for images instead. This is done by creating embedding vectors of sub-parts of an image, known as a patch. The model then uses an attention mechanism and positional encodings to know where in the image each part belongs.

![Alt text](images/vit_diagram.png)
