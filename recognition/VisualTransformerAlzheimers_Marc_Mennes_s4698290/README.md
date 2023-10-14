# Problem Description
The goal of this project is to diagnose Alzheimer's from an MRI scan accurately using a visual transformer network. The data used was obtained from the "Alzheimer's Disease Neuroimaging Initiative" or ADNI dataset. Here are some samples:

![Alt text](readmeImages/altzSample.jpeg)![Alt text](readmeImages/normSample.jpeg)
<pre>
Alzheimer's                         Normal Brain
</pre>

# The Visual Transformer

![Alt text](<readmeImages/Screenshot from 2023-10-13 22-47-10.png>)

The structure of the transformer was based off of the paper, "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale". The network has 3 distinct parts:

1): Embeddings: The images are cut up into patches, the patches are flattened, and each patch is fed into a single layer linear network to embed them. The idea here is that the network can learn a linear projection of the patches that is advantagious for classification. After the patches are projected, each embedded patch has its index in the sequence of patches added to it, called a positional encoding, this is to retain contextual information in the image, and was found in the paper to be advantagious for performance. Alongside these linear embeddings with positional encodings, there is an extra learned parameter, a "class embedding" added to position 0. This class embedding is what is used to classify the image after the transformer blocks.

2): Transformer Encoder: The structure of the encoder is on the right hand side of the image above. The "Norm" blocks are layernorm, standard in transformers. The skip connections are to stop the gradient signal from dying off and not effecting nodes at the start of a deep network. The mlp block at the end of the encoder exists so the model can learn to transform the output of the multi head attention into something maybe more useful for the next transformer encoder block. The multi-head (self) attention, however, is really the thing that makes this whole network work.

Multi-head self attention allows the network to find correlations between image patches that help it classify the image. It works by learning linear projections of the image patches for each head to convert them into keys, queries, and values, and then calculating attention on these and then concatenating them back together and then applying a learned linear projection again. This layer learns the weights that allow it to relate key parts of the image with eachother.

The transformer encoder is repeated multiple times in the network structure usually.

3): MLP Head: This plays the same role as it would in a convolutional network, the transformer encoder blocks are essentially learned feature extraction, these dense layers take in the output of the transformer encoder and use it to classify the image as either a positive Alzeimer's case or a negative.

# Training
