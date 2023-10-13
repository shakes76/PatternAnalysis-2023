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

2): Transformer Encoder: The structure of the encoder is on the right hand side of the image above.


