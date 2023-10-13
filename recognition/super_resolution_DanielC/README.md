Pytorch implmenetation of a brain MRI super-resolution network 

The padding for convolution network was calculated through: input size - kernel size + 2* padding size / 1 (stride is always 1)
(source https://stats.stackexchange.com/questions/297678/how-to-calculate-optimal-zero-padding-for-convolutional-neural-networks)

Changing the input and output channel size of conv2d layers in the model can help to decerase the loss.
However, after a certain point the difference is negligible as shown.

Channel size | Loss 
--- | --- 
32 | 0.00253
64 | 0.00249
128 | 0.00236