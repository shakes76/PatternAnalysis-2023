# Generative model for Alzheimer's Disease brain scans using styleGAN
The purpose of this model is to generate believable fake 2d MRI scans of the brains of Alzheimer's patients.
StyleGAN is a variation of a generative adversarial network (GAN) - which works by pitching competing "generator" and "discriminator" networks against one another. The main difference of styleGAN is how it uses progressive growing to train the generator network stably by increasing the resolution gradually during training (8x8, 16x16, 32x32, ect). This model uses a preprocessed version of the ADNI dataset for Alzheimer's disease. This model could be used for training medical professionals to recognise both healthy and unhealthy brains.
![StyleGAN generator architecture](https://machinelearningmastery.com/wp-content/uploads/2019/06/Summary-of-the-StyleGAN-Generator-Model-Architecture.png)

## Dependencies
- Pytorch (https://pytorch.org/)
- Numpy (https://numpy.org/)
- MatPlotLib (https://matplotlib.org/)

## Example usage
To train the generator, simply run train.py until completion.
Once the model has been trained, to show results of the completed generator, run predict.py, remembering to change MODEL_PATH to the path of the latest generator.pt file.

## Example output
Below are some generated outputs of the trained model.

![](https://github.com/Ashom2/CSS3710_lab_report/blob/topic-recognition/recognition/images/img_0.png)
![](https://github.com/Ashom2/CSS3710_lab_report/blob/topic-recognition/recognition/images/img_2.png)
![](https://github.com/Ashom2/CSS3710_lab_report/blob/topic-recognition/recognition/images/img_4.png)
![](https://github.com/Ashom2/CSS3710_lab_report/blob/topic-recognition/recognition/images/img_6.png)

## Training
#![](https://github.com/Ashom2/CSS3710_lab_report/blob/topic-recognition/recognition/images/losses.PNG)

## References
- StyleGAN paper https://github.com/NVlabs/stylegan
- ADNI Brain dataset https://adni.loni.usc.edu/

Much of the code was based off the styleGAN implementation found at https://blog.paperspace.com/implementation-stylegan-from-scratch/.