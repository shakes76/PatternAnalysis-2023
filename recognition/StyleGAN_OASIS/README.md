# StyleGAN for the OASIS Brain data set

Pattern recognition using a PyTorch style generative adverserial network (GAN) that is part of the COMP3710 repository. This particular implementation looks to create a generative model of the OASIS brain dataset using a variant of StyleGAN that has a "reasonably clear image".

## Motivation of sythnesis with StyleGAN

A StyleGAN is used for synthesis. By learning the probability distribution to some underlying data (images in this case), it is possible to synthesis new images following the learnt distribution. There are a number of motivating reasons why a StyleGAN would be used:
- **Image synthesis** As mentioned, images can be generated from the learnt underlying distribution for artistic and entertainment purposes
- **Augmentation** When limited data are available for training, data augmentation techniques expand the available training data through image variations such as rotation, shear, and colour perturbations. StyleGAN permits a richer form of augmentation by generating brand new variations rather than limited transforms of existing data
- **Content creation** Synthesis can be used to generate new content for media, such as gaming, social media, or even movies (background generation for example)
- **Annonymity** As a result of the synthesis, new samples produced are different from the training data. This permits datasets to be generated that hide the identity of the training data, such as annonymity through face synthesis

## StyleGAN Algorithm

_Description and explanation of the working principles of the algorithm implemented_

## How It Works

_Describe how my particlar implementation works_

_Provide example inputs, outputs and plots of your algorithm_

## Dependencies and Reproducibility of Results

_Referenced libraries and their versions_

_Results of multiple runs, including run times_
