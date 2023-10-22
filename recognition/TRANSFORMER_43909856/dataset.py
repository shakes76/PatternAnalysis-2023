"""
Contains the data loader for loading and preprocessing the data.
"""

"""
ADNI dataset subfolders:
Have a look at the docs for torchvision.datasets.ImageFolder - should be able to
handle subfolders

Ignoring class information:
I never use the labels in my network so I don't see the point in generating them. 
Maybe I'll just change my training loop to ignore all labels.
for image, _ in dataloader:
    # don't do anything with _
    something(image)


Yes, so during training, you should training based on patient level split. 
These 2d imaging are getting from the 3d scans since the hardware will be 
limited to fit in 3d scans using Transformer. You don't need to change model 
much, but you can just put all the same patients ID within either training or 
validation set. Only need to change how you load the data.
"""
