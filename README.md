# Visual Transformer 
#### Rhys Tyne s46481894

#### Task explanation
Problem 6: Classify Alzheimer's disease (normal or Alzheimer's Diseased) of the ADNI brain dataset using a visual transformer. 

#### Dependencies
in dependencies.txt

#### Description of Model
The model used is a visual transformer or ViT which is...

- training set is 21520 images
- testing set is 9000 images

the validation set is a subset of the training dataset, separated using the sklearn.model_selection.train_test_split() function.


#### Results
For my first attempt at training I used 10% of the training data as the validation set, along with the Adam optimiser with 
a learning rate of 0.001 and a weight decay of 0.0001. The loss function used was SparseCategoricalCrossentropy and the 
metric used was SparseCategoricalAccuracy. Finally, a ReduceLRonPlateau with patience of 3 and a factor of 0.1 was used.
The results from training can be seen in the repository (named accuracy1 and loss1) (note test in the legend actually refers to validation accuracy)
, and the final test accuracy was 0.655 which is below the goal of 0.8. It is also clear that the model plateaus in both
accuracy and loss after about 60 epochs and after this the learning rate decreased quickly due to the reduction factor 
and low patience time. 

To try and improve accuracy I decided to shuffle the validation set and increase the split to 0.15 as well as increase 
the patience and factor for the learning rate reduction and decreased the training epochs to 80 from 100. this made the testing 
accuracy go to 0.6499 (plots called accuracy2 and loss2). The plateauing occurred much later in the training as well, so I
am to increase the epochs back to 100 as well as change the optimiser to AdamW next what happens to the accuracy.




# TODO list
- add more comments
