# Siamese Network classifier for the [Alzheimer's Disease Neuroimaging Initiative Dataset](https://adni.loni.usc.edu/)

## Dependencies

Recommended library/runtime versions (all code was tested using these versions):

- `tensorflow >= 2.13.0`
- `numpy >= 1.26.1`
- `matplotlib >= 3.8.0`
- `python >= 3.11.5`

## Training and testing the model

### Training

To train the model, run the following command:

```bash
python train.py
```

This will train the model(s) and save the model in the TensorFlow `SavedModel` format in the `models` directory.

### Testing

To test the model(s), run the following command:

```bash
python predict.py
```

This will load the Siamese network and classifier models from the `models` directory and test them on the test dataset.

It will also sample some images from the test set, perform a similarity check using the original Siamese network and plot them along with their predicted labels using the classifier model.
