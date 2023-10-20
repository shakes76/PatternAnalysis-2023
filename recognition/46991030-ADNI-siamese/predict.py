"""
predict.py: Sample some images from the test set, compare their similarity
using the Siamese Network and predict their class using the Classifier.
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import constants
import dataset
import modules

# Load the models
snn = tf.keras.models.load_model(
    constants.SIAMESE_MODEL_PATH,
    custom_objects={"loss": modules.loss, "DistanceLayer": modules.DistanceLayer},
)
classifier = tf.keras.models.load_model(
    constants.CLASSIFIER_MODEL_PATH,
)

# Sample some images from the test set

AD, NC, AD_processed, NC_processed = dataset.load_samples(
    f"{constants.DATASET_PATH}/train", 20
)

"""
AD_shuffled = AD_processed.copy()
np.random.shuffle(AD_shuffled)

NC_shuffled = NC_processed.copy()
np.random.shuffle(NC_shuffled)
"""

# Compare their similarity using the Siamese Network
AD_both_encoded = snn([AD_processed, np.flip(AD_processed)])
NC_both_encoded = snn([NC_processed, np.flip(NC_processed)])
AD_mixed_encoded = snn([AD_processed, NC_processed])
NC_mixed_encoded = snn([np.flip(NC_processed), np.flip(AD_processed)])

# Plot pairs of images and their similarity

fig, axs = plt.subplots(2, 2)
fig.suptitle("AD/NC Similarity")

axs[0, 0].axis("off")
axs[0, 0].imshow(np.concatenate((AD[0], AD[-1]), axis=1), cmap="gray")
axs[0, 0].set_title(f"Predicted: {np.round(AD_both_encoded[0][0])}, Actual: 0")
axs[0, 1].axis("off")
axs[0, 1].imshow(np.concatenate((NC[0], NC[-1]), axis=1), cmap="gray")
axs[0, 1].set_title(f"Predicted: {np.round(NC_both_encoded[0][0])}, Actual: 0")
axs[1, 0].axis("off")
axs[1, 0].imshow(np.concatenate((AD[0], NC[0]), axis=1), cmap="gray")
axs[1, 0].set_title(f"Predicted: {np.round(AD_mixed_encoded[0][0])}, Actual: 1")
axs[1, 1].axis("off")
axs[1, 1].imshow(np.concatenate((NC[-1], AD[-1]), axis=1), cmap="gray")
axs[1, 1].set_title(f"Predicted: {np.round(NC_mixed_encoded[0][0])}, Actual: 1")

plt.show()

# Predict their class using the Classifier

AD_predicted = classifier.predict(AD_processed)
NC_predicted = classifier.predict(NC_processed)

fig, axs = plt.subplots(8, 5)
fig.suptitle("AD/NC Classification")

for i in range(40):
    axs[i // 5, i % 5].axis("off")
    if i < 20:
        axs[i // 5, i % 5].imshow(AD[i], cmap="gray")
        axs[i // 5, i % 5].set_title(
            f"Predicted: {round(AD_predicted[i][0])}, Actual: 0"
        )
    else:
        axs[i // 5, i % 5].imshow(NC[i - 20], cmap="gray")
        axs[i // 5, i % 5].set_title(
            f"Predicted: {round(NC_predicted[i - 20][0])}, Actual: 1"
        )

plt.show()
