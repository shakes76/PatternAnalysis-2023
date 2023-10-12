import dataset
import modules

(
    train_X1,
    train_X2,
    train_y,
    validate_X1,
    validate_X2,
    validate_y,
    test_X1,
    test_X2,
    test_y,
    all_train_X,
    all_train_y,
    all_validate_X,
    all_validate_y,
    all_test_X,
    all_test_y,
) = dataset.load_dataset("/home/groups/comp3710/ADNI/AD_NC")

print("Creating SNN model")
model = modules.snn()

print("Training SNN model")
model.fit(
    [train_X1, train_X2],
    train_y,
    epochs=20,
    validation_data=([validate_X1, validate_X2], validate_y),
    verbose=1,
    batch_size=32,
)

print("Testing SNN model")
model.evaluate([test_X1, test_X2], test_y, verbose=1)

print("Saving SNN model")
model.save("snn.h5")

twin = model.get_layer("sequential")
twin.trainable = False

print("Creating classifier model")
classifier = modules.snn_classifier(twin)

classifier.fit(
    all_train_X,
    all_train_y,
    epochs=100,
    validation_data=(all_validate_X, all_validate_y),
    verbose=1,
    batch_size=32,
)

print("Evaluating classifier model")
classifier.evaluate(all_test_X, all_test_y, verbose=1)

classifier.save("classifier.h5")
