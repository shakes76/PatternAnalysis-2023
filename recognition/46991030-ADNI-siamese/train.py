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
) = dataset.load_dataset("/home/groups/comp3710/ADNI/AD_NC")

print("Creating model")
model = modules.snn()

print("Training model")
model.fit(
    [train_X1, train_X2],
    train_y,
    epochs=20,
    validation_data=([validate_X1, validate_X2], validate_y),
    verbose=1,
    batch_size=32,
)

print("Testing model")
model.evaluate([test_X1, test_X2], test_y, verbose=1)

print("Saving model")
model.save("model.h5")
