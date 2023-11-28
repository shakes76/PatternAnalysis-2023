def predict(load_path, testData):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=LEARN_RATE,
        weight_decay=WEIGHT_DECAY
    )

    model = tf.keras.models.load_model(
        load_path,
        custom_objects={
            'PatchLayer': PatchLayer,
            'Embed_Patch': Embed_Patch,
            'MultiheadattentionLSA': Multi_Head_AttentionLSA,
            'AdamW': optimizer
        }
    )

    model.evaluate(testData)

    # Plot confusion matrix
    y_true = []
    y_pred = []

    for image_batch, label_batch in testData:
        y_true.append(label_batch)
        y_pred.append((model.predict(image_batch, verbose=0) > 0.5).astype('int32'))

    labels_true = tf.concat([tf.cast(item[0], tf.int32) for item in y_true], axis=0)
    labels_pred = tf.concat([item[0] for item in y_pred], axis=0)

    matrix = tf.math.confusion_matrix(labels_true, labels_pred, 2).numpy()

    fig, ax = plt.subplots(figsize=(8,8))
    ax.matshow(matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(x=j, y=i, s=matrix[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actual Label', fontsize=18)
    plt.suptitle('Confusion Matrix', fontsize=18)
    plt.savefig('confusion_matrix')
    plt.clf()


if __name__ == '__main__':
    train, val, test = loadData()
    predict(MODEL_SAVE_DEST, test)
