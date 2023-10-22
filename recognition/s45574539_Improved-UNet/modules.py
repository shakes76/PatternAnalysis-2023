import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Concatenate, UpSampling2D, Input, LeakyReLU, Add, Dropout, BatchNormalization
from tensorflow.keras.models import Model


def improved_unet():
    inputs = tf.keras.layers.Input((256, 256, 3))

    # ---- DOWN 1 (16 filter) ----
    # 3x3x3 conv
    c1 = Conv2D(16, (3, 3), strides=1, padding='same', activation=LeakyReLU(alpha=0.01))(inputs)

    # Context
    ctxt1 = BatchNormalization()(c1)
    ctxt1 = Conv2D(16, (3, 3), strides=1, padding='same', activation=LeakyReLU(alpha=0.01))(ctxt1)
    ctxt1 = Dropout(0.3)(ctxt1)
    ctxt1 = BatchNormalization()(ctxt1)
    ctxt1 = Conv2D(16, (3, 3), strides=1, padding='same', activation=LeakyReLU(alpha=0.01))(ctxt1)

    # Element-wise sum
    add1 = Add()([c1, ctxt1])

    # ---- DOWN 2 (32 filter) ----
    # 3x3x3 conv with stride 2
    c2 = Conv2D(32, (3, 3), strides=2, padding='same', activation=LeakyReLU(alpha=0.01))(add1)

    # Context
    ctxt2 = BatchNormalization()(c2)
    ctxt2 = Conv2D(32, (3, 3), strides=1, padding='same', activation=LeakyReLU(alpha=0.01))(ctxt2)
    ctxt2 = Dropout(0.3)(ctxt2)
    ctxt2 = BatchNormalization()(ctxt2)
    ctxt2 = Conv2D(32, (3, 3), strides=1, padding='same', activation=LeakyReLU(alpha=0.01))(ctxt2)

    # Element-wise sum
    add2 = Add()([c2, ctxt2])

    # ---- DOWN 3 (64 filter) ----
    # 3x3x3 conv with stride 2
    c3 = Conv2D(64, (3, 3), strides=2, padding='same', activation=LeakyReLU(alpha=0.01))(add2)

    # Context
    ctxt3 = BatchNormalization()(c3)
    ctxt3 = Conv2D(64, (3, 3), strides=1, padding='same', activation=LeakyReLU(alpha=0.01))(ctxt3)
    ctxt3 = Dropout(0.3)(ctxt3)
    ctxt3 = BatchNormalization()(ctxt3)
    ctxt3 = Conv2D(64, (3, 3), strides=1, padding='same', activation=LeakyReLU(alpha=0.01))(ctxt3)

    # Element-wise sum
    add3 = Add()([c3, ctxt3])

    # ---- DOWN 4 (128 filter) ----
    # 3x3x3 conv with stride 2
    c4 = Conv2D(128, (3, 3), strides=2, padding='same', activation=LeakyReLU(alpha=0.01))(add3)

    # Context
    ctxt4 = BatchNormalization()(c4)
    ctxt4 = Conv2D(128, (3, 3), strides=1, padding='same', activation=LeakyReLU(alpha=0.01))(ctxt4)
    ctxt4 = Dropout(0.3)(ctxt4)
    ctxt4 = BatchNormalization()(ctxt4)
    ctxt4 = Conv2D(128, (3, 3), strides=1, padding='same', activation=LeakyReLU(alpha=0.01))(ctxt4)

    # Element-wise sum
    add4 = Add()([c4, ctxt4])

    # ---- DOWN 5 (256 filter) ----
    # 3x3x3 conv with stride 2
    c5 = Conv2D(256, (3, 3), strides=2, padding='same', activation=LeakyReLU(alpha=0.01))(add4)

    # Context
    ctxt5 = BatchNormalization()(c5)
    ctxt5 = Conv2D(256, (3, 3), strides=1, padding='same', activation=LeakyReLU(alpha=0.01))(ctxt5)
    ctxt5 = Dropout(0.3)(ctxt5)
    ctxt5 = BatchNormalization()(ctxt5)
    ctxt5 = Conv2D(256, (3, 3), strides=1, padding='same', activation=LeakyReLU(alpha=0.01))(ctxt5)

    # Element-wise sum
    add5 = Add()([c5, ctxt5])

    # ---- UP 1 (128 filter) ----
    # Upsampling
    u1 = UpSampling2D((2, 2))(add5)
    u1 = Conv2D(128, (3, 3), padding='same', activation=LeakyReLU(alpha=0.01))(u1)

    # Concatenate
    concat1 = Concatenate()([u1, add4])

    # ---- UP 2 (128 and 64 filter) ----
    # Localization
    u2 = Conv2D(128, (3, 3), padding='same', activation=LeakyReLU(alpha=0.01))(concat1)
    u2 = Conv2D(128, (1, 1), padding='same', activation=LeakyReLU(alpha=0.01))(u2)

    # Upsampling
    u2 = UpSampling2D((2, 2))(u2)
    u2 = Conv2D(64, (3, 3), padding='same', activation=LeakyReLU(alpha=0.01))(u2)

    # Concatenate
    concat2 = Concatenate()([u2, add3])

    # ---- UP 3 (64 and 32 filter) ----
    # Localization
    u3 = Conv2D(64, (3, 3), padding='same', activation=LeakyReLU(alpha=0.01))(concat2)
    u3 = Conv2D(64, (1, 1), padding='same', activation=LeakyReLU(alpha=0.01))(u3)

    # Segmentation
    seg1 = Conv2D(3, (1, 1), strides=1, padding="same")(u3)
    seg1 = UpSampling2D((2, 2))(seg1)

    # Upsampling
    u3 = UpSampling2D((2, 2))(u3)
    u3 = Conv2D(32, (3, 3), padding='same', activation=LeakyReLU(alpha=0.01))(u3)

    # Concatenate
    concat3 = Concatenate()([u3, add2])

    # ---- UP 4 (32 and 16 filter) ----
    # Localization
    u4 = Conv2D(32, (3, 3), padding='same', activation=LeakyReLU(alpha=0.01))(concat3)
    u4 = Conv2D(32, (1, 1), padding='same', activation=LeakyReLU(alpha=0.01))(u4)

    # Segmentation
    seg2 = Conv2D(3, (1, 1), strides=1, padding="same")(u4)
    seg2 = Add()([seg1, seg2])
    seg2 = UpSampling2D((2, 2))(seg2)

    # Upsampling
    u4 = UpSampling2D((2, 2))(u4)
    u4 = Conv2D(16, (3, 3), padding='same', activation=LeakyReLU(alpha=0.01))(u4)

    # Concatenate
    concat4 = Concatenate()([u4, add1])

    # ---- UP 5 (32 filter) ----
    # 3x3x3 conv with stride 2
    u5 = Conv2D(32, (3, 3), strides=1, padding='same')(concat4)

    # Segmentation
    seg3 = Conv2D(3, (1, 1), strides=1, padding="same")(u5)
    seg3 = Add()([seg2, seg3])

    outputs = Conv2D(3, (1, 1), activation='softmax')(seg3)
    model = Model(inputs, outputs)

    return model
