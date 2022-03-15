import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Conv2D, BatchNormalization, ReLU, Concatenate
from tensorflow.keras.layers import MaxPool2D, Add, Dropout
from tensorflow.keras import Model


def conv_block(inputs: tf.Tensor, filters: int, kernel_size: int, strides) -> tf.Tensor:
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def conv_block_no_active(inputs: tf.Tensor, filters: int, kernel_size: int, strides) -> tf.Tensor:
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(inputs)
    x = BatchNormalization()(x)
    return x


def residual_channel_map_block_bottle_neck(inputs: tf.Tensor, filters_1: int, filters_2: int, strides) -> tf.Tensor:
    x = conv_block(inputs=inputs, filters=filters_1, kernel_size=1, strides=strides)
    x = conv_block(inputs=x, filters=filters_1, kernel_size=3, strides=(1, 1))
    x = conv_block_no_active(inputs=x, filters=filters_2, kernel_size=1, strides=(1, 1))
    identify_mapping = conv_block_no_active(inputs=inputs, filters=filters_2, kernel_size=1, strides=strides)
    x = Add()([x, identify_mapping])
    x = ReLU()(x)
    return x


def residual_block_bottle_neck(inputs: tf.Tensor, filters_1: int, filters_2: int, strides) -> tf.Tensor:
    x = conv_block(inputs=inputs, filters=filters_1, kernel_size=1, strides=strides)
    x = conv_block(inputs=x, filters=filters_1, kernel_size=3, strides=(1, 1))
    x = conv_block_no_active(inputs=x, filters=filters_2, kernel_size=1, strides=(1, 1))
    x = Add()([x, inputs])
    x = ReLU()(x)
    return x


def stage(inputs: tf.Tensor, filters_1: int, filters_2: int, strides, conv_block_num: int) -> tf.Tensor:
    x = residual_channel_map_block_bottle_neck(inputs=inputs, filters_1=filters_1, filters_2=filters_2, strides=strides)
    for conv_block_No in range(conv_block_num - 1):
        x = residual_block_bottle_neck(inputs=x, filters_1=filters_1, filters_2=filters_2, strides=(1, 1))
    return x


def resnet101(class_num: int, channel: int):
    Inputs = Input(shape=(224, 224, channel))
    x = Conv2D(64, 7, strides=(2, 2), padding='same')(Inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = stage(x, filters_1=64, filters_2=256, strides=(1, 1), conv_block_num=3)
    x = stage(x, filters_1=128, filters_2=512, strides=(2, 2), conv_block_num=4)
    x = stage(x, filters_1=256, filters_2=1024, strides=(2, 2), conv_block_num=23)
    x = stage(x, filters_1=512, filters_2=2048, strides=(2, 2), conv_block_num=3)
    x = GlobalAveragePooling2D()(x)
    Outputs = Dense(units=class_num, activation='softmax')(x)
    model = Model(inputs=Inputs, outputs=Outputs)
    return model


def resnet50(class_num: int, channel: int):
    Inputs = Input(shape=(224, 224, channel))
    x = Conv2D(64, 7, strides=(2, 2), padding='same')(Inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = stage(x, filters_1=64, filters_2=256, strides=(1, 1), conv_block_num=3)
    x = stage(x, filters_1=128, filters_2=512, strides=(2, 2), conv_block_num=4)
    x = stage(x, filters_1=256, filters_2=1024, strides=(2, 2), conv_block_num=6)
    x = stage(x, filters_1=512, filters_2=2048, strides=(2, 2), conv_block_num=3)
    x = GlobalAveragePooling2D()(x)
    Outputs = Dense(units=class_num, activation='softmax')(x)
    model = Model(inputs=Inputs, outputs=Outputs)
    return model


def resnet101_3_3(class_num: int, channel: int):
    inputs = Input(shape=(224, 224, 6))
    x = Conv2D(64, 7, strides=(2, 2), padding='same')(inputs[..., :3])
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = stage(x, filters_1=64, filters_2=256, strides=(1, 1), conv_block_num=3)
    x = stage(x, filters_1=128, filters_2=512, strides=(2, 2), conv_block_num=4)
    x = stage(x, filters_1=256, filters_2=1024, strides=(2, 2), conv_block_num=23)
    x = stage(x, filters_1=512, filters_2=2048, strides=(2, 2), conv_block_num=3)
    gx1 = GlobalAveragePooling2D()(x)

    x = Conv2D(64, 7, strides=(2, 2), padding='same')(inputs[..., 3:])
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = stage(x, filters_1=64, filters_2=256, strides=(1, 1), conv_block_num=3)
    x = stage(x, filters_1=128, filters_2=512, strides=(2, 2), conv_block_num=4)
    x = stage(x, filters_1=256, filters_2=1024, strides=(2, 2), conv_block_num=23)
    x = stage(x, filters_1=512, filters_2=2048, strides=(2, 2), conv_block_num=3)
    gx2 = GlobalAveragePooling2D()(x)

    x = Concatenate()([gx1, gx2])
    # x = Dropout(rate=0.5)(x)
    outputs = Dense(units=class_num, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model
# tf.keras.utils.plot_model(model, to_file='ResNet_DIY.png', show_shapes=True, show_dtype=True, show_layer_names=True,
#                           rankdir="TB", expand_nested=False, dpi=96, )  # 儲存模型圖
