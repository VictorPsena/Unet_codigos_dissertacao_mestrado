# Bibliotecas
from keras.layers import (
    Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate,
    BatchNormalization, Activation, add, ELU, LeakyReLU,
    )
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.utils import plot_model

# Camada convolucional 2D

def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1,1), activation= 'relu', name=None):
    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False, name=name)(x)
    x = BatchNormalization(axis=3, scale=False)(x)

    if activation == None:
        return x
    
    x = Activation(activation, name = name)(x)
    return x

# Camada Convolucional transposta 2D
def trans_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(2,2), activation= 'relu', name=None):

    x = Conv2DTranspose(filters, (num_row, num_col), strides=strides, padding=padding, name=name)(x)
    x = BatchNormalization(axis=3, scale=False)(x)

    return x

# Bloco MultiRes

def MultiResBlock(U, inp, alpha = 1.67):
    
    W = alpha * U

    shortcut = inp

    shortcut = conv2d_bn(shortcut, int(W*0.167)+int(W*0.333)+int(W*0.5), 1, 1, activation = None, padding = 'same')

    conv3x3 = conv2d_bn(inp, int(W*0.167),3,3, activation='relu', padding = 'same')

    conv5x5 = conv2d_bn(conv3x3, int(W*0.333),3,3, activation='relu', padding = 'same')

    conv7x7 = conv2d_bn(conv5x5, int(W*0.5),3,3, activation='relu', padding = 'same')

    out = concatenate([conv3x3, conv5x5, conv7x7], axis = 3)
    out = BatchNormalization(axis=3, scale=False)(out)

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    return out

# ResPath

def ResPath(filters, length, inp):

    shortcut = inp
    shortcut = conv2d_bn(shortcut, filters, 1,1, activation=None, padding='same')

    out = conv2d_bn(inp, filters, 3,3, activation='relu', padding='same')

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    for i in range(length-1):
        shortcut = out
        shortcut = conv2d_bn(shortcut, filters, 1,1, activation=None, padding='same')

        out = conv2d_bn(out, filters, 3,3, activation='relu', padding='same')

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization(axis=3)(out)

    return out


# MultiResUNet

def MultiResUnet(height, width, n_channels, base_filters=32):

    inputs = Input((height, width, n_channels))

    # Encoder

    b = int(base_filters)

    mresblock1 = MultiResBlock(b, inputs)
    pool1 = MaxPooling2D((2,2))(mresblock1)
    mresblock1 = ResPath(b, 4, mresblock1)

    mresblock2 = MultiResBlock(b * 2, pool1)
    pool2 = MaxPooling2D((2,2))(mresblock2)
    mresblock2 = ResPath(b * 2, 3, mresblock2)
    

    mresblock3 = MultiResBlock(b * 4, pool2)
    pool3 = MaxPooling2D((2,2))(mresblock3)
    mresblock3 = ResPath(b * 4, 2, mresblock3)
    

    mresblock4 = MultiResBlock(b * 8, pool3)
    pool4 = MaxPooling2D((2,2))(mresblock4)
    mresblock4 = ResPath(b * 8, 1, mresblock4)

    # Bridge

    mresblock5 = MultiResBlock(b * 16, pool4)

    # Decoder

    up6 = concatenate(
        [
            Conv2DTranspose(b * 8, (2, 2), strides=(2, 2), padding='same')(mresblock5),
            mresblock4,
        ],
        axis=3,
    )
    mresblock6 = MultiResBlock(b * 8, up6)

    up7 = concatenate(
        [
            Conv2DTranspose(b * 4, (2, 2), strides=(2, 2), padding='same')(mresblock6),
            mresblock3,
        ],
        axis=3,
    )
    mresblock7 = MultiResBlock(b * 4, up7)

    up8 = concatenate(
        [
            Conv2DTranspose(b * 2, (2, 2), strides=(2, 2), padding='same')(mresblock7),
            mresblock2,
        ],
        axis=3,
    )
    mresblock8 = MultiResBlock(b * 2, up8)

    up9 = concatenate(
        [
            Conv2DTranspose(b, (2, 2), strides=(2, 2), padding='same')(mresblock8),
            mresblock1,
        ],
        axis=3,
    )
    mresblock9 = MultiResBlock(b, up9)

    # Saída em float32 ajuda quando mixed precision está ligado no notebook
    conv10 = conv2d_bn(mresblock9, 1, 1, 1, activation='sigmoid')

    model = Model(inputs=[inputs], outputs=[conv10])

    return model

def main():
    # Definindo o modelo
    model = MultiResUnet(256, 256, 3)
    model.summary()


if __name__ == "__main__":
    main()
        