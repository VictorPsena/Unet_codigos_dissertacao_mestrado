# Bibliotecas
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dense, Reshape, Multiply
from keras.models import Model


# Encoder
def encoder_block(filters, inputs, activation='relu'):
    x = Conv2D(filters, kernel_size = (3,3), padding = 'same', strides = 1, activation = activation)(inputs)
    s = Conv2D(filters, kernel_size = (3,3), padding = 'same', strides = 1, activation = activation)(x)
    p = MaxPooling2D(pool_size = (2, 2), padding = 'same')(s)
    return s, p 

# Baseline
def baseline_layer(filters, inputs, activation='relu'):
    x = Conv2D(filters, kernel_size = (3,3), padding = 'same', strides = 1, activation = activation)(inputs)
    x = Conv2D(filters, kernel_size = (3,3), padding = 'same', strides = 1, activation = activation)(x)
    return x

# Decoder
def decoder_block(filters, connections, inputs, activation='relu'):
    x = Conv2DTranspose(filters, kernel_size = (2,2), padding = 'same', activation = activation, strides = 2)(inputs)
    skip_connections = concatenate([x, connections], axis = -1)
    # x = Conv2D(filters, kernel_size = (2,2), padding = 'same', activation = activation)(skip_connections)
    x = Conv2D(filters, kernel_size = (3,3), padding = 'same', activation = activation)(skip_connections)
    x = Conv2D(filters, kernel_size = (3,3), padding = 'same', activation = activation)(x)
    return x


# U-Net Model
def unet(input_shape=(256, 256, 1), base_filters=64, max_filters=512, use_class_input=None):
    """U-Net configurável.

    Parâmetros
    - input_shape: tupla (H, W, C)
    - base_filters: nº de filtros do 1º bloco 
    - max_filters: teto de filtros ao dobrar em profundidade
    """

    # Defining the input layer
    inputs = Input(shape=input_shape) # 256x256x1

    # Encoder filters
    f1 = int(min(base_filters, max_filters))
    f2 = int(min(base_filters * 2, max_filters))
    f3 = int(min(base_filters * 4, max_filters))
    f4 = int(min(base_filters * 8, max_filters))
    bottleneck_filters = int(min(base_filters * 16, max_filters * 2))

    # Defining the encoder
    func = 'relu'
    s1, p1 = encoder_block(f1, inputs=inputs, activation=func) # 128x128x64
    s2, p2 = encoder_block(f2, inputs=p1, activation=func) # 64x64x128
    s3, p3 = encoder_block(f3, inputs=p2, activation=func) # 32x32x256
    s4, p4 = encoder_block(f4, inputs=p3, activation=func) # 16x16x512

    # Baseline (bottleneck)
    baseline = baseline_layer(bottleneck_filters, p4, activation=func) # 16x16x1024
    if use_class_input:
        class_input = Input(shape=(use_class_input,), name = 'animal_class')
        # Projeta o escalar para um vetor do damanho dos filtros do bottleneck
        class_embedding = Dense(bottleneck_filters, activation='relu')(class_input)
        # Reshape para multiplicação elemento a elemento
        class_embedding = Reshape((1, 1, bottleneck_filters))(class_embedding)
        # Modula os features do bottleneck com a informação da classe
        baseline = Multiply()([baseline, class_embedding])
    # Defining the decoder
    d1 = decoder_block(f4, connections=s4, inputs=baseline, activation=func) # 32x32x512
    d2 = decoder_block(f3, connections=s3, inputs=d1, activation=func) # 64x64x256
    d3 = decoder_block(f2, connections=s2, inputs=d2, activation=func) # 128x128x128
    d4 = decoder_block(f1, connections=s1, inputs=d3, activation=func) # 256x256x64

    # Output layer
    # Mantém saída em float32 
    outputs = Conv2D(1, kernel_size=1, padding='same', activation='sigmoid', dtype='float32')(d4) # 256x256x1

    # Finalizing the model
    if use_class_input:
        model = Model(inputs=[inputs, class_input], outputs=outputs, name='U-Net-Conditioned')
    else:
        model = Model(inputs=inputs, outputs=outputs, name='U-Net')

    return model