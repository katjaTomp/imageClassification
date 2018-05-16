from keras import backend as K
from keras.applications import NASNetLarge,NASNetMobile
from keras.optimizers import Adam,RMSprop
from keras.layers import Input,merge,Dense
from keras import Model
input_shape = (150,150,3)

model = NASNetLarge(weights=None, input_shape=input_shape, pooling='max')


left_input = Input(input_shape)
right_input = Input(input_shape)

L1_distance = lambda x: K.abs(x[0] - x[1])

h1 = model(left_input)
h2 = model(right_input)

concat_features = merge([h1, h2], mode=L1_distance, output_shape=lambda x: x[0])

# were removed from below: bias_initializer= initializer.RandomNormal(mean=0.5,stddev=1e-2),kernel_initializer=initializer.RandomNormal(mean=0, stddev=2e-1)

prediction = Dense(1, activation="sigmoid")(concat_features)


siamese_net = Model(input=[left_input, right_input], output=prediction)
optimizer = RMSprop()
siamese_net.compile(optimizer=optimizer, loss="binary_crossentropy")

siamese_net.summary()