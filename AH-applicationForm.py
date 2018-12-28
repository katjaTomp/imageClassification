#from keras import backend as k

#if k.backend() == 'tensorflow':
#    k.set_image_dim_ordering("th")

from keras.utils import np_utils
import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.optimizers import SGD, RMSprop, Adagrad, Adam
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNet,InceptionResNetV2
from keras.callbacks import ModelCheckpoint
from keras.models import load_model, model_from_json
from keras.models import Model

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

seed = np.random.seed(12344)
image_size = 224
batch_size = 200
num_classes = 2
epochs = 150

## Training data
train_data = 'data/forms_2cat_1/train/'

training_instances = 34200
ids = 19729
other_instances = 14471

### Testing data
test_data = 'data/forms_2cat_1/test/'

test_instances = 6000
test_ids = 4356
test_other = 1644



def loadDataset():
    generator_training = ImageDataGenerator(rescale=1. / 255,
                                            horizontal_flip=True,
                                            vertical_flip=True,
                                            rotation_range=0.4,
                                            shear_range=0.2
                                            )

    augmented_training = generator_training.flow_from_directory(train_data,
                                                                target_size=(image_size, image_size),
                                                                class_mode="categorical",
                                                                shuffle=True,
                                                                batch_size=batch_size,
                                                                seed=seed)
    generator_testing = ImageDataGenerator(rescale=1. / 255)
    augmented_testing = generator_testing.flow_from_directory(test_data,
                                                              target_size=(image_size, image_size),
                                                              class_mode="categorical",
                                                              shuffle=True,
                                                              seed=seed)
    return [augmented_training, augmented_testing]


### Create the model

def model_custom():
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(3, image_size, image_size)))
    model.add(BatchNormalization())  ## keep the activations around mean zero and std 1
    model.add(Activation("relu"))  ## add non linearity
    model.add(Dropout(0.6))

    model.add(Convolution2D(32, 3, 3))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.6))


    model.add(MaxPooling2D(pool_size=(2, 2)))  ## downsample
    model.add(Flatten())
    model.add(Dense(128))  ## fully connected layer

    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.6))

    model.add(Dense(num_classes))
    model.add(BatchNormalization())
    model.add(Activation("softmax"))

    return model


def train_network(model, train, test):
    from time import time
    from keras.callbacks import TensorBoard

    tensorboard = TensorBoard(log_dir="logs/resnet_CosmeticsBrands/{}".format(time()), histogram_freq=0,
                              write_graph=True,
                              write_images=True)

    optimizer = RMSprop()

    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit_generator(train, steps_per_epoch=training_instances // batch_size,
                        epochs=epochs,
                        validation_data=test,
                        validation_steps=test_instances // batch_size,
                        callbacks=[tensorboard])

    ### save the model
    model.save('models_archit/smallModel.h5')

def trainFromScratch_MobileNet():
    from time import time
    from keras.callbacks import TensorBoard
    tensorboard = TensorBoard(log_dir="logs/resnet_CosmeticsBrands/{}".format(time()), histogram_freq=0,
                              write_graph=True,
                              write_images=True)

    checkPoint = ModelCheckpoint(filepath='weights/ah_mobile_2cats.h5', save_best_only=True)

    model = MobileNet(weights="imagenet",  include_top=False, input_shape=(image_size,image_size,3))

    genterator_training = ImageDataGenerator(rescale=1. / 255,
                                             #rotation_range=0.4,
                                             #shear_range=0.2,
                                             #zoom_range=0.1,
                                             #width_shift_range=0.1,
                                             #height_shift_range=0.1,
                                             vertical_flip=True,
                                             horizontal_flip=True)

    augmented_training = genterator_training.flow_from_directory(train_data,
                                                                 target_size=(image_size, image_size),
                                                                 class_mode="categorical",
                                                                 shuffle=False,
                                                                 batch_size=batch_size)

    generator_test = ImageDataGenerator(rescale=1. / 255)
    augmented_testing = generator_test.flow_from_directory(test_data,
                                                           target_size=(image_size, image_size),
                                                           class_mode="categorical",
                                                           batch_size=batch_size,
                                                           shuffle=False)

#    optimizer_1 = RMSprop(lr=0.0005)
#    optimizer_2 = Adagrad()
#    model.compile(optimizer_1, loss='categorical_crossentropy', metrics=["accuracy"])

    training_bottleneck_features = model.predict_generator(augmented_training, steps=training_instances // batch_size)
    np.save(open('weights/ah_trainig_bottleneck.npy', 'w'), training_bottleneck_features)

    testing_bottleneck_features = model.predict_generator(augmented_testing, steps=test_instances // batch_size)
    np.save(open('weights/ah_testing_bottleneck.npy','w'), testing_bottleneck_features)



#  history = model.fit_generator(augmented_training,
#                               steps_per_epoch=training_instances // batch_size,
#                                  epochs=epochs,
#                                  validation_data=augmented_testing,
#                                  validation_steps=test_instances // batch_size,
#                                  callbacks=[tensorboard, checkPoint])

#    model.save_weights('weights/resnets/ah_subclasses_coefficients_MobileNet_2.h5')

#    model.save('models_archit/ah_MobileNet_2.h5')



def pretrained_InceptionResnet():



        model = InceptionResNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))

        # save only model
        model_json = model.to_json()
        with open('models_archit/InceptionResnets_ah_pretrained_top_model.json', 'w') as json:
            json.write(model_json)

        genterator_training = ImageDataGenerator(rescale=1. / 255,
                                                 rotation_range=0.4,
                                                 shear_range=0.2,
                                                 zoom_range=0.1,
                                                 width_shift_range=0.1,
                                                 height_shift_range=0.1,
                                                 vertical_flip=True,
                                                 horizontal_flip=True)

        augmented_training = genterator_training.flow_from_directory(train_data,
                                                                     target_size=(image_size, image_size),
                                                                     class_mode="categorical",
                                                                     shuffle=False,
                                                                     batch_size=batch_size)

        generator_test = ImageDataGenerator(rescale=1. / 255)
        augmented_testing = generator_test.flow_from_directory(test_data,
                                                               target_size=(image_size, image_size),
                                                               class_mode="categorical",
                                                               batch_size=batch_size,
                                                               shuffle=False)



        training_bottleneck_features = model.predict_generator(augmented_training,
                                                               steps=training_instances // batch_size)
        np.save(open('weights/ah_trainig_bottleneck.npy', 'w'), training_bottleneck_features)

        testing_bottleneck_features = model.predict_generator(augmented_testing, steps=test_instances // batch_size)
        np.save(open('weights/ah_testing_bottleneck.npy', 'w'), testing_bottleneck_features)

def data():
    from keras.utils.np_utils import to_categorical

    train_data = np.load(open('weights/ah_trainig_bottleneck.npy'))
    train_labels = to_categorical(np.array([0] * ids + [1]*other_instances), 2)

    validation_data = np.load(open('weights/ah_testing_bottleneck.npy'))
    validation_labels = to_categorical(np.array([0] * test_ids + [1] * test_other ),2)

    return train_data, train_labels, validation_data, validation_labels


def trainTopModel(x_train, y_train, x_test, y_test):
    from time import time
    from keras.callbacks import TensorBoard


    tensorboard = TensorBoard(log_dir="logs/resnet_CosmeticsBrands/{}".format(time()), histogram_freq=0,
                              write_graph=True,
                              write_images=True)

    checkPoint = ModelCheckpoint(filepath='weights/ah_resnets_2cats_2.h5', save_best_only=True)


    model_top = Sequential()
    model_top.add(Flatten(input_shape=x_train.shape[1:]))
    model_top.add(Dense(256, activation="relu"))
    model_top.add(Dropout(0.6))
    model_top.add(Dense(num_classes, activation='softmax'))

    optimizer_rmsprop = RMSprop(lr=0.0005)
    optimizer_adam = Adam()

    model_top.compile(optimizer=optimizer_rmsprop, loss="categorical_crossentropy", metrics=['accuracy'])

    model_top.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  callbacks=[tensorboard,checkPoint])
    ## save models and weights

    model_top.save_weights('weights/ah_pretrained_top_model_2.h5')
    model_top.save('models_archit/ah_top_model.h5')


    # save only model
    model_json = model_top.to_json()
    with open('models_archit/ah_pretrained_top_model.json', 'w') as json:
        json.write(model_json)


def loadModel():

    ## Load bottom model
    bottom_model = InceptionResNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))

    ## Load top model

    json_file = open('models_archit/ah_pretrained_top_model.json', 'r')
    model_top_json = json_file.read()
    json_file.close()
    top_model = model_from_json( model_top_json)

    ## Load weights
    top_model.load_weights('weights/ah_pretrained_top_model.h5')

    ## Combine two models
    res_final_model = Model(inputs=bottom_model.input, outputs=top_model(bottom_model.output))

    return res_final_model


def preprocess_img(imgPath):
    import PIL.Image as Image
    from keras.applications.inception_resnet_v2 import preprocess_input
    import numpy as np
    from keras.preprocessing import image as images


    img = Image.open(imgPath)
    img = img.resize((224, 224))
    x = images.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return x

def predict(model, img):

    prediction = model.predict(img)

    for pred in prediction:
        print(pred)
        print(max(pred))



### custom data
#data = loadDataset()
#model = model_custom()
#train_network(model, data[0], data[1])


### mobile net
#trainFromScratch_MobileNet()



## pre-trained
#pretrained_InceptionResnet()
#x_train, y_train, x_test, y_test = data()
#trainTopModel(x_train, y_train, x_test, y_test)

model = loadModel()

imags = []

for i in imags:
    print(i)
    img = preprocess_img(i)
    predict(model, img)

