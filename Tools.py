import keras
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras import models, layers, optimizers


def load_model():
    global trainable_model
    trainable_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    global network_model
    network_model = VGG16(weights="imagenet")
    global graph
    graph = tf.get_default_graph()


def prepare_image(file):
    image = load_img(file, target_size=(224, 224))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    vgg16_input = preprocess_input(image_array)

    return vgg16_input


def network_predict(image):
    image_to_predict = prepare_image(image)

    with graph.as_default():
        prediction = network_model.predict(image_to_predict)

    label = decode_predictions(prediction)

    return label


def set_model(request_type):
    if "org" in request_type:
        global network_model
        with graph.as_default():
            network_model = VGG16(weights="imagenet")
        print("org")
        return "Original Model Loaded"
    elif "catdog" in request_type:
        train_model()
        print("cat")
        return "Model Fine-Tuned for classifying cats and dogs"


def train_model():

    with graph.as_default():
        train_batchsize = 100
        val_batchsize = 10
        train_dir = './data/train'
        validation_dir = './data/validation'

        for layer in trainable_model.layers[:-4]:
            layer.trainable = False

        new_model = models.Sequential()

        new_model.add(trainable_model)
        new_model.add(layers.Flatten(input_shape=trainable_model.output_shape[1:]))
        new_model.add(layers.Dense(1024, activation='relu'))
        new_model.add(layers.Dropout(0.5))
        new_model.add(layers.Dense(2, activation='softmax'))

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        validation_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(224, 224),
            batch_size=train_batchsize,
            class_mode='categorical')

        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(224, 224),
            batch_size=val_batchsize,
            class_mode='categorical',
            shuffle=False)

        new_model.compile(loss='categorical_crossentropy',
                          optimizer=optimizers.RMSprop(lr=1e-4),
                          metrics=['acc'])

        new_model.fit_generator(
            train_generator,
            steps_per_epoch=train_generator.samples / train_generator.batch_size,
            epochs=5, # should be like 30 epochs, just for test
            validation_data=validation_generator,
            validation_steps=validation_generator.samples / validation_generator.batch_size,
            verbose=2)

        # Save the model
        new_model.save('small_last4.h5')

        global network_model
        network_model = new_model
