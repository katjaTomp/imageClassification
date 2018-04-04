from kombu import Connection, Exchange, Queue, Consumer
import socket
from keras.models import load_model, model_from_json
from keras.models import Model
from keras.applications import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input
import numpy as np
from keras.preprocessing import image as images

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True





def loadModel():
    """ Loads combined model

           :return  res_final_model  Model:  model is a combination of two classifiers, one is InceptionResNetV2 pretrained on Imagenet
           and the second is linear softmax classifier.

       """

    ## Load bottom model
    bottom_model = InceptionResNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))

    ## Load top model

    #json_file = open('../model_3/ah_pretrained_top_model_3.json', 'r')#binary - '../models_archit/ah_pretrained_top_model.json'
    #model_top_json = json_file.read()
    #json_file.close()
    #top_model = model_from_json( model_top_json)

    top_model = load_model('../model_3/ah_top_model_3.h5')

    ## Load weights
    #top_model.load_weights('../model_3/ah_pretrained_top_model_3.h5')# binary - ../weights/ah_pretrained_top_model.h5

    ## Combine two models
    res_final_model = Model(inputs=bottom_model.input, outputs=top_model(bottom_model.output))

    return res_final_model


def preprocess(imgPath):
    """
        Resizes and flattens the image to array
        :param imgPath: str , image path
        :return: array
        """
    img = Image.open(imgPath)
    img = img.resize((224, 224))
    x = images.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return x


def mapToCategory(pred):
    """
       Ouputs predicted category name

       :param pred: array
       :return: str
       """

    for prediction in pred:
        if prediction[0] > prediction[1]:
            return "id"
        else:
            return "other"


def predict(img):
    """
        Predicts the score for each class
        :param img: array
        :return: (needs to be json)
        """

    prediction = model.predict(img)

    print mapToCategory(prediction)


def process_message(body, message):
    """
        Parses the rabbitMQ message

        :param body: str, content of message
        :param message: object
        """

    img = preprocess(format(body))
    predict(img)

    message.ack()  ## removes the message from the queue


def consume():
    """
        Starts the consumer
        """
    new_connection = establish_connection()
    while True:
        try:
            new_connection.drain_events()  #timeout= 1 wait 1 sec to consume the message
        except socket.timeout:  ## if no message occured
            new_connection.heartbeat_check()


def establish_connection():
    """
        Establishes and maintains the connection
        :return: connection to RabbitMQ
        """

    revived_connection = conn.clone()
    revived_connection.ensure_connection(max_retries=5)
    channel = revived_connection.channel()
    consumer.revive(channel)
    consumer.consume()

    return revived_connection


def run():
    """
        Runs the consumer
        """
    while True:
        try:
            consume()
        except conn.connection_errors:
            pass



model = loadModel()
print("Model is loaded")

url = "amqp://localhost:5672/"

conn = Connection(url, heartbeat=10)
exchange = Exchange("example-exchange", type='direct')
queue = Queue(name="example-exchange", exchange=exchange, routing_key="IdPrediction")
consumer = Consumer(conn, queues=queue, callbacks=[process_message], accept=["text/plain"])
consumer.consume()
run()