# coding=utf-8
import tensorflow as tf
import numpy as np
import keras
import os
import time

# SQLite for information
import sqlite3

# Keras
from keras.models import load_model, model_from_json
from keras.preprocessing import image
from PIL import Image

# Flask utils
from flask import Flask, url_for, render_template, request,current_app,send_from_directory,redirect
from werkzeug.utils import secure_filename

#twilio stuffs 
from twilio.twiml.messaging_response import MessagingResponse
import requests
from twilio.rest import Client

# Define a flask app
app = Flask(__name__)




# load json file before weights
loaded_json = open("models/crop.json", "r")
# read json architecture into variable
loaded_json_read = loaded_json.read()
# close file
loaded_json.close()
# retreive model from json
loaded_model = model_from_json(loaded_json_read)
# load weights
loaded_model.load_weights("models/crop_weights.h5")
model1 = load_model("models/one-class.h5")
global graph
graph = tf.get_default_graph()


def info():
    conn = sqlite3.connect("models/crop.sqlite")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM crop")
    rows = cursor.fetchall()
    return rows


img_dict = {}
img_path = img_dict.get(entry)


#sending replies
resp = MessagingResponse()
msg = resp.message()


def leaf_predict(img_path):
    # load image with target size
    img = image.load_img(img_path, target_size=(256, 256))
    # convert to array
    img = image.img_to_array(img)
    # normalize the array
    img /= 255
    # expand dimensions for keras convention
    img = np.expand_dims(img, axis=0)

    with graph.as_default():
        opt = keras.optimizers.Adam(lr=0.001)
        loaded_model.compile(optimizer=opt, loss='mse')
        preds = model1.predict(img)
        dist = np.linalg.norm(img - preds)
        if dist <= 20:
            return "leaf"
        else:
            return "not leaf"


def model_predict(img_path):
    # load image with target size
    img = image.load_img(img_path, target_size=(256, 256))
    # convert to array
    img = image.img_to_array(img)
    # normalize the array
    img /= 255
    # expand dimensions for keras convention
    img = np.expand_dims(img, axis=0)

    with graph.as_default():
        opt = keras.optimizers.Adam(lr=0.001)
        loaded_model.compile(
            optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        preds = loaded_model.predict_classes(img)
        return int(preds)


#flask app for bot
@app.route('/test', methods=['POST', 'GET'])
def upload():
    #recieveing messeage
    sender = request.form.get('From')
    if int(request.values['NumMedia']) > 0:
        img = request.values['MediaUrl0']
        img_path[sender] = img
        leaf = leaf_predict(img_path)
        if leaf == "leaf":
            #Make prediction
            preds = model_predict(img_path)
            rows = info()
            res = np.asarray(rows[preds])
            value = (preds == int(res[0]))
            if value:
                Disease, Pathogen, Symptoms, Management = [i for i in res]
                return msg(Pathogen=Pathogen, Symptoms=Symptoms, Management=Management, result=Disease )
        else:
             return msg(Error="ERROR: UPLOADED IMAGE IS NOT A LEAF (OR) MORE LEAVES IN ONE IMAGE")
    return None         




if __name__ == '__main__':
    app.run()
