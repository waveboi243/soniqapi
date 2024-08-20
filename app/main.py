# imports 
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow import math
import ast
from numpy import *
import numpy as np
import itertools
import collections
import collections.abc
import sys
import json
from fastapi import FastAPI
import urllib.request
import uvicorn

app = FastAPI()

# this function removes scalar (non-list) values from a given list
def descalar(_list):
    for x in range(0, len(_list)):
        if not isinstance(_list[x], collections.abc.Sequence):
            _list[x] = [_list[x]]
    return(_list)

def dechain(x):
    return list(itertools.chain.from_iterable(x))

# this function flattens a possibly nested list into a regular list
def flatten(l):
    for el in l:
        if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

def normal_input(x, min, max):
    return ((x+min)/max)

def normal_output(y):
    for x in range(0, 10):
        y[x] = ((y[x]+220)/440)
    y[10] = (y[10]/5)
    y[11] = (y[11]/300)
    for x in range(12, 15):
        y[x] = (y[x]/100)
    return y

def denormal_output(y):
    for x in range(0, 10):
        y[x] = ((y[x]*440)-220)
    y[10] = (y[10]*5)
    y[11] = (y[11]*300)
    for x in range(12, 15):
        y[x] = (y[x]*100)
    return y

# this function sums every 5th array to simplify feature extractions
def process(input, max_length, fd):
    d = input
    for i in range(1, int(len(d)/5)+1):
        d[(i*5)-1] = [sum(d[(i*5)-1])]
    d = list(flatten(d))
    # pads or truncates data to the given max amount of fd-element groups
    length = len(d)
    units = int(max_length * fd)
    if length < units: 
        d = d + [0] * (units - length)
    elif length > units: 
        d = d[:units]
    d = [normal_input(x, min=min(d), max=max(d)) for x in d]
    return tf.reshape(tf.constant(d, dtype="float32"), [1, int(max_length), int(fd)])

# loss function comparing actual and predicted sequences 
def custom_loss(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    y_pred = tf.slice(y_pred, [0], [len(y_true)])
    mid = (int((len(y_true))/15))
    y_true = tf.reshape(y_true, [1, mid, 15])
    y_pred = tf.reshape(y_pred, [1, mid, 15])
    sq = math.square(y_true-y_pred)
    return math.reduce_mean(sq)

# learning rate scheduler - credit to Daniel Onugha for tutorial
learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01, # DO NOT EDIT
    decay_steps=150,
    decay_rate=0.96
)

# optimizer with learning rate scheduler
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)

SoniqModel = keras.models.load_model("app/smallmodel", custom_objects={"opt":opt, "custom_loss":custom_loss})
SoniqModel.compile()

async def pred_seq(input_mp3, ml, feD):
    model_output, midi_data, note_events = predict(input_mp3)
    _list = dechain(note_events)
    _list = descalar(_list)
    input_data = _list
    x = process(input_data, ml, feD)
    splines, amount = SoniqModel.predict(x)
    splines = np.array(tf.squeeze(splines)).tolist()
    splines = list(map(denormal_output, splines))
    amount = int(amount[0][0] * 200)
    # truncates sequences to predicted number of valid sequences
    return splines[:(amount+1)]

# pred_seq(sys.argv[1], 136, 5)

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/mp3post/")
async def generateSeq(mp3url : str = ""):
    print(mp3url)
    urllib.request.urlretrieve(mp3url, "app/audio/ad.mp3")
    seq = await pred_seq("app/audio/ad.mp3", 136, 5)
    return {"sequences":seq}

if __name__ == "__main__":
    uvicorn.run(app, port=8000)