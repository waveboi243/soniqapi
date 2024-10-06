#!/usr/bin/env python

# imports 
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
import os
import tensorflow as tf
import ast
from numpy import *
import numpy as np
import itertools
import collections
import collections.abc
import sys
import subprocess
import json
from fastapi import FastAPI, Response
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
import urllib.request
import uvicorn
import base64
from io import BytesIO
from pydub import AudioSegment
import wave
import re
from pydantic import BaseModel
import requests
import json
import ast

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=['*'],
        #allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*']
    )
]

app = FastAPI(middleware=middleware)

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
    return tf.reshape(tf.convert_to_tensor(d, dtype="float32"), [1, int(max_length), int(fd)])

# loss function comparing actual and predicted sequences 
def custom_loss(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    y_pred = tf.slice(y_pred, [0], [len(y_true)])
    mid = (int((len(y_true))/15))
    y_true = tf.reshape(y_true, [1, mid, 15])
    y_pred = tf.reshape(y_pred, [1, mid, 15])
    sq = tf.square(y_true-y_pred)
    return np.mean(sq)

interpreter = tf.lite.Interpreter(model_path="app/soniqmodel_small.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

'''
SoniqModel = keras.models.load_model("app/streamlinedmodel", custom_objects={"opt":opt, "custom_loss":custom_loss})
SoniqModel.compile()
'''

def bytes_to_wav(byte_data, filename):
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(44100)
        wav_file.writeframes(byte_data)
    print(wav_file.getnframes())
async def pred_seq(input_mp3, ml, feD):
    url = 'https://basicpitchapi.onrender.com/inference'
    obj = {'audioData': input_mp3}
    response = requests.post(url, json = obj)
    r = json.loads(response.text)
    n = ast.literal_eval(r["note_events"])
    _list = dechain(n)
    _list = descalar(_list)
    input_data = _list
    x = process(input_data, ml, feD)
    interpreter.set_tensor(input_details[0]['index'], x)
    '''
    splines, amount = SoniqModel.predict(x)
    '''
    interpreter.invoke()
    splines = interpreter.get_tensor(output_details[0]['index'])
    print("///// Splines: " + str(splines))
    amount = interpreter.get_tensor(output_details[1]['index'])
    print("///// Amount: " + str(amount))
    splines = np.array(tf.squeeze(splines)).tolist()
    splines = list(map(denormal_output, splines))
    amount = int(amount[0] * 200)
    # truncates sequences to predicted number of valid sequences
    return splines[:(amount+1)]

# pred_seq(sys.argv[1], 136, 5)

@app.get("/")
def read_root():
    return {"Hello": "World"}

class AudioData(BaseModel):
    audioData: str

@app.post("/mp3post/")
async def generateSeq(audioData : AudioData):
    seq = await pred_seq(audioData.audioData, 68, 5)
    return {"sequences":seq}

if __name__ == "__main__":
    uvicorn.run("main:app", host='0.0.0.0',  port=8000, log_level="info")
