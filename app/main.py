# imports 
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
import os
from keras import *
from keras.layers import *
import tflite_runtime.interpreter as tflite
import ast
from numpy import *
import numpy as np
import itertools
import collections
import collections.abc
import sys
import subprocess
import json
from fastapi import FastAPI
import urllib.request
import uvicorn

def uninstall_package(package):
  subprocess.check_call(["pip", "uninstall", "-y", package])

uninstall_package('jaxlib')
uninstall_package('chex')
uninstall_package('distrax')
uninstall_package('dopamine-rl')
uninstall_package('flax')
uninstall_package('optax')
uninstall_package('orbax-checkpoint')
uninstall_package('trax')
uninstall_package('gensim')
uninstall_package('fastai')
uninstall_package('seaborn')
uninstall_package('matplotlib')
uninstall_package('blis')
uninstall_package('thinc')
uninstall_package('spacy')
uninstall_package('en-core-web-sm')
uninstall_package('language_data')
uninstall_package('jax')
uninstall_package('debugpy')
uninstall_package('scikit-image')
uninstall_package('torch')
uninstall_package('torchvision')
uninstall_package('torchaudio')
uninstall_package('torchtext')
uninstall_package('accelerate')
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
    return keras.layers.Reshape([1, int(max_length), int(fd)])(keras.ops.convert_to_tensor(d, dtype="float32"))

# loss function comparing actual and predicted sequences 
def custom_loss(y_true, y_pred):
    y_true = keras.layers.Reshape([-1])(y_true)
    y_pred = keras.layers.Reshape([-1])(y_pred)
    y_pred = keras.ops.slice(y_pred, [0], [len(y_true)])
    mid = (int((len(y_true))/15))
    y_true = keras.layers.Reshape([1, mid, 15])(y_true)
    y_pred = keras.layers.Reshape([1, mid, 15])(y_pred)
    sq = keras.ops.square(y_true-y_pred)
    return np.mean(sq)

interpreter = tflite.Interpreter(model_path="app/soniqmodel_small.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

'''
SoniqModel = keras.models.load_model("app/streamlinedmodel", custom_objects={"opt":opt, "custom_loss":custom_loss})
SoniqModel.compile()
'''
async def pred_seq(input_mp3, ml, feD):
    model_output, midi_data, note_events = predict(input_mp3)
    _list = dechain(note_events)
    _list = descalar(_list)
    input_data = _list
    x = process(input_data, ml, feD)
    interpreter.set_tensor(input_details[0]['index'], x)
    '''
    splines, amount = SoniqModel.predict(x)
    '''
    interpreter.invoke()
    splines = interpreter.get_tensor(output_details[0]['index'])[0]
    amount = interpreter.get_tensor(output_details[0]['index'])[1]
    splines = np.array(keras.ops.squeeze(splines)).tolist()
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
    urllib.request.urlretrieve(mp3url, "app/audio/ad.mp3")
    seq = await pred_seq("app/audio/ad.mp3", 136, 5)
    return {"sequences":seq}
