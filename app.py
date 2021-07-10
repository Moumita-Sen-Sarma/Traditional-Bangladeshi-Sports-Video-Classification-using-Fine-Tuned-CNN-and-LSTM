import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from keras_video import VideoFrameGenerator
from tensorflow.keras.models import model_from_json
import os
import urllib.request

from flask import Flask, flash, request, redirect, render_template, jsonify
from werkzeug.utils import secure_filename

model_path = 'vgg19_last_8_layer.json' # the path of json file of trained model
weight_path = 'vgg19_last_8_layer_2.h5' # the path of weight file of trained model

cls={'Boli_Khela':0, 'Kabaddi':1 ,'Kho Kho':2, 'Lathi Khela':3, 'Nouka Baich':4} # categorical values of classes of the dataset


def get_frames(filename):
  SIZE = (128, 128)
  CHANNELS = 3
  NBFRAME = 20
  data_aug = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
  test=VideoFrameGenerator(glob_pattern=filename,nb_frames=NBFRAME,shuffle=False, batch_size=1, 
                           target_shape=SIZE, transformation=data_aug, use_frame_cache=False)
  return test

def get_model(model_path,weight_path):  
  json_file = open(model_path, 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  # load weights into new model
  loaded_model.load_weights(weight_path)
  return loaded_model

def predict_class(filename, model_path=model_path, weight_path=weight_path,classes=cls):
    video_frames = get_frames(filename)
    model=get_model(model_path,weight_path)
    Y_pred = model.predict_generator(video_frames, 1)
    y_pred = np.argmax(Y_pred, axis=1)
    label = [k for k, v in classes.items() if v == y_pred[0]][0]
    return label


UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


ALLOWED_EXTENSIONS = set(['mp4', 'avi'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('home.html')

# def get_class(filename):
#   clss=predict_class(path+filename)
#   return clss

# def get_class(name):
#   return name

@app.route('/upload', methods=['POST'])
def upload_file():
  # check if the post request has the file part
  if 'upvid[]' not in request.files:
    resp = jsonify({'message' : 'No file part in the request'})
    resp.status_code = 400
    return resp
  
  files = request.files.getlist('upvid[]')
  
  errors = {}
  success = False
  
  for file in files:
    if file and allowed_file(file.filename):
      filename = secure_filename(file.filename)
      file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
      success = True
    else:
      errors[file.filename] = 'File type is not allowed'
  
  if success and errors:
    errors['message'] = 'File(s) successfully uploaded'
    resp = jsonify(errors)
    resp.status_code = 206
    return resp
  if success:
    resp = jsonify({'message' : 'Files successfully uploaded'})
    resp.status_code = 201
    return resp
  else:
    resp = jsonify(errors)
    resp.status_code = 400
    return resp


@app.route('/get_name')
def get_name():
    name = request.args.get('name', 0, type=str)
    # print(name)
    name = name.split('\\')[-1]
    name = UPLOAD_FOLDER+name
    print(name)
    label = predict_class(name)
    return jsonify(result=label)


if __name__ == "__main__":
    app.run(debug=True)