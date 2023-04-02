# Importing essential libraries and modules
from charset_normalizer import detect
from flask import Flask, jsonify, render_template, request, url_for, redirect, Response
import os
import cv2
import numpy as np
import tensorflow as tf
from jinja2 import escape
import pandas as pd
import pickle # the process of converting a Python object into a byte stream to store it in a file/database, maintain program state across sessions, or transport data over the network
import joblib
from keras.models import load_model
#model = pickle.load(open('preprocessed.pkl','rb'))

# ===============================================================================================
# ------------------------------------ FLASK APP ------------------------------------------------


# Creating a Flask app
app = Flask(__name__)

#Load the pre-trained model
model_path = 'models/preprocessed_model.pkl'
model = pickle.load(
  open(model_path, 'rb'))

# Define the video stream
cap = cv2.VideoCapture(0)

#------------------------------------------------------------------------------------------------------------------------
# Define a function to perform the sign language detection
def detect_sign_language(frame):
    # Preprocess the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    X = np.array(resized).reshape(-1, 28, 28, 1)
    
    # Make a prediction using the model
    y_pred = model.predict(X)
    label = np.argmax(y_pred)
   

# Define a function to read frames from a video file and detect sign language in real-time
def detect_sign_language_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while True:
        # Read a frame from the video file
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect sign language in the frame
        label = detect_sign_language(frame)
        
        # Draw the label on the frame
        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        
        # Convert the frame to a byte string and yield it to the client
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
#////////////////////////////////////////////////////////////////

def detect_objects(frame):
    # perform object detection on the frame
    classIds, confs, bbox = net.detect(frame)
     # draw bounding boxes and labels for detected persons
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId-1]
    
    # encode the frame as JPEG and return it
    ret, buffer = cv2.imencode('.jpg', frame)
    frame = buffer.tobytes()
    return frame

def detect_objects(frame, mode):
    global net, classNames
    
    # perform object detection on the frame
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerNames = net.getLayerNames()
    outputLayers = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputLayers)
    
    # draw bounding boxes and labels for detected persons
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if classNames[classId] == 'person':
                # draw the bounding box and label
                box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (x, y, w, h) = box.astype('int')
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, classNames[classId], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # encode the frame as JPEG and return it
    ret, buffer = cv2.imencode('.jpg', frame)
    frame = buffer.tobytes()
    return frame


def generate_frames(): 
    while True: 
        # read a frame from the camera 
        success, frame = cap.read() 
        if not success: 
            break 
         # process the frame 
        frame = detect_objects(frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    # Release the video file
    cap.release()

# Define a route to display the video stream
@app.route('/detect')
def detect():
    global cap, model
    ret, frame = cap.read()
    #image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #image = cv2.resize(image, (224, 224))
    #image = np.reshape(image, (1, 224, 224, 1))
    #prediction = model.predict(model)
    video_path = 'models/src_2'
    #result = prediction.argmax()
    return Response(detect_sign_language_video(video_path),mimetype='multipart/x-mixed-replace; boundary=frame'); return render_template('result.html')
   
#-----------------------------------------------------------------------------------------------------------------------------------

# Define the video stream generator function
#def gen():
#    while True:
#        ret, frame = cap.read()
#        yield (b'--frame\r\n'
#               b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', frame)[1].tobytes() + b'\r\n')
               
# Define the API endpoint for the live camera stream
@app.route('/video_feed')
def video_feed():
    return Response(detect_sign_language(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Render In-Lesson Page
@ app.route('/inlesson')
def inlesson():
    title = 'In-Lesson'
    return render_template('inlessonpage.html', title=title)

# Render Welcome Page
@ app.route('/')
def home():
    title = 'Sign Language Translator - Home'
    return render_template('welcomepage.html', title=title)

# Render SignUp Page
@ app.route('/signup')
def signup():
    title = 'Sign In'
    return render_template('signupform.html', title=title)

# Render Register Page
@ app.route('/register')
def register():
    title = 'Register'
    return render_template('register.html', title=title)

# Render Lesson Page
@ app.route('/lesson')
def lesson():
    title = 'Lesson'
    return render_template('lessonpage.html', title=title)

# Render FAQ Page
@ app.route('/FAQ')
def FAQ():
    title = 'FAQ'
    return render_template('FAQ.html', title=title)

# Render Register Page
@ app.route('/user guide')
def U_guide():
    title = 'User_Guide'
    return render_template('user guide.html', title=title)

# Run Flask app
if __name__=='__main__':
    app.run(debug=True, port=5000)
