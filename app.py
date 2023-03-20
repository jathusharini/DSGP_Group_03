# Importing essential libraries and modules
from flask import Flask, jsonify, render_template, request, url_for, redirect, Response
import os
import cv2
import numpy as np
import tensorflow as tf
from jinja2 import escape
import pandas as pd
import pickle # the process of converting a Python object into a byte stream to store it in a file/database, maintain program state across sessions, or transport data over the network

#model = pickle.load(open('preprocessed.pkl','rb'))

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


# Creating a Flask app
app = Flask(__name__)

# Load the pre-trained model
model_path = 'models/preprocessed_model.pkl'
model = pickle.load(
  open(model_path, 'rb'))

# Initialize the video stream
video_stream = cv2.VideoCapture(0)

# Define the function to generate video frames
def generate_frames():
    while True:
        # Read a frame from the video stream
        success, frame = video_stream.read()
        if not success:
            break

        # Resize the frame to fit the input shape of the model
        #frame = cv2.resize(frame, (224, 224))

        # Convert the frame to a numpy array and normalize it
        #frame = np.array(frame, dtype=np.float32) / 255.

        
        prediction = model.predict(np.array([frame]))
        predicted_label = np.argmax(prediction)

        # Get the corresponding class names
        words = ['bye', 'hello', 'home', 'love', 'thankyou', 'angry','bank', 'brother', 'excuse me',
         'father', 'good evening', 'good morning', 'goodnight', 'happy', 'help', 'hospital',
         'how much', 'hungry', 'love', 'mother', 'police station', 'sad', 'school', 'sister',
         'sorry', 'welcome', 'what', 'when', 'where', 'who']  
        predicted_video = words[predicted_label]

        # Draw the predicted class name on the frame
        #cv2.putText(frame, predicted_video, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame as a response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Define the API endpoint for the live camera stream
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

#controls src="{{ url_for('video_feed') }}"

#Setting the source of the video element to the 1Flask API endpoint


# Define route to handle video upload and recognition
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Check if file was uploaded
        if 'src_2' not in request.models:
            return redirect(request.url)
        file = request.models['src_2']        

        # Perform recognition using CNN model
        return redirect(url_for('inlesson'))
    return render_template('inlesson.html')

# Define route to display recognition results
@ app.route('/inlesson')
def inlesson():
    title = 'In-Lesson'
    return render_template('inlessonpage.html', title=title)


# Defineing the web-based UI----------
# Render In-Lesson Page
#@ app.route('/inlesson')
#def inlesson():
 #   title = 'In-Lesson'
  #  return render_template('inlessonpage.html', title=title)

# Run Flask app
#if __name__ == '__main__':
 #   app.run(debug=True, port=5000)


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

# Render In-Lesson Page
#@ app.route('/inlesson')
#def inlesson():
    #title = 'In-Lesson'
   # return render_template('inlessonpage.html', title=title)

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




#@ app.route('/api',methods=['POST'])
#def predict():
 #   prediction = model.predict([[np.array(data['exp'])]])
  #  output = prediction[0]
   # return jsonify(output)


# Run Flask app
if __name__=='__main__':
    app.run(debug=True, port=5000)
