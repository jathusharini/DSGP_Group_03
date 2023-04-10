# Importing essential libraries and modules
import urllib.request

from charset_normalizer import detect
from flask import Flask, jsonify, render_template, request, url_for, redirect, Response
import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import webbrowser


# import pandas as pd
import pickle # the process of converting a Python object into a byte stream to store it in a file/database, maintain program state across sessions, or transport data over the network
import joblib
from keras.models import load_model
#model = pickle.load(open('preprocessed.pkl','rb'))

# ===============================================================================================
# ------------------------------------ FLASK APP ------------------------------------------------


# Creating a Flask app
app = Flask(__name__)

#Load the pre-trained model
model_path = 'preprocessed_model.pkl'
model = tf.keras.models.load_model('Final_Tf_mp_model')
words = ['angry', 'bank', 'brother', 'bye', 'excuse me', 'father', 'good evening', 'good morning', 'good night', 'happy', 'hello', 'help', 'home', 'hospital', 'how much', 'hungry', 'love', 'mother', 'police station', 'sad', 'school', 'sister', 'sorry', 'thankyou', 'welcome', 'what', 'when', 'where', 'who', 'why']
# Define the video stream
cap = cv2.VideoCapture(0)


url = None

def stretch(video, size):
    arr = np.array(video)
    n = len(arr)
    x = np.linspace(0, n - 1, n)
    new_x = np.linspace(0, n - 1, size)
    new_arr = np.zeros((size, len(video[0])))
    for i in range(size):
        new_arr[:, i] = np.interp(new_x, x, arr[:, i])
    
    return new_arr

# Preprocess video to get prediction
def get_prediction(video):
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
        vidcap = cv2.VideoCapture('recorded_video.webm')
        success,frame = vidcap.read()
        count = 0
        test = []
        while success:     
            success,frame = vidcap.read()
            if success:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmarks = hand_landmarks.landmark
                        landmarks_array = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks])
                        test.append(landmarks_array.flatten())

    test2 = stretch(test, 63)
    nptest = np.array([test2[:]])
    z = model.predict(nptest)
    print("test shape: ", nptest.shape, z)
    result = words[np.argmax(z[0])]
    if "goodnight" in url:
        if "good night" in result:
            print(result)
            answer(True,result)

        else:
            print("wrong decision!")
            answer(False,result)

    elif "bye" in url:
        if "bye" in result:
            print(result)
            answer(True,result)
        else:
            print("wrong decision!")
            answer(False,result)

    elif "goodmorning" in url:
        if "good morning" in result:
            print(result)
            answer(True, result)
        else:
            print("Wrong decision!")
            answer(False,result)

    elif "goodevening" in url:
        if "good evening" in result:
            print(result)
            answer(True, result)
        else:
            print("Wrong decision!")
            answer(False,result)

    elif "hello" in url:
        if "hello" in result:
            print(result)
            answer(True, result)
        else:
            print(result)
            answer(False,result)

    elif "sorry" in url:
        if "sorry" in result:
            print(result)
            answer(True, result)
        else:
            print(result)
            answer(False, result)

    elif "help" in url:
        if "help" in result:
            print(result)
            answer(True, result)
        else:
            print(result)
            answer(False, result)

    elif "welcome" in url:
        if "welcome" in result:
            print(result)
            answer(True, result)
        else:
            print(result)
            answer(False, result)

    elif "excuseme" in url:
        if "excuseme" in result:
            print(result)
            answer(True, result)
        else:
            print(result)
            answer(False, result)

    elif "thankyou" in url:
        if "thankyou" in result:
            print(result)
            answer(True, result)
        else:
            print(result)
            answer(False, result)

    elif "father" in url:
        if "father" in result:
            print(result)
            answer(True, result)
        else:
            print(result)
            answer(False, result)

    elif "mother" in url:
        if "mother" in result:
            print(result)
            answer(True, result)
        else:
            print(result)
            answer(False, result)

    elif "sister" in url:
        if "sister" in result:
            print(result)
            answer(True, result)
        else:
            print(result)
            answer(False, result)

    elif "brother" in url:
        if "brother" in result:
            print(result)
            answer(True, result)
        else:
            print(result)
            answer(False, result)

    elif "happy" in url:
        if "happy" in result:
            print(result)
            answer(True, result)
        else:
            print(result)
            answer(False, result)

    elif "sad" in url:
        if "sad" in result:
            print(result)
            answer(True, result)
        else:
            print(result)
            answer(False, result)

    elif "love" in url:
        if "love" in result:
            print(result)
            answer(True, result)
        else:
            print(result)
            answer(False, result)

    elif "hungry" in url:
        if "hungry" in result:
            print(result)
            answer(True, result)
        else:
            print(result)
            answer(False, result)

    elif "angry" in url:
        if "angry" in result:
            print(result)
            answer(True, result)
        else:
            print(result)
            answer(False, result)

    elif "homes" in url:
        if "homes" in result:
            print(result)
            answer(True, result)
        else:
            print(result)
            answer(False, result)

    elif "bank" in url:
        if "bank" in result:
            print(result)
            answer(True, result)
        else:
            print(result)
            answer(False, result)

    elif "school" in url:
        if "school" in result:
            print(result)
            answer(True, result)
        else:
            print(result)
            answer(False, result)

    elif "hospital" in url:
        if "hospital" in result:
            print(result)
            answer(True, result)
        else:
            print(result)
            answer(False, result)

    elif "policeststion" in url:
        if "policestation" in result:
            print(result)
            answer(True, result)
        else:
            print(result)
            answer(False, result)

    elif "how" in url:
        if "how" in result:
            print(result)
            answer(True, result)
        else:
            print(result)
            answer(False, result)

    elif "what" in url:
        if "what" in result:
            print(result)
            answer(True, result)
        else:
            print(result)
            answer(False, result)

    elif "when" in url:
        if "when" in result:
            print(result)
            answer(True, result)
        else:
            print(result)
            answer(False, result)

    elif "where" in url:
        if "where" in result:
            print(result)
            answer(True, result)
        else:
            print(result)
            answer(False, result)

    elif "who" in url:
        if "who" in result:
            print(result)
            answer(True, result)
        else:
            print(result)
            answer(False, result)

    elif "why" in url:
        if "why" in result:
            print(result)
            answer(True, result)
        else:
            print(result)
            answer(False, result)

    return result

def index():
    global url
    url = request.url
    return url


#------------------------------------------------------------------------------------------------------------------------
# Define a function to perform the sign language detection
def detect_sign_language(video):
    return get_prediction(video)
    # Preprocess the frame
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    # X = np.array(resized).reshape(-1, 28, 28, 1)
    
    # # Make a prediction using the model
    # y_pred = model.predict(X)
    # label = np.argmax(y_pred)
   

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

        else:
            # process the frame
            frame = detect_objects(frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            break

        # Release the video file
        cap.release()
def answer(condition,result):

    message = "Hello, this is a pop-up message!"
    if condition == True:
        message = "Result is "+result
    else:
        message = "Wrong prediction!"

    # Create an HTML file with the pop-up message
    html = f"""<html>
                <body>
                    <script>
                        alert("{message}")
                    </script>
                </body>
              </html>"""

    # Write the HTML code to a file
    with open("popup.html", "w") as f:
        f.write(html)

    # Open the HTML file in the default web browser
    webbrowser.open("popup.html")



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
@app.route('/video_feed', methods=['POST'])
def video_feed():
    video = request.files['video']
    video.save('recorded_video.webm')
    return Response(detect_sign_language(video), mimetype='multipart/x-mixed-replace; boundary=frame')

# Render In-Lesson Page
@ app.route('/inlesson')
def inlesson():
    title = 'In-Lesson'
    url = index()

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

@ app.route('/profile')
def profile():
    title = 'Profile'
    return render_template('profile.html', title=title)
#---------------------------------------------------------------------------------------------------------
#=========================================GREETINGS======================================================
@ app.route('/hello')
def hello():
    title = 'Hello'
    url = index()
    return render_template('inlessonpage.hello.html', title=title)

@ app.route('/bye')
def bye():
    title = 'Bye'
    url = index()
    return render_template('inlessonpage.bye.html', title=title)

@ app.route('/goodmorning')
def goodmorning():
    title = 'good morning'
    url = index()
    return render_template('inlessonpage.goodmorning.html', title=title)

@ app.route('/goodnight')
def goodnight():
    title = 'Good Evening'
    url = index();

    return render_template('inlessonpage.goodnight.html', title=title)

@ app.route('/goodevening')
def goodevening():
    title = 'Good evening'
    url = index()
    return render_template('inlessonpage.goodevening.html', title=title)

#=========================================GENERAL============================================================
@ app.route('/welcome')
def welcome():
    title = 'You Are Welcome'
    url = index()
    return render_template('inlessonpage.welcome.html', title=title)
@ app.route('/excuseme')
def excuseme():
    title = 'Excuse me'
    url = index()
    return render_template('inlessonpage.excuseme.html', title=title)
@ app.route('/thankyou')
def thankyou():
    title = 'Thankyou'
    url = index()
    return render_template('inlessonpage.thankyou.html', title=title)
@ app.route('/sorry')
def sorry():
    title = 'Sorry'
    url = index()
    return render_template('inlessonpage.sorry.html', title=title)
@ app.route('/help')
def help():
    title = 'Help'
    url = index()
    return render_template('inlessonpage.help.html', title=title)
#=========================================FAMILY=============================================================
@ app.route('/father')
def father():
    title = 'Father'
    url = index()
    return render_template('inlessonpage.father.html', title=title)
@ app.route('/mother')
def mother():
    title = 'Mother'
    url = index()
    return render_template('inlessonpage.mother.html', title=title)
@ app.route('/sister')
def sister():
    title = 'Sister'
    url = index()
    return render_template('inlessonpage.sister.html', title=title)
@ app.route('/brother')
def brother():
    title = 'Brother'
    url = index()
    return render_template('inlessonpage.brother.html', title=title)

#=========================================EMOTIONS=============================================================
@ app.route('/happy')
def happy():
    title = 'Happy'
    url = index()
    return render_template('inlessonpage.happy.html', title=title)
@ app.route('/sad')
def sad():
    title = 'Sad'
    url = index()
    return render_template('inlessonpage.sad.html', title=title)
@ app.route('/love')
def love():
    title = 'Love'
    url = index()
    return render_template('inlessonpage.love.html', title=title)
@ app.route('/hungry')
def hungry():
    title = 'Hungry'
    url = index()
    return render_template('inlessonpage.hungry.html', title=title)
@ app.route('/angry')
def angry():
    title = 'Angry'
    url = index()
    return render_template('inlessonpage.angry.html', title=title)
#=========================================PLACES=============================================================
@ app.route('/homes')
def homes():
    title = 'Home'
    url = index()
    return render_template('inlessonpage.home.html', title=title)
@ app.route('/hospital')
def hospital():
    title = 'Hospital'
    url = index()
    return render_template('inlessonpage.hospital.html', title=title)
@ app.route('/school')
def school():
    title = 'School'
    url = index()
    return render_template('inlessonpage.school.html', title=title)
@ app.route('/bank')
def bank():
    title = 'Bank'
    url = index()
    return render_template('inlessonpage.bank.html', title=title)
@ app.route('/policestation')
def policestation():
    title = 'Police station'
    url = index()
    return render_template('inlessonpage.policestation.html', title=title)
#=========================================QUESTION WORDS=============================================================
@ app.route('/how')
def how():
    title = 'How'
    url = index()
    return render_template('inlessonpage.how.html', title=title)
@ app.route('/what')
def what():
    title = 'What'
    url = index()
    return render_template('inlessonpage.what.html', title=title)
@ app.route('/when')
def when():
    title = 'When'
    url = index()
    return render_template('inlessonpage.when.html', title=title)
@ app.route('/where')
def where():
    title = 'Where'
    url = index()
    return render_template('inlessonpage.where.html', title=title)
@ app.route('/who')
def who():
    title = 'Who'
    url = index()
    return render_template('inlessonpage.who.html', title=title)
@ app.route('/why')
def why():
    title = 'Why'
    url = index()
    return render_template('inlessonpage.why.html', title=title)


# Run Flask app
if __name__=='__main__':
    app.run(debug=True, port=5000)
