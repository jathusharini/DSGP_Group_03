const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const sendBtn = document.getElementById('sendBtn');
const video = document.getElementById('video');
let mediaRecorder;
let recordedChunks = [];

// User clicks the "Start Recording" button
startBtn.addEventListener('click', () => {
  // Request access to the user's camera and microphone
  navigator.mediaDevices.getUserMedia({ video: true, audio: true })
    .then((stream) => {
      // Display the video stream in the video element
      video.srcObject = stream;
      video.play();

      // Create a MediaRecorder instance to record the stream
      mediaRecorder = new MediaRecorder(stream);

      // Add event listener for when data is available
      mediaRecorder.addEventListener('dataavailable', (event) => {
        // Push the recorded data to the recordedChunks array
        recordedChunks.push(event.data);
      });

      // Start recording
      mediaRecorder.start();
    })
    .catch((error) => {
      console.error(`Error accessing media devices: ${error}`);
    });
});

// User clicks the "Stop Recording" button
stopBtn.addEventListener('click', () => {
  // Stop recording
  mediaRecorder.stop();

  // Stop the video stream in the video element
  video.srcObject.getTracks().forEach((track) => track.stop());
});

// User clicks the "Send Video to Server" button
sendBtn.addEventListener('click', () => {
  // Create a Blob object from the recordedChunks array
  const recordedBlob = new Blob(recordedChunks, { type: 'video/webm' });

  // Create a FormData object to send the recorded video to the server
  const formData = new FormData();
  formData.append('video', recordedBlob);

  // Use the fetch API to send the recorded video to the server
  fetch('/video_feed', {
    method: 'POST',
    body: formData
  })
    .then((response) => {
      if (response.ok) {
        console.log('Recorded video sent to server!');
      } else {
        console.error(`Server returned ${response.status} ${response.statusText}`);
      }
    })
    .catch((error) => {
      console.error(`Error sending recorded video to server: ${error}`);
    });
});