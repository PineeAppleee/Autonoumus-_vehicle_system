import socketio
import eventlet
import numpy as np
from flask import Flask
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2

# Initialize SocketIO server
sio = socketio.Server()
app = Flask(__name__)  # Flask app instance
speed_limit = 10  # Maximum speed limit for throttle control

# Function to preprocess the input image
def img_preprocess(img):
    img = img[60:135, :, :]  # Crop the image to remove unnecessary parts
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)  # Convert to YUV color space
    img = cv2.GaussianBlur(img, (3, 3), 0)  # Apply Gaussian blur
    img = cv2.resize(img, (200, 66))  # Resize the image to (200, 66)
    img = img / 255.0  # Normalize pixel values to [0, 1]
    return img

# Event handler for client connection
@sio.on('connect')
def connect(sid, environ):
    print('Client connected:', sid)
    send_control(0, 0)  # Send initial control values

# Function to send control commands (steering angle and throttle)
def send_control(steering_angle, throttle):
    sio.emit(
        'steer',
        data={
            'steering_angle': str(steering_angle),
            'throttle': str(throttle)
        }
    )

# Event handler for telemetry data
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # Get current speed
        speed = float(data['speed'])
        
        # Decode and preprocess the received image
        image = Image.open(BytesIO(base64.b64decode(data['image'])))
        image = np.asarray(image)
        image = img_preprocess(image)
        image = np.array([image])  # Convert to batch format
        
        # Predict steering angle using the loaded model
        steering_angle = float(model.predict(image))
        
        # Calculate throttle based on speed
        throttle = 1.0 - (speed / speed_limit)
        
        # Print the telemetry data
        print('Steering Angle: {:.3f}, Throttle: {:.3f}, Speed: {:.2f}'.format(
            steering_angle, throttle, speed))
        
        # Send control commands
        send_control(steering_angle, throttle)

# Main entry point
if __name__ == '__main__':
    # Load the trained model (ensure the correct path is used)
    model = load_model('model/model.h5')
    print("Model loaded successfully!")
    
    # Wrap Flask app with SocketIO's WSGI middleware
    app = socketio.WSGIApp(sio, app)
    
    # Start the server on port 4567
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
