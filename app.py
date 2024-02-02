from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

# Load YOLO model
net = cv2.dnn.readNet('yolov2.cfg', 'yolov2.weights')
classes = open('coco.names').read().strip().split('\n')

def detect_person(frame):
    height, width = frame.shape[:2]

    # YOLO object detection
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Process the detection results
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == 'person':
                center_x, center_y, w, h = (detection[:4] * np.array([width, height, width, height])).astype(int)
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame

def generate_frames():
    cap = cv2.VideoCapture(0)  # 0 corresponds to the default camera

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame = detect_person(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

    cap.release()

@app.route('/')
def index():
    return render_template('index_person.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
