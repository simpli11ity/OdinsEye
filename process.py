#!/usr/bin/env python
from flask import Flask, render_template, Response
import cv2
import sys
import numpy

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


# where all processing begins
def process_get_frame():
    camera_port=0
    camera=cv2.VideoCapture(camera_port)
    camera.set(cv2.CAP_PROP_FPS,100)
    backSub = cv2.createBackgroundSubtractorKNN()
    
    while True:
        retval, frame = camera.read()
        frame = cv2.flip(frame, 1)
        fgMask = backSub.apply(frame)

        shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        c, hierarchy = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        c = sorted(c, key=cv2.contourArea, reverse=True)
        cv2.drawContours(frame, c, -1, (120,120,63), 3)
    
        x, y, w, h = cv2.boundingRect(c[0])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        imgencode=cv2.imencode('.jpg',frame)[1]
        stringData=imgencode.tostring()
        yield (b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')
    
    camera.release()


@app.route('/output')
def output():
     return Response(process_get_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='localhost', port ="5000",debug=True, threaded=True)
