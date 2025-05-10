
# Author: vlarobbyk
# Version: 1.0
# Date: 2024-10-20
# Description: A simple example to process video captured by the ESP32-XIAO-S3 or ESP32-CAM-MB in Flask.


from flask import Flask, render_template, Response, send_file, stream_with_context, Request,request
from io import BytesIO

import cv2
import numpy as np
import requests

app = Flask(__name__)
# IP Address
_URL = 'http://192.168.34.157'
# Default Streaming Port
_PORT = '81'
# Default streaming route
_ST = '/stream'
SEP = ':'

stream_url = ''.join([_URL,SEP,_PORT,_ST])


def video_capture():
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=True)

    res = requests.get(stream_url, stream=True)
    for chunk in res.iter_content(chunk_size=100000):
        if len(chunk) > 100:
            try:
                img_data = BytesIO(chunk)
                cv_img = cv2.imdecode(np.frombuffer(img_data.read(), np.uint8), 1)
                if cv_img is None:
                    continue

                # Resize (opcional, si va lento)
                # cv_img = cv2.resize(cv_img, (320, 240))

                # Substracción de fondo
                fg_mask = bg_subtractor.apply(cv_img)

                # Limpieza de ruido
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

                # Detección de contornos (zonas con movimiento)
                contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if cv2.contourArea(cnt) > 500:  # umbral mínimo
                        x, y, w, h = cv2.boundingRect(cnt)
                        cv2.rectangle(cv_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Codificar y enviar frame
                (flag, encodedImage) = cv2.imencode(".jpg", cv_img)
                if not flag:
                    continue

                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                      bytearray(encodedImage) + b'\r\n')

            except Exception as e:
                print(e)
                continue
@app.route("/")
def index():
    return render_template("index.html")


@app.route('/video')
def show_video():
    return render_template('index.html', show_video=True)



@app.route('/procesar', methods=['POST'])
def procesar_imagen():
    result_img = None
    if request.method == 'POST':
        image_name = request.form['image']
        operation = request.form['operation']
        kernel_size = int(request.form['kernel'])
        path = f'static/images/{image_name}'
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

        if operation == 'erosion':
            filtered = cv2.erode(img, kernel)
        elif operation == 'dilation':
            filtered = cv2.dilate(img, kernel)
        elif operation == 'tophat':
            filtered = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
        elif operation == 'blackhat':
            filtered = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
        elif operation == 'combined':
            tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
            blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
            filtered = cv2.add(img, cv2.subtract(tophat, blackhat))
        combined = np.hstack((img, filtered))
        __, buffer = cv2.imencode('.jpg', combined)
        return send_file(BytesIO(buffer), mimetype='image/jpeg')

    cv2.imwrite('static/results/resultado.jpg', combined)
    return render_template('index.html', result_image=True)


if __name__ == "__main__":
    app.run(debug=False)