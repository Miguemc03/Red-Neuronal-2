from flask import Flask,render_template,request
import cv2
import numpy as np
from tensorflow import keras


app = Flask(__name__)

model = keras.models.load_model('mnist.keras')

@app.route('/')
def index():
    return render_template("formulario.html")

@app.route('/predict' , methods=['GET'])
def predict():
    image=request.args.get('image')

    img = cv2.imread(image)
    resized_img = cv2.resize(img, (28, 28))
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    inverted_image = cv2.bitwise_not(gray_img)
    inverted_image = np.expand_dims(inverted_image, axis=0)/255.0
    prediction = model.predict(inverted_image)
    return render_template("formulario.html", prediction=np.argmax(prediction))

if __name__ == '__main__':
    app.run()

