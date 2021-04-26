from flask import Flask,jsonify,request
#from keras.models import load_model
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model("model_v5.h5")
#print(model.summary())
print("Model Loaded Successfully")
classes = ['EB',
           'LB',
           'H']

@app.route('/',methods=['GET'])
def test():
    return "<h1>Agritech Crop Doctor Server is Up and Running</h1> \n POST a request on \"https://crop-doctor-namal.herokuapp.com/predict\" for prediction"

@app.route('/predict',methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'File not recieved'
    else:
        img_file = Image.open(request.files['image'])
        img = img_file.resize([224,224])
        img = np.array(img)
        img = tf.cast(img,tf.float32)
        return predict(img)

def predict(img):
    img = np.expand_dims(img,axis=0)
    result = model.predict_classes(img)
    disease_name = classes[result[0]]
    print("prediction from model: "+disease_name)
    return disease_name

if __name__ == '__main__':
    #app.run(host="0.0.0.0", port=5000, debug=False,threaded=False)
    app.run()
