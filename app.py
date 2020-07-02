import os
from flask import Flask,request,redirect,url_for,render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

#loading the model

from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image,ImageOps

model = load_model('digits.h5')

@app.route('/',methods=["GET","POST"])
def main_page():
	if request.method == 'POST':
		file = request.files['file']
		filename = secure_filename(file.filename)
		file.save(os.path.join('uploads',filename))
		return redirect(url_for('prediction',filename=filename))
		#return redirect(url_for('display',filename=filename))
	return render_template('index.html')
	
@app.route('/prediction/<filename>')
def prediction(filename):
	img_size = 28
	img_array = cv2.imread(os.path.join('uploads',filename),cv2.IMREAD_GRAYSCALE)
	img_array = cv2.bitwise_not(img_array)
	new_array = cv2.resize(img_array,(28,28))
	new_array = paddedImg(new_array)
	new_array = new_array.reshape((1,img_size * img_size)).astype("float32")
	new_array /= 255
	print("image ready")
	predictions = model.predict_classes(new_array)
	print(predictions)
	return render_template('prediction.html',predictions=predictions)
	
def paddedImg(img):
	ht,wd = img.shape
	ww = 50
	hh = 50
	color = 0
	result = np.full((hh,ww),color,dtype=np.uint8)
	xx = (ww - wd) // 2
	yy = (hh - ht) // 2
	result[yy:yy+ht, xx:xx+wd] = img
	return cv2.resize(result,(28,28))
if __name__ == "__main__":	
	app.run(debug=True)


