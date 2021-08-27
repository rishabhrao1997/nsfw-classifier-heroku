import flask
from flask import Flask, request
import os
import numpy as np
from pipeline import Pipeline
import urllib
import gc

app = Flask(__name__)

nsfw_classifier = Pipeline(model_config_path='model_essentials/nsfw_model_cfg.json', 
							weights_path = 'model_essentials/nsfw_weights.h5')

UPLOAD_PATH = 'static/images'
#home page
@app.route('/', methods = ['GET'])
def home_page():
    return flask.render_template('nsfw_classifier.html')

#results page
@app.route('/predict', methods = ['POST'])
def prediction():

	for file in os.listdir('static/images'):
		if file:
			os.remove('static/images/' + file)
	
	url = request.form['url']
	files = request.files
	if url:
		random_num = np.random.randint(0, 2000000, 1)
		file_path = f'static/images/{str(random_num)}.jpg'
		urllib.request.urlretrieve(url, file_path)
	
	elif files['image']:
		f = request.files['image']
		file_path = os.path.join(UPLOAD_PATH, f.filename)
		f.save(file_path)

	else:
		return flask.render_template('nsfw_classifier.html', image_loc = '', predicted_class = '')

	result = nsfw_classifier.prediction(file_path)

	gc.collect()

	return flask.render_template('nsfw_classifier.html', image_loc = file_path,
									predicted_class = result)


if __name__ == '__main__':
	app.run()
