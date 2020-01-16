from flask import *
from werkzeug.utils import secure_filename
import os
from keras.preprocessing import image
from keras.models import load_model
import numpy as np

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def index():
	return render_template('upload.html')

@app.route('/upload',methods=['POST'])
def upload():
	
	allowed_formats = set(['.png','.gif','.jpg','.jpeg','.svg','.bmp'])

	classes = {
		1:'Speed limit (20km/h)',
		2:'Speed limit (30km/h)',
		3:'Speed limit (50km/h)',
		4:'Speed limit (60km/h)',
		5:'Speed limit (70km/h)',
		6:'Speed limit (80km/h)',
		7:'End of Speed limit (80km/h)',
		8:'Speed limit (100km/h)',
		9:'Speed limit (120km/h)',
		10:'No Passing',
		11:'No Passing veh over 3.5 tons',
		12:'Right-of-way at intersection',
		13:'Priority Road',
		14:'Yield',
		15:'Stop',
		16:'No Vehicles',
		17:'Veh > 3.5 tons Prohibited',
		18:'No entry',
		19:'General Caution',
		20:'Dangerous Curve Left',
		21:'Dangerous Curve Right',
		22:'Double Curve',
		23:'Bumpy Road',
		24:'Slippery Road',
		25:'Road Narrows On The Right',
		26:'Road Work',
		27:'Traffic Signals',
		28:'Pedestrians',
		29:'Children Crossing',
		30:'Bicycles Crossing',
		31:'Beware Of ice/Snow',
		32:'Wild Animals Crossing',
		33:'End Speed + Passing Limits',
		34:'Turn Right Ahead',
		35:'Turn Left Ahead',
		36:'Ahead Only',
		37:'Go Straight Or Right',
		38:'Go Straight or Left',
		39:'Keep Right',
		40:'Keep Left',
		41:'Roundabout mandatory',
		42:'End Of No Passing',
		43:'End No Passing Veh > 3.5 tons'
	}

	target = os.path.join(APP_ROOT, 'images/')
	
	if not os.path.isdir(target):
		os.mkdir(target)
	
	files = request.files.getlist('file')
	if files!=[]:
		for file in files:
			filename = file.filename

			ext = os.path.splitext(filename)[1]

			if ext.lower() not in allowed_formats:
				flag = 1
				return render_template('success.html', filename=filename, status = flag)


			destination = '/'.join([target,filename])
			file.save(destination)

		model = load_model('Traffic_Sign_Recognition_model.h5')

		filepath = target+filename

		img = image.load_img(filepath,target_size=(30,30,3))
		img = img.resize((30,30))
		img = np.expand_dims(img, axis=0)
		img = np.array(img)

		pred = model.predict_classes([img])[0]

		sign = classes[pred+1]

		return render_template('success.html',filename=filename,signname=sign)

	else:
		return render_template('success.html',filename='')

@app.route('/about')
def about():
	return render_template('about.html')

@app.route('/upload/<filename>')
def send_image(filename):
	return send_from_directory('images',filename)

if __name__ == '__main__':
	app.run(debug=True)
