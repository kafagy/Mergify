import os
from werkzeug.utils import secure_filename
from werkzeug import generate_password_hash, check_password_hash, MultiDict
from flask import Flask, render_template, json, request, session, redirect, url_for, flash

app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))
app.secret_key = 'why would I tell you my secret key?'
app.config['UPLOAD_FOLDER'] = os.path.join(basedir, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filenames):
	return '.' in filenames[0] and '.' in filenames[1] and filenames[0].rsplit('.', 1)[1].lower() and filenames[1].rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload():
    return render_template('index.html')

@app.route('/', methods = ['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		files = [file.filename for file in request.files.getlist('file')]
		if '' in files:
			flash('No selected file')
			return redirect(request.url)
		if files and allowed_file(files):
			for file in request.files.getlist('file'):
				file.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))
			return 'Content Image: ' + files[0] + '*******' + ' Style Image: ' + files[1]

@app.route('/uploads/<filename>')
def uploaded_file(filename):
	return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
	app.run(port=5000)

