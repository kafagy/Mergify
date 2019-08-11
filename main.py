import os
import time
import torch
import numpy as np
from PIL import Image
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms, models
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, flash, send_from_directory, url_for, Response, session

hostname = 'kafagy.ddns.net'
#hostname = '127.0.0.1'
port = 9201
app = Flask(__name__)
app.debug = True
UPLOAD_FOLDER = './uploads'
app.secret_key = 'why would I tell you my secret key?'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filenames):
	return '.' in filenames[0] and '.' in filenames[1] and filenames[0].rsplit('.', 1)[1].lower() and filenames[1].rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# force browser to hold no cache. Otherwise old result might return.
@app.after_request
def set_response_headers(response):
	response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
	response.headers['Pragma'] = 'no-cache'
	response.headers['Expires'] = '0'
	return response

@app.route('/')
def result():
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
			print('Content Image: ' + files[0] + '*******' + ' Style Image: ' + files[1])
			session['content'] = os.path.join(app.config['UPLOAD_FOLDER'], files[0])
			session['style'] = os.path.join(app.config['UPLOAD_FOLDER'], files[1])
			return redirect(url_for('train'))

@app.route('/show')
def show():
	return redirect(url_for('uploaded_file', filename='result.png'))

@app.route('/show/<filename>')
def uploaded_file(filename):
	filename = 'http://{}:{}/uploads/'.format(hostname, port) + filename
	return render_template('result.html', filename=filename)

@app.route('/uploads/<filename>')
def send_file(filename):
	return send_from_directory(UPLOAD_FOLDER, filename)

def load_image(img_path, max_size=400, shape=None):
	""" Load in and transform an image, making sure the image
	   is <= 400 pixels in the x-y dims."""

	image = Image.open(img_path).convert('RGB')

	# large images will slow down processing
	if max(image.size) > max_size:
		size = max_size
	else:
		size = max(image.size)

	if shape is not None:
		size = shape

	in_transform = transforms.Compose([
						transforms.Resize(size),
						transforms.ToTensor(),
						transforms.Normalize((0.485, 0.456, 0.406),
								 (0.229, 0.224, 0.225))])
	# discard the transparent, alpha channel (that's the :3) and add the batch dimension
	image = in_transform(image)[:3,:,:].unsqueeze(0)

	return image

# helper function for un-normalizing an image
# and converting it from a Tensor image to a NumPy image for display
def im_convert(tensor):
	""" Display a tensor as an image. """

	image = tensor.to("cpu").clone().detach()
	image = image.numpy().squeeze()
	image = image.transpose(1,2,0)
	image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
	image = image.clip(0, 1)

	return image

def gram_matrix(tensor):
	""" Calculate the Gram Matrix of a given tensor
		Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
	"""

	# get the batch_size, depth, height, and width of the Tensor
	_, d, h, w = tensor.size()

	# reshape so we're multiplying the features for each channel
	tensor = tensor.view(d, h * w)

	# calculate the gram matrix
	gram = torch.mm(tensor, tensor.t())

	return gram

def get_features(image, model, layers=None):
	""" Run an image forward through a model and get the features for
		a set of layers. Default layers are for VGGNet matching Gatys et al (2016)
	"""

	## Need the layers for the content and style representations of an image
	if layers is None:
		layers = {'0': 'conv1_1',
				  '5': 'conv2_1',
				  '10': 'conv3_1',
				  '19': 'conv4_1',
				  '21': 'conv4_2',  ## content representation
				  '28': 'conv5_1'}

	features = {}
	x = image
	# model._modules is a dictionary holding each module in the model
	for name, layer in model._modules.items():
		x = layer(x)
		if name in layers:
			features[layers[name]] = x

	return features

@app.route('/train')
def train():
	return 	render_template('training.html')

@app.route('/training', methods = ['GET', 'POST'])
def training():
	content = session.get('content', None)
	style = session.get('style', None)
	# get the "features" portion of VGG19 (we will not need the "classifier" portion)
	vgg = models.vgg19(pretrained=True).features

	# freeze all VGG parameters since we're only optimizing the target image
	for param in vgg.parameters():
		param.requires_grad_(False)

	# move the model to GPU, if available
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	vgg.to(device)

	# load in content and style image
	content = load_image(content).to(device)
	# Resize style to match content, makes code easier
	style = load_image(style, shape=content.shape[-2:]).to(device)

	# display the images
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
	# content and style ims side-by-side
	ax1.imshow(im_convert(content))
	ax2.imshow(im_convert(style))

	# print out VGG19 structure so you can see the names of various layers
	print(vgg)

	# get content and style features only once before training
	content_features = get_features(content, vgg)
	style_features = get_features(style, vgg)

	# calculate the gram matrices for each layer of our style representation
	style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

	# create a third "target" image and prep it for change
	# it is a good idea to start of with the target as a copy of our *content* image
	# then iteratively change its style
	target = content.clone().requires_grad_(True).to(device)

	# weights for each style layer
	# weighting earlier layers more will result in *larger* style artifacts
	# notice we are excluding `conv4_2` our content representation
	style_weights = {'conv1_1': 1.,
					 'conv2_1': 0.75,
					 'conv3_1': 0.2,
					 'conv4_1': 0.2,
					 'conv5_1': 0.2}

	content_weight = 1  # alpha
	style_weight = 1e6  # beta

	# for displaying the target image, intermittently
	show_every = 400

	# iteration hyperparameters
	optimizer = optim.Adam([target], lr=0.003)

	def progress():
		steps = 2000  # decide how many iterations to update your image (5000)
		ii = 0
		while ii <= steps:
			print(ii)
			yield "data:" + str(ii) + "\n\n"
			target_features = get_features(target, vgg) # get the features from your target image
			content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)     # the content loss
			style_loss = 0 # initialize the style loss to 0
			for layer in style_weights: # then add to it for each layer's gram matrix loss
				target_feature = target_features[layer] # get the "target" style representation for the layer
				target_gram = gram_matrix(target_feature)
				_, d, h, w = target_feature.shape
				style_gram = style_grams[layer] # get the "style" style representation
				layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2) # the style loss for one layer, weighted appropriately
				style_loss += layer_style_loss / (d * h * w) # add to the style loss
			total_loss = content_weight * content_loss + style_weight * style_loss     # calculate the *total* loss

			# update your target image
			optimizer.zero_grad()
			total_loss.backward()
			optimizer.step()

			# display intermediate images and print the loss
			if  ii % show_every == 0:
				print('Total loss: ', total_loss.item())
				plt.imshow(im_convert(target))
				plt.show()
			ii += 1
		plt.imsave(os.path.join(app.config['UPLOAD_FOLDER'], 'result.png'), im_convert(target))
	return Response(progress(), mimetype='text/event-stream')

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=9201, threaded=True)

