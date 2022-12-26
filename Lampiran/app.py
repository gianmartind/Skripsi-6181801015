from os import listdir
from os.path import join
from werkzeug.utils import secure_filename
from flask import jsonify
from routes.detect_image import detect_image
import json

#--Methods--
def listmodels():
    models_list = [m for m in listdir('./static/models')]
    response = jsonify(models_list)
    return response

def save_image(file):
    fname = file.filename
    save_dir = join(_app.config['UPLOAD_FOLDER'], secure_filename(fname))
    file.save(save_dir)
    return save_dir

def detect():
    global _req
    global _app
    img_dir = save_image(_req.files['file'])
    param = json.loads(_req.form['param'])

    response = jsonify(detect_image(param['model'], float(param['consistency']), float(param['uniqueness']), img_dir))

    return response

#--Routes--
routes = {
    'models': listmodels,
    'detect': detect
}

#--app & request object--
_app = None
_req = None

def app_routes(url, app, req):
    global routes
    global _req
    global _app

    _req = req
    _app = app

    return routes[url]()