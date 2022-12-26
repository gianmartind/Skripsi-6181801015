#--Libraries
from flask import Flask, request, jsonify, request
from flask_cors import CORS
import os

#--Routing Files--
from routes.app import app_routes

#--Flask init--
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads'
cors = CORS(app)

#--Routes--
@app.route('/app/<url>', methods=['GET', 'POST'])
def apps(url):
    return app_routes(url, app, request)

if __name__ == '__main__':
    app.run(host='localhost', port=1113)


