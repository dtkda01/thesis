import json
import codecs
import os
from PIL import Image
from flask import Flask, flash, request, redirect, url_for, render_template, Response
from werkzeug.utils import secure_filename
from detect import detect_obj
from werkzeug.exceptions import BadRequest

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
@app.route('/')
def home():
    return render_template('index.html')

with codecs.open('./thanhphan.json', 'r', 'utf-8-sig') as json_file:
    thanhphan = json.load(json_file)

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('Không có hình nào được tải lên!')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        id_label = detect_obj(app.config['UPLOAD_FOLDER'], os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('Hình đã được tải lên')
        # print(os.path.join('detected/', filename))
        if id_label != None:
            materials = thanhphan[str(id_label)]
        else: 
            materials = ""
        return render_template('index.html', filename=filename, materials=materials)
    else:
        flash('File hình ảnh phải là dạng - png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    
    return redirect(url_for('static', filename='uploads/' + filename), code=301)



if __name__ == "__main__":
    app.run()
