import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import sqlite3
from flask import g
import torch

from src import llm, embedding, init_app
import time

app = Flask(__name__, template_folder='./src/templates')
UPLOAD_FOLDER = app.root_path + '/uploaded_files'
ALLOWED_EXTENSIONS = {'txt', 'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
DATABASE = 'nextg-llm.db'
# LLM_MODEL = 'llama-2-13b-chat.Q6_K.gguf'
# LLM_MODEL = 'llama-2-7b-chat.Q4_K_M.gguf'
# LLM_MODEL_PATH = app.root_path + '/models/' + LLM_MODEL
VECTOR_DB_PATH = app.root_path + '/chromadb'



def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        cur = db.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS files(name, path, chroma_dir, embedding_model)")
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def insert_file(name, path, chroma_dir, embedding_model):
    db = get_db()
    db_cursor = db.cursor()
    db_cursor.execute(f"insert into files values ('{name}', '{path}', '{chroma_dir}', '{embedding_model}')")
    db.commit()

def get_files():
    db_cursor = get_db().cursor()
    files = db_cursor.execute("select * from files limit 100").fetchall()
    return files

def get_file(name):
    db_cursor = get_db().cursor()
    file = db_cursor.execute(f"select * from files where name = '{name}'").fetchone()
    return file

@app.route('/files', methods=['GET', 'POST'])
def upload_file():

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            chroma_dir = embedding.document_embedding(filepath)
            insert_file(file.filename, filename, chroma_dir, embedding.model_name)
            return redirect(url_for('upload_file'))
    else:
        files = get_files()
        return render_template('upload.html', files=files)

@app.route('/qna', methods=['GET', 'POST'])
def question_answering():
    if request.method == 'POST':
        question = request.form.get('question')
        files = get_files()
        target_file = get_file(request.form.get('file'))
        answer = llm.qna(target_file, question)
        return render_template("qna.html", answer=answer, files=files)
    else:
        files = get_files()
        return render_template("qna.html", files=files, device=app.config['DEVICE'])


if __name__ == '__main__':
    app.config['DATABASE'] = DATABASE
    app.config['LLM_MODEL'] = os.getenv('LLM_MODEL')
    app.config['LLM_MODEL_PATH'] = app.root_path + '/models/' + app.config['LLM_MODEL']
    app.config['VECTOR_DB_PATH'] = VECTOR_DB_PATH
    if torch.cuda.is_available():
        app.config['DEVICE'] = torch.device('cuda')
    elif torch.backends.mps.is_available():
        app.config['DEVICE'] = torch.device('mps')
    else:
        app.config['DEVICE'] = torch.device('cpu')

    init_app(app)

    app.run(host="0.0.0.0")
    # app.run(host="0.0.0.0", port=8000)

