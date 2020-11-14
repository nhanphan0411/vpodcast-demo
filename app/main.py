#!/usr/bin/env python3
from flask import Flask, Blueprint, jsonify, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
import os
import matplotlib.pyplot as plt
import io
import librosa
from librosa import display

app = Flask(__name__)

cmaps = sorted(['Accent', 'Blues', 'BrBG', 'BuGn','BuPu','CMRmap','Dark2','GnBu', 'Greens','Greys', 'OrRd','Oranges','PRGn','Paired','Pastel1','Pastel2','PiYG','PuBu','PuBuGn','PuOr','PuRd','Purples','RdBu','RdGy','RdPu','RdYlBu','RdYlGn','Reds','Set1','Set2','Set3','Spectral','Wistia','YlGn','YlGnBu','YlOrBr','YlOrRd','afmhot','afmhot','autumn','binary','bone','brg','bwr','cividis','cool','coolwarm','copper','cubehelix','flag','gist_earth','gist_gray','gist_heat','gist_ncar','gist_rainbow','gist_stern','gist_yarg','gnuplot','gnuplot2','gray','hot','hsv','inferno','jet','magma','nipy_spectral','ocean','pink','plasma','prism','rainbow','seismic','spring','summer','tab10','tab20','tab20b','tab20c','terrain','twilight','twilight_shifted','viridis','winter'])

app.config['UPLOAD_FOLDER'] = os.getcwd() + '/static/audio/'
app.config['OUTPUT_FOLDER'] = os.path.join(
    app.config['UPLOAD_FOLDER'], 'output')


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('home_page.html', cmaps=cmaps)


@app.route('/upload', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    print(request.files)
    if 'file' not in request.files:
        print('No file part')
        return redirect(request.url)
    file = request.files['file']
    filename = file.filename
    filename = filename.replace(' ', '')
    style = request.form['cmaps']

    print('The style is', style)

    # if user does not select file, browser also
    # submit an empty part without filename
    if filename == '':
        print('No selected file')
        return redirect(request.url)
    if file:
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('predict',
                                filename=filename, style=style))
    return


def visualize_one(path, style, fig_width=15, fig_height=10, dpi=300):
    path = os.path.join(app.config['UPLOAD_FOLDER'], path)
    x, sr = librosa.load(path)
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))

    plt.figure(figsize=(fig_width, fig_height))
    # if style == 'random':
    #     i = np.random.randint(98)
    #     style = cmaps[i]
    #     print(f'Palette | {style}')
    display.specshow(Xdb, sr=sr, cmap=style)
    plt.axis('off')

    fname = path.split('.')[0]
    fname = os.path.join(app.config['UPLOAD_FOLDER'], fname + '_'+style)
    plt.savefig(fname=fname, dpi=dpi, transparent=True,
                bbox_inches='tight', pad_inches=0)
    print(f'Image is successfully saved at {fname}.png')


@app.route('/predict/<filename>/<style>', methods=['GET'])
def predict(filename, style):
    visualize_one(filename, style)
    fname = filename.split('.')[0]
    fimg = os.path.join('/static/audio', fname + '_' + style + '.png')
    fname = os.path.join('/static/audio', filename)

    return render_template('predict.html', fimg=fimg, fname=fname)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
