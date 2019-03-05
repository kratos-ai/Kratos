"""
This software is developed and distributed under the MIT open source license.

MCopyright 2019 [A. Kowalski, J. Le, R. Emory, S. Lambert, Y. Han, Y. Li, Z. Salah]
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
@author: Adam Kowalski
"""

# %% imports
import tensorflow as tf
from flask import Flask, request, jsonify
import pandas as pd


# %% enable eager execution
tf.enable_eager_execution()


# %% flags
class FLAGS:
    data_dir = 'deep-fashion/'
    height = 300
    width = 300
    upload_folder = ""


# %% data
category_cloth = pd.read_csv(
    f'{FLAGS.data_dir}/anno/list_category_cloth.txt',
    delim_whitespace=True, header=1)


# %% load image
def load_image(filename):
    image = tf.io.read_file(filename)
    return tf.image.decode_jpeg(image)


# %% preprocess image
def preprocess_image(image):
    image = tf.image.resize_image_with_crop_or_pad(
        image, FLAGS.height, FLAGS.width)
    image = tf.image.per_image_standardization(image)
    return tf.reshape(image, (1, FLAGS.height, FLAGS.width, 3))


# %% predict
def predict(filename):
    logits = model(preprocess_image(load_image(filename)))
    predictions = (tf.nn.top_k(logits, k=5, sorted=True, name=None)
        .indices
        .numpy()[0])
    predictions = category_cloth['category_name'][predictions - 1].values
    return {
        'name': 'Adam',
        'type': 'category',
        'prediction': list(predictions)
    }


# %% load model
model = tf.keras.models.load_model("checkpoints/model-03.hdf5")


# %% app
app = Flask(__name__)


# %% root route
@app.route("/")
def root_route():
    return "Connected!!!"


# %% predict
@app.route("/predict", methods=["POST"])
def predict_route():
    try:
        file = request.files['photo']
        filename = "UploadPhoto.jpg"
        file.save(os.path.join(FLAGS.upload_folder), filename)
        return jsonify(predict(filename))
    except Exception as e:
        return jsonify({
            'type': 'error',
            'message': 'Unable to predict...'
        })


# %% main
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, threaded=False)
