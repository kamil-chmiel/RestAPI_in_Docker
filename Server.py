import os
from flask import Flask, render_template, request, jsonify
import Tools

app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(f)

    result = Tools.network_predict(file)

    label = [result[0][0], result[0][1], result[0][2]]  # three most probable results
    print(label)  # show results format

    results = [
        {
            'result': label[0][1],
            'probability': str(label[0][2])
        },
        {
            'result': label[1][1],
            'probability': str(label[1][2])
        },
        {
            'result': label[2][1],
            'probability': str(label[2][2])
        }
    ]

    return jsonify({'results': results})


@app.route('/train')
def fine_tune_network():

    result = Tools.set_model(request.args)
    return result


if __name__ == "__main__":
    print("Starting server...")
    Tools.load_model()
    app.run("0.0.0.0", port=80, debug=True)
