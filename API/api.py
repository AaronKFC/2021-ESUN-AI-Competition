from argparse import ArgumentParser
import base64
import datetime
import hashlib
import time

from model import Model
from io import BytesIO
from PIL import Image
from flask import Flask
from flask import request
from flask import jsonify


app = Flask(__name__)
model = Model()
model.initialize(
    model_name='efficientnet-b3',
    model_path='./models/efficientnet-b3_with_additional.pth')

####### PUT YOUR INFORMATION HERE #######
CAPTAIN_EMAIL = 'aiallen.cs07g@nctu.edu.tw'
SALT = 'baseline'
#########################################


def generate_server_uuid(input_string):
    """ Create your own server_uuid.

    @param:
        input_string (str): information to be encoded as server_uuid
    @returns:
        server_uuid (str): your unique server_uuid
    """
    s = hashlib.sha256()
    data = (input_string + SALT).encode("utf-8")
    s.update(data)
    server_uuid = s.hexdigest()
    return server_uuid


def base64_to_PIL(image_64_encoded):
    image_bytes = base64.b64decode(image_64_encoded)
    image_data = BytesIO(image_bytes)
    image = Image.open(image_data)
    return image

@app.route('/inference', methods=['POST'])
def inference():
    """ API that return your model predictions when E.SUN calls this API. """
    data = request.get_json(force=True)

    # 自行取用，可紀錄玉山呼叫的 timestamp
    esun_timestamp = data['esun_timestamp']

    image_64_encoded = data['image']
    image = base64_to_PIL(image_64_encoded)

    t = datetime.datetime.now()
    ts = str(int(t.utcnow().timestamp()))
    server_uuid = generate_server_uuid(CAPTAIN_EMAIL + ts)

    try:
        answer = model.predict(image)
    except TypeError as type_error:
        # You can write some log...
        raise type_error
    except Exception as e:
        # You can write some log...
        raise e
    server_timestamp = time.time()

    return jsonify({'esun_uuid': data['esun_uuid'],
                    'server_uuid': server_uuid,
                    'answer': answer,
                    'server_timestamp': server_timestamp})


if __name__ == "__main__":
    arg_parser = ArgumentParser(
        usage='Usage: python ' + __file__ + ' [--port <port>] [--help]'
    )
    arg_parser.add_argument('-p', '--port', default=5000, help='port')
    arg_parser.add_argument('-d', '--debug', default=False, help='debug')
    options = arg_parser.parse_args()

    app.run(debug=options.debug, port=options.port, host='0.0.0.0')
