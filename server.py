from typing import Tuple, List, Text, Dict, Any, Iterator
import os
import sys
import time

import numpy as np
import cv2
import skimage.io as io
import tensorflow as tf
from flask import Flask, request, json, send_file

sys.path.append("./models/")
sys.path.append("./models/object_detection")

from utils import label_map_util
from utils import visualization_utils as vis_util

PATH_TO_CKPT = os.path.join('ssd_mobilenet_v1_coco_11_06_2017', 'frozen_inference_graph.pb') # type: str
PATH_TO_LABELS = os.path.join('models', 'object_detection', 'data', 'mscoco_label_map.pbtxt') # type: str
NUM_CLASSES = 90

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS) # type: Dict[str, List[Dict[str, str]]]
# label_map: { item: List<{id: str; name: str; display_name?: str; }>; }
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True) # type: List[Dict[str,str]]
# categories: List<{ id: str; name: str; }>
category_index = label_map_util.create_category_index(categories) # type: Dict[str, Dict[str,str]]
# category_index: List<{ [id: str]: {id: str; name: str; }; }>

def predict(img_expanded):
    # type: (np.ndarray)-> Tuple[np.ndarray, np.ndarray, np.ndarray]
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: img_expanded}
            )
    return (
        np.squeeze(boxes),
        np.squeeze(scores),
        np.squeeze(classes).astype(np.int32),
    )


app = Flask(__name__, static_url_path='')

@app.route('/')
def _root():
    return app.send_static_file('index.html')

@app.route('/png', methods=['POST'])
def _png():
    return _proc(request, True)

@app.route('/json', methods=['POST'])
def _json():
    return _proc(request, False)

def _proc(request, PNG):
    # type: (Request, bool)-> Response
    if not request.method == 'POST':
        return app.response_class(
            response=json.dumps({'message': "400 Bad Request. use POST"}),
            status=400,
            mimetype='application/json'
        )

    files = request.files.getlist("files")

    if len(files) == 0:
        # 何もすることがない
        return app.response_class(
            response=json.dumps([]),
            status=200,
            mimetype='application/json'
        )

    start = time.time()

    # 一旦 /tmp/* に保存
    filename = "/tmp/img.img"
    files[0].save(filename)

    img = io.imread(filename).astype(np.uint8)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    img_expanded = np.expand_dims(img, axis=0)

    # /tmp/* 削除
    os.remove(filename)

    print("imread　{0:8.4g} sec".format(time.time() - start))


    start = time.time()
    (boxes, scores, classes) = predict(img_expanded)
    print("predict {0:8.4g} sec".format(time.time() - start))

    if PNG:
        # return png
        vis_util.visualize_boxes_and_labels_on_image_array(
            img,
            boxes,
            classes,
            scores,
            category_index,
            min_score_thresh=.2,
            max_boxes_to_draw=None,
            use_normalized_coordinates=True,
            line_thickness=8
        )
        # 一旦 png で保存してから投げる
        filename = "/var/tmp/foo.png"
        io.imsave(filename, img)
        res = send_file(filename, mimetype='image/png')
        os.remove(filename)
        return res

    # return json
    result_list = []
    for i in range(boxes.shape[0]):
        box = tuple(boxes[i].tolist())
        class_name = category_index[classes[i]]['name'] if classes[i] in category_index.keys() else 'N/A'
        score = float(scores[i])
        ymin, xmin, ymax, xmax = box
        height, width, ch = img.shape
        (left, right) = (int(xmin * width), int(xmax * width))
        (top, bottom) = (int(ymin * height), int(ymax * height))
        result_list.append({
            "class_name": class_name,
            "box": (left, top, right, bottom),
            "score": score
        })

    return app.response_class(
        response=json.dumps(result_list),
        status=200,
        mimetype='application/json'
    )
