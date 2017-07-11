read this - https://github.com/tensorflow/models/blob/master/object_detection/g3doc/installation.md

```sh
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz
git clone https://github.com/tensorflow/models.git
pushd models
protoc object_detection/protos/*.proto --python_out=.
popd
```

run

```sh
gunicorn -w 4 -b 0.0.0.0:8888 server:app
```

or

```
env FLASK_APP=server.py flask run --host=0.0.0.0 --port 8888
```
