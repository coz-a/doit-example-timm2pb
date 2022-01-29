#! /bin/bash

sudo usermod -u ${UID} developer
sudo groupmod -g ${GID} developer

python /scripts/onnx_tf_frontend.py "$@"
