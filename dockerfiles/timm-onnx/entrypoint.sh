#! /bin/bash

sudo usermod -u ${UID} developer
sudo groupmod -g ${GID} developer

python /scripts/timm_onnx.py "$@"
