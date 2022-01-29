#! /bin/bash

SCRIPT_DIR=$(dirname $0)
docker build ${SCRIPT_DIR}/timm-onnx -t timm-onnx
docker build ${SCRIPT_DIR}/onnx-tf -t onnx-tf
