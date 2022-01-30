FROM python:3.9.10-bullseye

RUN apt-get update && apt-get install -y sudo

COPY onnx_tf_frontend.py /scripts/onnx_tf_frontend.py

WORKDIR /workspace

RUN pip install -U pip && pip install tensorflow onnx-tf

CMD ["python", "/scripts/onnx_tf_frontend.py"]
