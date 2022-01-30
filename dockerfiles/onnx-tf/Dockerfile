FROM python:3.9.10-bullseye

RUN apt-get update && apt-get install -y sudo

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
COPY onnx_tf_frontend.py /scripts/onnx_tf_frontend.py
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /workspace

RUN pip install -U pip && pip install tensorflow onnx-tf

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
