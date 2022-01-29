FROM python:3.9.10-bullseye

RUN apt-get update && apt-get install -y sudo

COPY onnx_tf_frontend.py /scripts/onnx_tf_frontend.py

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

RUN mkdir /workspace
WORKDIR /workspace

RUN groupadd developer && \
    useradd developer -m -g developer && \
    chown -R developer /workspace && \
    echo "developer ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER developer
RUN pip install -U pip && pip install tensorflow onnx-tf

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD []
