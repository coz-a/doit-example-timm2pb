FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

RUN apt-get update && apt-get install -y sudo

COPY timm_onnx.py /scripts/timm_onnx.py

RUN pip install -U pip && pip install timm

CMD ["python", "/scripts/timm_onnx.py"]
