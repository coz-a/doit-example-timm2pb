from argparse import ArgumentParser
import json
import torch
from torch.onnx import export
import timm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model-name", default="mobilenetv3_large_100")
    parser.add_argument("--model-output", default="model.onnx")
    parser.add_argument("--metadata-output", default="metadata.json")
    return parser.parse_args()


def main(args):
    metadata = timm.data.resolve_data_config({}, model=args.model_name, verbose=True)
    with open(args.metadata_output, "w") as fp:
        json.dump(metadata, fp)
    model = timm.create_model(args.model_name, pretrained=True).eval()
    input_shape = (1,) + metadata["input_size"]
    dummy_input = torch.rand(input_shape, dtype=torch.float32)
    export(model, dummy_input, args.model_output)


if __name__ == "__main__":
    args = parse_args()
    main(args)
