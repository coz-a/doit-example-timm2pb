import os
import shutil
from argparse import ArgumentParser
from tempfile import TemporaryDirectory
import onnx
from onnx_tf.backend import prepare


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input", default="model.onnx")
    parser.add_argument("--output", default="model.pb")
    return parser.parse_args()


def main(args):
    onnx_model = onnx.load(args.input)
    tf_rep = prepare(onnx_model)
    with TemporaryDirectory() as tempd:
        tf_rep.export_graph(tempd)
        pbfile = os.path.join(tempd, "saved_model.pb")
        shutil.move(pbfile, args.output)


if __name__ == "__main__":
    args = parse_args()
    main(args)
