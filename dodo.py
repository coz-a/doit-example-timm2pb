from pathlib import Path
from doit import get_var

model_name = get_var("model-name", "mobilenetv2_100")
output_root = get_var("output-root", "compiled")

output_dir = Path(output_root) / model_name
output_dir.mkdir(parents=True, exist_ok=True)

def task_timm_onnx():
    tgt_model_file = str(output_dir / "model.onnx")
    tgt_metadata_file = str(output_dir / "metadata.json")
    return {
        "targets": [tgt_model_file, tgt_metadata_file],
        "actions": [
            f"docker-compose run --rm timm-onnx --model-name {model_name} --model-output {tgt_model_file} --metadata-output {tgt_metadata_file}"
        ],
        "uptodate": [True]
    }


def task_onnx_tf():
    dep_file = str(output_dir / "model.onnx")
    tgt_file = str(output_dir / "model.pb")
    return {
        "file_dep": [dep_file],
        "targets": [tgt_file],
        "actions": [
            f"docker-compose run --rm onnx-tf --input {dep_file} --output {tgt_file}"
        ]
    }
