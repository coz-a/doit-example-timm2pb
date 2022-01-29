# doit-example-timm2pb

Example of [doit](https://pydoit.org/) pipeline with docker containers.

This pipeline prepare [timm](https://github.com/rwightman/pytorch-image-models)'s pretrained model with given name, and convert into tensorflow frozen graph format (.pb).

## Usage

``` sh
$ doit model-name=resnet18
```

then, these files will be output.
```
compiled
└── resnet18
    ├── metadata.json
    ├── model.onnx
    └── model.pb
```
