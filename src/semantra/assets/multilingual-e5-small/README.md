# multilingual-e5-small

This directory contains the quantized ONNX export of
[`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small),
distributed under the MIT License. The export is provided by the open-source
[`nixiesearch/multilingual-e5-small-onnx`](https://huggingface.co/nixiesearch/multilingual-e5-small-onnx)
repository and runs locally with ONNX Runtime.

Semantra adds the model-required `query:` and `passage:` prefixes internally.
The model supports approximately 100 languages through its XLM-R backbone;
quality can be lower for low-resource languages. No network request is made at
runtime.
