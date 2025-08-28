from rknn.api import RKNN

rknn  = RKNN()
rknn.config(target_platform='rk3588')
rknn.load_onnx(model="visual.onnx")
rknn.build(do_quantization=False)
rknn.export_rknn("visual.rknn")