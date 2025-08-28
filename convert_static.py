import onnx

# 加载 ONNX 模型
model_path = 'visual.onnx'
model = onnx.load(model_path)

# 获取模型的图
graph = model.graph

# 遍历图中的所有输入节点
for input in graph.input:
    # 检查输入是否是模型的主要输入（通常第一个输入是模型的主要输入）
    if input.name == 'input':
        # 获取输入的维度信息
        shape = input.type.tensor_type.shape
        # 修改第一个维度（batch size）为 1
        shape.dim[0].dim_value = 1

# 保存修改后的模型
onnx.save(model, 'visual_static.onnx')