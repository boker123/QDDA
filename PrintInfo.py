import torch
from thop import profile
import os
import traceback
from torchinfo import summary


from models.QDDA import QDDANet


def print_model_info(model, input_size=(1, 3, 224, 224)):
    """
    打印模型的详细信息

    Args:
        model: PyTorch模型
        input_size: 输入张量的尺寸 (batch_size, channels, height, width)
    """
    print("=" * 80)
    print("QDDANet 模型信息")
    print("=" * 80)

    # 1. 打印模型结构
    print("\n1. 模型结构:")
    print("-" * 50)
    print(model)

    print("\n1.1 模型摘要信息 (summary):")
    print("-" * 50)
    try:
        summary(model, input_data=[
            torch.randn(input_size),
            torch.randn(input_size),
            torch.randn(input_size),
            torch.randn(input_size)
        ], col_names=["input_size", "output_size", "num_params", "params_percent", "kernel_size"], depth=3)
    except Exception as e:
        print(f"summary 打印失败: {e}")

    # 2. 打印模型参数数量
    print("\n2. 模型参数统计:")
    print("-" * 50)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    print(f"不可训练参数数量: {non_trainable_params:,}")
    print(f"模型大小 (MB): {total_params * 4 / (1024 * 1024):.2f}")

    # 3. 按层打印参数
    print("\n3. 各层参数详情:")
    print("-" * 50)
    for name, param in model.named_parameters():
        print(
            f"{name:40s} | Shape: {str(param.shape):20s} | Params: {param.numel():>8,} | Trainable: {param.requires_grad}")

    # 4. 按模块打印参数统计
    print("\n4. 各模块参数统计:")
    print("-" * 50)
    module_params = {}
    for name, param in model.named_parameters():
        module_name = name.split('.')[0] if '.' in name else name
        if module_name not in module_params:
            module_params[module_name] = 0
        module_params[module_name] += param.numel()

    for module_name, param_count in sorted(module_params.items(), key=lambda x: x[1], reverse=True):
        percentage = (param_count / total_params) * 100
        print(f"{module_name:20s} | Params: {param_count:>10,} | Percentage: {percentage:>6.2f}%")

    # 5. 计算FLOPs (如果thop可用)
    print("\n5. 计算复杂度分析:")
    print("-" * 50)
    try:
        # 创建四个输入张量 (anchor, positive, negative, negative2)
        dummy_input = (
            torch.randn(input_size),
            torch.randn(input_size),
            torch.randn(input_size),
            torch.randn(input_size)
        )

        flops, params = profile(model, inputs=dummy_input, verbose=False)
        print(f"FLOPs: {flops:,}")
        print(f"参数数量 (通过thop): {params:,}")
        print(f"FLOPs (GFLOPs): {flops / 1e9:.2f}")
    except Exception as e:
        print(f"无法计算FLOPs: {e}")
        print("可能需要调整输入格式或安装thop库")

    # 6. 内存使用估算
    print("\n6. 内存使用估算:")
    print("-" * 50)

    # 模型参数内存
    param_memory = total_params * 4 / (1024 * 1024)  # float32, 4 bytes per parameter

    # 输入数据内存 (4个输入)
    input_memory = 4 * input_size[0] * input_size[1] * input_size[2] * input_size[3] * 4 / (1024 * 1024)

    print(f"模型参数内存: {param_memory:.2f} MB")
    print(f"输入数据内存: {input_memory:.2f} MB")
    print(f"估算总内存需求: {param_memory + input_memory:.2f} MB")


def test_model_forward(model, input_size=(2, 3, 224, 224)):
    """
    测试模型前向传播

    Args:
        model: PyTorch模型
        input_size: 输入张量的尺寸
    """
    print("\n7. 模型前向传播测试:")
    print("-" * 50)

    try:
        model.eval()
        with torch.no_grad():
            # 创建随机输入
            x_anchor = torch.randn(input_size)
            x_positive = torch.randn(input_size)
            x_negative = torch.randn(input_size)
            x_negative2 = torch.randn(input_size)

            print(f"输入尺寸:")
            print(f"  x_anchor: {x_anchor.shape}")
            print(f"  x_positive: {x_positive.shape}")
            print(f"  x_negative: {x_negative.shape}")
            print(f"  x_negative2: {x_negative2.shape}")

            # 前向传播
            outputs = model(x_anchor, x_positive, x_negative, x_negative2)

            print(f"\n输出尺寸:")
            for i, output in enumerate(outputs):
                print(f"  output_{i}: {output}")

            print(f"\n前向传播成功！")

    except Exception as e:
        print(f"前向传播失败: {e}")
        traceback.print_exc()  # 打印完整堆栈信息


def main():
    """
    主函数
    """
    try:
        # 初始化模型
        print("正在初始化QDDANet模型...")
        model = QDDANet(
            num_class=7,
            num_head=2,
            embed_dim=786,
            pretrained=False  # 设置为False以避免加载预训练权重的问题
        )

        # 打印模型信息
        print_model_info(model, input_size=(1, 3, 112, 112))

        # 测试前向传播
        test_model_forward(model, input_size=(2, 3, 112, 112))

        # 保存模型信息到文件
        print("\n8. 保存模型信息:")
        print("-" * 50)

        # 保存模型结构
        with open('qddanet_structure.txt', 'w', encoding='utf-8') as f:
            f.write(str(model))
        print("模型结构已保存到: qddanet_structure.txt")

        # 保存参数信息
        with open('qddanet_parameters.txt', 'w', encoding='utf-8') as f:
            f.write("QDDANet 参数信息\n")
            f.write("=" * 50 + "\n")

            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            f.write(f"总参数数量: {total_params:,}\n")
            f.write(f"可训练参数数量: {trainable_params:,}\n")
            f.write(f"模型大小: {total_params * 4 / (1024 * 1024):.2f} MB\n\n")

            f.write("各层参数详情:\n")
            f.write("-" * 50 + "\n")

            for name, param in model.named_parameters():
                f.write(f"{name:40s} | Shape: {str(param.shape):20s} | Params: {param.numel():>8,}\n")

        print("参数信息已保存到: qddanet_parameters.txt")

    except Exception as e:
        print(f"运行出错: {e}")
        print("请确保已正确导入所有依赖模块")


if __name__ == "__main__":
    main()