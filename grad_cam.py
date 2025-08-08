from PIL import Image
import torchvision.transforms as transforms
import cv2
from models.QDDA import QDDANet
from run_raf_DDA import *
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms



class QDDANetGradCAM:
    def __init__(self, model, target_layer_name='backbone.4'):
        """
        专门为QDDANet设计的Grad-CAM实现

        Args:
            model: 训练好的QDDANet模型
            target_layer_name: 目标层名称，默认为backbone的最后一层
        """
        self.model = model
        self.model.eval()

        # 存储梯度和特征图
        self.gradients = None
        self.activations = None

        # 注册hook到目标层
        self.target_layer = self._get_target_layer(target_layer_name)
        self.forward_hook = self.target_layer.register_forward_hook(self._forward_hook)
        self.backward_hook = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _get_target_layer(self, layer_name):
        """根据层名称获取目标层"""
        for name, module in self.model.named_modules():
            if name == layer_name:
                return module

        # 如果没找到指定层，尝试找backbone的最后一个卷积层
        backbone_layers = []
        for name, module in self.model.backbone.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                backbone_layers.append((name, module))

        if backbone_layers:
            print(f"Using last conv layer in backbone: backbone.{backbone_layers[-1][0]}")
            return backbone_layers[-1][1]

        raise ValueError(f"Layer {layer_name} not found in model")

    def _forward_hook(self, module, input, output):
        """前向传播hook，保存激活值"""
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        """反向传播hook，保存梯度"""
        self.gradients = grad_output[0]

    def generate_cam_single(self, input_tensor, class_idx=None):
        """
        为单个输入生成Grad-CAM（推理模式）

        Args:
            input_tensor: 输入张量 (1, C, H, W)
            class_idx: 目标类别索引，如果为None则使用预测概率最高的类别

        Returns:
            cam: Grad-CAM热力图
            prediction: 模型预测结果
        """
        input_tensor.requires_grad_(True)

        # 使用QDDANet的推理模式（只传入anchor，其他设为None）
        cls_base_anchor, x_an_head = self.model(input_tensor, None, None, None)

        # 获取预测类别
        if class_idx is None:
            class_idx = cls_base_anchor.argmax(dim=1).item()

        # 反向传播
        self.model.zero_grad()
        class_score = cls_base_anchor[0, class_idx]
        class_score.backward(retain_graph=True)

        # 计算Grad-CAM
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)

        # 计算权重 (全局平均池化)
        weights = torch.mean(gradients, dim=(1, 2))  # (C,)

        # 加权求和
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)  # (H, W)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        # ReLU激活
        cam = F.relu(cam)

        # 归一化到0-1
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.detach().cpu().numpy(), cls_base_anchor.detach().cpu().numpy()

    def generate_cam_quadruplet(self, x_anchor, x_positive, x_negative, x_negative2,
                                target_input='anchor', class_idx=None):
        """
        为四元组输入生成Grad-CAM（训练模式的完整输入）

        Args:
            x_anchor: anchor输入
            x_positive: positive输入
            x_negative: negative输入
            x_negative2: negative2输入
            target_input: 要分析的目标输入 ('anchor', 'positive', 'negative', 'negative2')
            class_idx: 目标类别索引

        Returns:
            cam: 目标输入的Grad-CAM热力图
            predictions: 所有输出的预测结果
        """
        # 设置需要梯度的输入
        inputs = {
            'anchor': x_anchor,
            'positive': x_positive,
            'negative': x_negative,
            'negative2': x_negative2
        }

        for key, tensor in inputs.items():
            if tensor is not None:
                tensor.requires_grad_(True)

        # 前向传播
        outputs = self.model(x_anchor, x_positive, x_negative, x_negative2)
        cls_base_anchor, cls_positive, cls_negative, cls_negative2 = outputs[:4]

        # 选择目标输出和对应的激活
        target_outputs = {
            'anchor': cls_base_anchor,
            'positive': cls_positive,
            'negative': cls_negative,
            'negative2': cls_negative2
        }

        target_output = target_outputs[target_input]

        # 获取预测类别
        if class_idx is None:
            class_idx = target_output.argmax(dim=1).item()

        # 反向传播
        self.model.zero_grad()
        class_score = target_output[0, class_idx]
        class_score.backward(retain_graph=True)

        # 计算Grad-CAM
        if self.gradients is not None and self.activations is not None:
            gradients = self.gradients[0]  # (C, H, W)
            activations = self.activations[0]  # (C, H, W)

            # 计算权重
            weights = torch.mean(gradients, dim=(1, 2))  # (C,)

            # 加权求和
            cam = torch.zeros(activations.shape[1:], dtype=torch.float32)  # (H, W)
            for i, w in enumerate(weights):
                cam += w * activations[i, :, :]

            # ReLU激活和归一化
            cam = F.relu(cam)
            if cam.max() > 0:
                cam = cam / cam.max()
        else:
            cam = torch.zeros((7, 7))  # 默认尺寸，根据实际情况调整

        # 收集所有预测结果
        predictions = {
            'anchor': cls_base_anchor.detach().cpu().numpy(),
            'positive': cls_positive.detach().cpu().numpy() if cls_positive is not None else None,
            'negative': cls_negative.detach().cpu().numpy() if cls_negative is not None else None,
            'negative2': cls_negative2.detach().cpu().numpy() if cls_negative2 is not None else None
        }

        return cam.detach().cpu().numpy(), predictions

    def visualize_cam(self, input_image, cam, alpha=0.4, title="Grad-CAM"):
        """
        可视化Grad-CAM结果

        Args:
            input_image: 原始输入图像 (PIL Image 或 numpy array)
            cam: Grad-CAM热力图
            alpha: 叠加透明度
            title: 图像标题

        Returns:
            result: 叠加后的图像
        """
        # 转换输入图像格式
        if isinstance(input_image, Image.Image):
            input_image = np.array(input_image)
        elif torch.is_tensor(input_image):
            # 如果是tensor，转换为numpy并调整维度
            if input_image.dim() == 4:
                input_image = input_image.squeeze(0)
            if input_image.dim() == 3 and input_image.shape[0] == 3:
                input_image = input_image.permute(1, 2, 0)
            input_image = input_image.cpu().numpy()

            # 反归一化（假设使用了ImageNet的归一化）
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            input_image = input_image * std + mean
            input_image = np.clip(input_image, 0, 1)
            input_image = (input_image * 255).astype(np.uint8)

        # 获取原始图像尺寸
        h, w = input_image.shape[:2]

        # 将CAM调整到原始图像尺寸
        cam_resized = cv2.resize(cam, (w, h))

        # 创建热力图
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # 叠加热力图和原始图像
        if len(input_image.shape) == 3:
            result = heatmap * alpha + input_image * (1 - alpha)
        else:  # 灰度图像
            input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
            result = heatmap * alpha + input_image_rgb * (1 - alpha)

        return result.astype(np.uint8)

    def analyze_sample(self, input_image_path, class_idx=None, save_path=None):
        """
        分析单个样本的完整流程

        Args:
            input_image_path: 输入图像路径
            class_idx: 目标类别索引
            save_path: 保存结果的路径
        """
        # 图像预处理
        transform = transforms.Compose([
            transforms.Resize((112, 112)),  # 根据您的模型输入尺寸调整
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # 加载和预处理图像
        image = Image.open(input_image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)

        # 生成Grad-CAM
        cam, prediction = self.generate_cam_single(input_tensor, class_idx)

        # 可视化结果
        result_image = self.visualize_cam(image, cam)

        # 显示结果
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(cam, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')

        axes[2].imshow(result_image)
        axes[2].set_title('Grad-CAM Overlay')
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

        # 打印预测结果
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        print(f"Predicted class: {predicted_class}, Confidence: {confidence:.4f}")

        return cam, result_image, predicted_class, confidence

    def compare_quadruplet_inputs(self, x_anchor, x_positive, x_negative, x_negative2,
                                  class_idx=None, save_path=None):
        """
        比较四元组输入的Grad-CAM结果

        Args:
            x_anchor, x_positive, x_negative, x_negative2: 四个输入张量
            class_idx: 目标类别索引
            save_path: 保存结果的路径
        """
        input_names = ['anchor', 'positive', 'negative', 'negative2']
        input_tensors = [x_anchor, x_positive, x_negative, x_negative2]

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        for i, (name, tensor) in enumerate(zip(input_names, input_tensors)):
            if tensor is not None:
                # 生成CAM
                cam, predictions = self.generate_cam_quadruplet(
                    x_anchor, x_positive, x_negative, x_negative2,
                    target_input=name, class_idx=class_idx
                )

                # 可视化原始图像
                original_img = self._tensor_to_image(tensor)
                axes[0, i].imshow(original_img)
                axes[0, i].set_title(f'{name.capitalize()} Input')
                axes[0, i].axis('off')

                # 可视化Grad-CAM
                result_img = self.visualize_cam(tensor, cam)
                axes[1, i].imshow(result_img)
                axes[1, i].set_title(f'{name.capitalize()} Grad-CAM')
                axes[1, i].axis('off')

                # 打印预测结果
                pred = predictions[name]
                if pred is not None:
                    predicted_class = np.argmax(pred[0])
                    confidence = np.max(pred[0])
                    print(f"{name.capitalize()} - Predicted class: {predicted_class}, Confidence: {confidence:.4f}")
            else:
                axes[0, i].text(0.5, 0.5, 'No Input', ha='center', va='center')
                axes[0, i].axis('off')
                axes[1, i].text(0.5, 0.5, 'No CAM', ha='center', va='center')
                axes[1, i].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def _tensor_to_image(self, tensor):
        """将tensor转换为可显示的图像"""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        if tensor.shape[0] == 3:
            tensor = tensor.permute(1, 2, 0)

        # 反归一化
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = tensor.cpu().numpy() * std + mean
        image = np.clip(image, 0, 1)

        return image

    def __del__(self):
        """清理hook"""
        if hasattr(self, 'forward_hook'):
            self.forward_hook.remove()
        if hasattr(self, 'backward_hook'):
            self.backward_hook.remove()


# 使用示例
def demo_qddanet_gradcam():
    """
    QDDANet Grad-CAM演示
    """
    # 1. 加载训练好的模型
    model = QDDANet(num_class=7, pretrained=True)
    model.load_state_dict(torch.load('C:\\Users\\boker\\PycharmProjects\\QDDA\\checkpoint_raf_db\\[08-05]-[04-22]-checkout_only_params.pth'))
    model.eval()

    # 2. 创建Grad-CAM分析器
    gradcam = QDDANetGradCAM(model)

    # 3. 分析单个图像（推理模式）
    cam, result, pred_class, confidence = gradcam.analyze_sample('C:\\Users\\boker\\PycharmProjects\\QDDA\\datas\\RAF-DB\\basic\\test_0007_aligned.jpg', save_path='gradcam_result.png')

    # 4. 分析四元组输入（如果有完整的四元组数据）
    # gradcam.compare_quadruplet_inputs(x_anchor, x_positive, x_negative, x_negative2)

    print("请按照注释中的步骤使用QDDANet Grad-CAM分析器")


# 方案1: 使用现有的analyze_sample方法（已包含可视化）
def visualize_with_analyze_sample(gradcam, image_path, class_idx=None, save_path=None):
    """
    使用analyze_sample方法进行可视化（推荐方案）
    这个方法会自动显示三个子图：原图、热力图、叠加图
    """
    cam, result_image, pred_class, confidence = gradcam.analyze_sample(
        input_image_path=image_path,
        class_idx=class_idx,
        save_path=save_path
    )

    return cam, result_image, pred_class, confidence


# 方案2: 手动控制可视化过程
def manual_visualization(gradcam, image_path, class_idx=None, save_path=None):
    """
    手动控制可视化过程，可以自定义显示内容
    """
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 加载和预处理图像
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    # 生成Grad-CAM
    cam, prediction = gradcam.generate_cam_single(input_tensor, class_idx)

    # 可视化结果
    result_image = gradcam.visualize_cam(image, cam)

    # 自定义显示
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 原始图像
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image', fontsize=14)
    axes[0, 0].axis('off')

    # 热力图
    axes[0, 1].imshow(cam, cmap='jet')
    axes[0, 1].set_title('Grad-CAM Heatmap', fontsize=14)
    axes[0, 1].axis('off')

    # 叠加图
    axes[1, 0].imshow(result_image)
    axes[1, 0].set_title('Grad-CAM Overlay', fontsize=14)
    axes[1, 0].axis('off')

    # 预测结果柱状图
    predicted_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0])

    # 显示前5个最高概率的类别
    top5_indices = np.argsort(prediction[0])[-5:][::-1]
    top5_probs = prediction[0][top5_indices]

    axes[1, 1].bar(range(5), top5_probs)
    axes[1, 1].set_title('Top 5 Predictions', fontsize=14)
    axes[1, 1].set_xlabel('Class Index')
    axes[1, 1].set_ylabel('Probability')
    axes[1, 1].set_xticks(range(5))
    axes[1, 1].set_xticklabels([str(idx) for idx in top5_indices])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    print(f"Predicted class: {predicted_class}, Confidence: {confidence:.4f}")

    return cam, result_image, predicted_class, confidence


# 方案3: 交互式可视化
def interactive_visualization(gradcam, image_path, save_path=None):
    """
    交互式可视化，可以选择不同的类别进行分析
    """
    # 预处理
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    # 获取所有类别的预测
    cam_default, prediction = gradcam.generate_cam_single(input_tensor, None)

    # 创建子图
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # 显示原始图像
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image', fontsize=12)
    axes[0, 0].axis('off')

    # 显示默认预测的CAM
    result_default = gradcam.visualize_cam(image, cam_default)
    axes[0, 1].imshow(result_default)
    pred_class = np.argmax(prediction[0])
    axes[0, 1].set_title(f'Predicted Class {pred_class}', fontsize=12)
    axes[0, 1].axis('off')

    # 显示前6个类别的CAM
    top6_classes = np.argsort(prediction[0])[-6:][::-1]

    for i, class_idx in enumerate(top6_classes):
        if i >= 6:
            break

        cam_class, _ = gradcam.generate_cam_single(input_tensor, class_idx)
        result_class = gradcam.visualize_cam(image, cam_class)

        row = (i + 2) // 4
        col = (i + 2) % 4

        if row < 2 and col < 4:
            axes[row, col].imshow(result_class)
            prob = prediction[0][class_idx]
            axes[row, col].set_title(f'Class {class_idx}: {prob:.3f}', fontsize=12)
            axes[row, col].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    return top6_classes, prediction


# 方案4: 单独显示热力图
def show_heatmap_only(gradcam, image_path, class_idx=None, colormap='jet'):
    """
    只显示热力图，不叠加原图
    """
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    cam, prediction = gradcam.generate_cam_single(input_tensor, class_idx)

    plt.figure(figsize=(8, 6))
    plt.imshow(cam, cmap=colormap)
    plt.colorbar()
    plt.title('Grad-CAM Heatmap')
    plt.axis('off')
    plt.show()

    return cam


# 方案5: 保存结果而不显示
def save_visualization(gradcam, image_path, save_dir, class_idx=None):
    """
    保存可视化结果到文件，不在屏幕上显示
    """
    import os

    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    cam, prediction = gradcam.generate_cam_single(input_tensor, class_idx)
    result_image = gradcam.visualize_cam(image, cam)

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 保存原图
    image.save(os.path.join(save_dir, 'original.png'))

    # 保存热力图
    plt.figure(figsize=(8, 6))
    plt.imshow(cam, cmap='jet')
    plt.colorbar()
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, 'heatmap.png'), bbox_inches='tight', dpi=300)
    plt.close()

    # 保存叠加图
    Image.fromarray(result_image).save(os.path.join(save_dir, 'overlay.png'))

    # 保存预测结果
    pred_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0])

    with open(os.path.join(save_dir, 'prediction.txt'), 'w') as f:
        f.write(f"Predicted class: {pred_class}\n")
        f.write(f"Confidence: {confidence:.4f}\n")
        f.write(f"All probabilities: {prediction[0]}\n")

    return cam, result_image, pred_class, confidence


# 使用示例
def main():
    """
    使用示例
    """
    # 假设你已经创建了gradcam对象
    model = QDDANet(num_class=7, pretrained=True)
    model.load_state_dict(torch.load('C:\\Users\\boker\\PycharmProjects\\QDDA\\checkpoint_raf_db\\[08-05]-[04-22]-checkout_only_params.pth'))
    model.eval()

    gradcam = QDDANetGradCAM(model)

    image_path = 'C:\\Users\\boker\\PycharmProjects\\QDDA\\datas\\RAF-DB\\basic\\test_0003_aligned.jpg'

    print("选择可视化方案:")
    print("1. 使用analyze_sample方法（推荐）")
    print("2. 手动控制可视化")
    print("3. 交互式可视化")
    print("4. 只显示热力图")
    print("5. 保存结果到文件")

    # 方案1: 最简单的使用方法
    # cam, result, pred_class, confidence = visualize_with_analyze_sample(
    #     gradcam, image_path, save_path='gradcam_result.png'
    # )

    # 方案2: 自定义显示
    cam, result, pred_class, confidence = manual_visualization(
        gradcam, image_path, save_path='manual_result.png'
    )

    # 方案3: 交互式
    # top_classes, prediction = interactive_visualization(
    #     gradcam, image_path, save_path='interactive_result.png'
    # )

    # 方案4: 只看热力图
    # cam = show_heatmap_only(gradcam, image_path)

    # 方案5: 保存到文件
    # cam, result, pred_class, confidence = save_visualization(
    #     gradcam, image_path, 'output_dir'
    # )


if __name__ == "__main__":
    main()