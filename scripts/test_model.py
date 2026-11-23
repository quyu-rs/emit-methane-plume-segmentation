# -*- coding: utf-8 -*-
"""
测试DiT-Mamba模型
验证前向传播、参数量、FLOPs、推理速度等
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import time
import numpy as np


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.dit_mamba import DiTMamba

def count_parameters(model):
    """统计模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def test_forward_pass():
    """测试前向传播"""
    print("=" * 70)
    print("1. 测试前向传播")
    print("=" * 70)

    model = DiTMamba(
        in_channels=4,
        base_channels=64,
        num_encoder_layers=4,
        mamba_dim=256,
        num_classes=1,
        dropout=0.1
    )
    model.eval()


    batch_size = 2
    input_tensor = torch.randn(batch_size, 4, 512, 512)

    print(f"输入形状: {input_tensor.shape}")


    with torch.no_grad():
        output = model(input_tensor)

    print(f"输出形状: {output.shape}")
    print(f"预期形状: ({batch_size}, 1, 512, 512)")


    print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")


    has_nan = torch.isnan(output).any()
    has_inf = torch.isinf(output).any()
    print(f"包含NaN: {has_nan.item()}")
    print(f"包含Inf: {has_inf.item()}")


    has_nan_param, has_inf_param = model.check_parameters()
    print(f"参数包含NaN: {has_nan_param}")
    print(f"参数包含Inf: {has_inf_param}")

    assert output.shape == (batch_size, 1, 512, 512), f"输出形状不正确: {output.shape}"
    assert not has_nan.item(), "输出包含NaN"
    assert not has_inf.item(), "输出包含Inf"
    assert not has_nan_param, "参数包含NaN"
    assert not has_inf_param, "参数包含Inf"

    print("[PASS] 前向传播测试通过\n")
    return model

def test_parameters(model):
    """测试参数量"""
    print("=" * 70)
    print("2. 统计参数量")
    print("=" * 70)

    total_params, trainable_params = count_parameters(model)

    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"参数量 (M): {total_params / 1e6:.2f}M")
    print(f"目标: <=25M")

    assert total_params <= 25e6, f"参数量超过目标: {total_params / 1e6:.2f}M > 25M"
    print("[PASS] 参数量符合目标\n")

    return total_params

def test_flops(model):
    """测试FLOPs（简化计算）"""
    print("=" * 70)
    print("3. 计算FLOPs（简化估算）")
    print("=" * 70)

    try:
        from thop import profile, clever_format
        input_tensor = torch.randn(1, 4, 512, 512)
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        flops_g = flops / 1e9
        print(f"FLOPs: {flops:,}")
        print(f"FLOPs (G): {flops_g:.2f}G")
        print(f"目标: <=80G")

        assert flops <= 80e9, f"FLOPs超过目标: {flops_g:.2f}G > 80G"
        print("[PASS] FLOPs符合目标\n")
        return flops
    except ImportError:
        print("[SKIP] thop未安装，跳过FLOPs计算")
        print("安装命令: pip install thop\n")
        return None

def test_inference_speed(model, device='cpu'):
    """测试推理速度"""
    print("=" * 70)
    print(f"4. 测试推理速度 (设备: {device})")
    print("=" * 70)

    model = model.to(device)
    model.eval()

    input_tensor = torch.randn(1, 4, 512, 512).to(device)


    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)


    num_runs = 100
    times = []

    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(input_tensor)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            times.append(end - start)

    avg_time = np.mean(times)
    fps = 1.0 / avg_time

    print(f"平均推理时间: {avg_time*1000:.2f}ms")
    print(f"推理速度: {fps:.2f} FPS")
    print(f"目标: >=100 FPS")

    if device == 'cpu':
        print("[NOTE] CPU推理速度较慢，GPU上会更快")
    else:
        assert fps >= 100, f"推理速度低于目标: {fps:.2f} FPS < 100 FPS"
        print("[PASS] 推理速度符合目标\n")

    return fps

def test_memory_usage(model, device='cpu'):
    """测试训练时内存占用"""
    print("=" * 70)
    print(f"5. 测试训练时内存占用 (设备: {device}, batch_size=4)")
    print("=" * 70)

    model = model.to(device)
    model.train()

    batch_size = 4
    input_tensor = torch.randn(batch_size, 4, 512, 512).to(device)
    target = torch.randint(0, 2, (batch_size, 1, 512, 512)).float().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


    output = model(input_tensor)
    loss = criterion(output, target)


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if device == 'cuda':
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"GPU内存占用: {memory_mb:.2f} MB")
        print(f"GPU内存占用: {memory_mb / 1024:.2f} GB")
        print(f"目标: <=4GB")

        assert memory_mb <= 4096, f"内存占用超过目标: {memory_mb/1024:.2f}GB > 4GB"
        print("[PASS] 内存占用符合目标\n")
    else:
        print("[NOTE] CPU模式下无法准确测量GPU内存占用")
        print("请在GPU环境下测试\n")

    return memory_mb if device == 'cuda' else None

def main():
    """主测试函数"""
    print("\n" + "=" * 70)
    print("DiT-Mamba模型测试")
    print("=" * 70 + "\n")


    model = test_forward_pass()


    total_params = test_parameters(model)


    flops = test_flops(model)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fps = test_inference_speed(model, device=device)


    import torch.nn as nn
    memory = test_memory_usage(model, device=device)


    print("=" * 70)
    print("测试总结")
    print("=" * 70)
    print(f"参数量: {total_params / 1e6:.2f}M (目标: <=25M)")
    if flops:
        print(f"FLOPs: {flops / 1e9:.2f}G (目标: <=80G)")
    print(f"推理速度: {fps:.2f} FPS (目标: >=100 FPS)")
    if memory:
        print(f"内存占用: {memory / 1024:.2f}GB (目标: <=4GB)")
    print("=" * 70)
    print("所有测试完成！")
    print("=" * 70 + "\n")

if __name__ == '__main__':
    main()



