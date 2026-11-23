# -*- coding: utf-8 -*-
"""
主训练脚本
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import logging
import argparse
from datetime import datetime


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.dit_mamba import DiTMamba, DiT
from datasets.rows_dataset import RowsDataset
from datasets.dataset_adapter import DatasetAdapter
from datasets.transforms import RandomAugment, Rotate180, HorizontalFlip, VerticalFlip
from losses.dice_focal_loss import DiceFocalLoss
from training.trainer import Trainer
from training.config import TrainingConfig

def create_data_loaders(config: dict, device: torch.device):
    """创建数据加载器"""

    data_config = config.get('data', {})
    if data_config is None or not isinstance(data_config, dict):
        data_config = {}


    use_dataset_adapter = data_config.get('use_dataset_adapter', False)
    dataset_class = DatasetAdapter if use_dataset_adapter else RowsDataset


    if use_dataset_adapter:
        base_path = str(Path(__file__).parent.parent / 'data/dataset')
        channels = 0

        project_root = Path(__file__).parent.parent


        train_split_adapter_path = project_root / 'datasets/dataset_adapter_train_split.json'
        val_split_adapter_path = project_root / 'datasets/dataset_adapter_val_split.json'


        train_split_path = project_root / 'datasets/train_split.json'
        val_split_path = project_root / 'datasets/val_split.json'

        train_samples = None
        val_samples = None


        if train_split_adapter_path.exists():
            try:
                with open(train_split_adapter_path, 'r', encoding='utf-8') as f:
                    train_samples = json.load(f)
                print(f"使用DatasetAdapter格式的划分文件: {train_split_adapter_path}")
            except Exception as e:
                print(f"警告: 无法加载DatasetAdapter划分文件 {train_split_adapter_path}: {e}")
                train_samples = None
        elif train_split_path.exists():
            try:
                with open(train_split_path, 'r', encoding='utf-8') as f:
                    train_samples = json.load(f)
                print(f"使用RowsDataset格式的划分文件: {train_split_path} (注意: 可能不兼容)")
            except Exception as e:
                print(f"警告: 无法加载训练集划分文件: {e}，将使用全部数据")
                train_samples = None
        else:
            print("未找到训练集划分文件，将使用全部数据")


        if val_split_adapter_path.exists():
            try:
                with open(val_split_adapter_path, 'r', encoding='utf-8') as f:
                    val_samples = json.load(f)
                print(f"使用DatasetAdapter格式的划分文件: {val_split_adapter_path}")
            except Exception as e:
                print(f"警告: 无法加载DatasetAdapter划分文件 {val_split_adapter_path}: {e}")
                val_samples = None
        elif val_split_path.exists():
            try:
                with open(val_split_path, 'r', encoding='utf-8') as f:
                    val_samples = json.load(f)
                print(f"使用RowsDataset格式的划分文件: {val_split_path} (注意: 可能不兼容)")
            except Exception as e:
                print(f"警告: 无法加载验证集划分文件: {e}，将使用全部数据")
                val_samples = None
        else:
            print("未找到验证集划分文件，将使用全部数据")
    else:

        project_root = Path(__file__).parent.parent
        with open(project_root / 'datasets/train_split.json', 'r', encoding='utf-8') as f:
            train_samples = json.load(f)
        with open(project_root / 'datasets/val_split.json', 'r', encoding='utf-8') as f:
            val_samples = json.load(f)

        base_path = str(project_root / 'data/rows')
        channels = [0, 1, 2]


    augment = data_config.get('augment', True)
    if augment is None:
        augment = True
    if augment:
        train_transform = RandomAugment([
            Rotate180(),
            HorizontalFlip(),
            VerticalFlip()
        ])
    else:
        train_transform = None


    cache_enabled = data_config.get('cache_enabled', False)
    if cache_enabled:
        print("警告: 数据缓存已启用，将占用大量内存（约17GB）")


    train_dataset_kwargs = {
        'base_path': base_path,
        'channels': channels,
        'mask_mode': 'enhanced',
        'transform': train_transform,
        'sample_list': train_samples
    }

    if use_dataset_adapter:
        train_dataset_kwargs['cache_enabled'] = False

    train_dataset = dataset_class(**train_dataset_kwargs)


    val_dataset_kwargs = {
        'base_path': base_path,
        'channels': channels,
        'mask_mode': 'enhanced',
        'transform': None,
        'sample_list': val_samples
    }

    if use_dataset_adapter:
        val_dataset_kwargs['cache_enabled'] = cache_enabled

    val_dataset = dataset_class(**val_dataset_kwargs)


    is_cuda = device.type == 'cuda'

    pin_memory = data_config.get('pin_memory', True)
    if pin_memory is None:
        pin_memory = True
    pin_memory = pin_memory and is_cuda


    import platform
    is_windows = platform.system() == 'Windows'


    num_workers_config = data_config.get('num_workers', 4)
    if num_workers_config is None:
        num_workers_config = 0


    try:
        num_workers_config_int = int(num_workers_config) if num_workers_config is not None else 0
    except (ValueError, TypeError):
        num_workers_config_int = 0

    if is_windows:
        num_workers = min(max(num_workers_config_int, 0), 2)
    else:
        num_workers = max(num_workers_config_int, 0)


    base_config = config.get('base', {})
    if base_config is None or not isinstance(base_config, dict):
        base_config = {}

    batch_size = base_config.get('batch_size', 4)
    if batch_size is None:
        batch_size = 4


    try:
        batch_size = int(batch_size) if batch_size is not None else 4
    except (ValueError, TypeError) as e:
        print(f"警告: batch_size类型转换失败: {e}")
        batch_size = 4

    if batch_size <= 0:
        batch_size = 4








    assert batch_size is not None, "batch_size 不能为 None"
    assert num_workers is not None, "num_workers 不能为 None"
    assert isinstance(batch_size, (int, float)), f"batch_size 必须是数字类型，当前类型: {type(batch_size)}, 值: {batch_size}"
    assert isinstance(num_workers, (int, float)), f"num_workers 必须是数字类型，当前类型: {type(num_workers)}, 值: {num_workers}"


    batch_size = int(batch_size)


    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': bool(pin_memory)
    }



    if num_workers > 0:
        loader_kwargs['timeout'] = 60


    if num_workers > 0:
        prefetch_factor = data_config.get('prefetch_factor', 2)
        if prefetch_factor is None:
            prefetch_factor = 2
        try:
            prefetch_factor = int(prefetch_factor) if prefetch_factor is not None else 2
        except (ValueError, TypeError):
            prefetch_factor = 2
        loader_kwargs['prefetch_factor'] = prefetch_factor


        persistent_workers = data_config.get('persistent_workers', True)
        if persistent_workers is None:
            persistent_workers = True
        loader_kwargs['persistent_workers'] = bool(persistent_workers)


    print(f"DEBUG: DataLoader 参数: batch_size={batch_size} (type: {type(batch_size)}), "
          f"num_workers={num_workers} (type: {type(num_workers)}), "
          f"pin_memory={pin_memory} (type: {type(pin_memory)}), "
          f"timeout={loader_kwargs.get('timeout')}, "
          f"prefetch_factor={loader_kwargs.get('prefetch_factor', 'N/A')}")


    try:
        train_len = len(train_dataset)
        val_len = len(val_dataset)
        print(f"DEBUG: 训练集长度: {train_len} (type: {type(train_len)})")
        print(f"DEBUG: 验证集长度: {val_len} (type: {type(val_len)})")

        if train_len is None:
            raise ValueError("训练集长度不能为 None")
        if val_len is None:
            raise ValueError("验证集长度不能为 None")
        if not isinstance(train_len, (int, float)):
            raise ValueError(f"训练集长度必须是数字类型，当前类型: {type(train_len)}, 值: {train_len}")
        if not isinstance(val_len, (int, float)):
            raise ValueError(f"验证集长度必须是数字类型，当前类型: {type(val_len)}, 值: {val_len}")
    except Exception as e:
        print(f"错误: 检查数据集长度失败: {e}")
        import traceback
        traceback.print_exc()
        raise

    try:
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            **loader_kwargs
        )

        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            **loader_kwargs
        )
    except Exception as e:
        print(f"DEBUG: DataLoader 创建失败，错误类型: {type(e).__name__}")
        print(f"DEBUG: 错误消息: {str(e)}")
        import traceback
        traceback.print_exc()
        if "timeout" in str(e).lower() or num_workers > 0:
            print(f"警告: 多进程数据加载失败 ({e})，尝试单进程模式")
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=False
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False
            )
        else:
            raise

    return train_loader, val_loader

def load_class_weights_from_calculation(calculation_file='class_weights_calculation.json'):
    """
    从计算结果文件加载类别权重
    
    参数:
        calculation_file: 计算结果文件路径
    
    返回:
        class_weights: [背景权重, 羽流权重] 或 None
        focal_alpha: Focal Loss alpha 或 None
    """
    project_root = Path(__file__).parent.parent
    calc_path = project_root / calculation_file
    if calc_path.exists():
        try:
            with open(calc_path, 'r', encoding='utf-8') as f:
                calc_data = json.load(f)
            weights = calc_data.get('class_weights')
            alpha = calc_data.get('focal_alpha')
            if weights and alpha is not None:
                print(f"从 {calculation_file} 加载类别权重: {weights}")
                print(f"从 {calculation_file} 加载 Focal alpha: {alpha:.3f}")
                return weights, alpha
        except Exception as e:
            print(f"警告: 无法加载计算结果文件 {calculation_file}: {e}")
    return None, None

def main():
    """主函数"""

    project_root = Path(__file__).parent.parent
    log_dir = project_root / 'logs'
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_log_file = log_dir / f"train_script_{timestamp}.log"


    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)


    root_logger.handlers.clear()


    file_handler = logging.FileHandler(script_log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)


    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    print("=" * 70)
    print("DiT-Mamba 训练脚本")
    print("=" * 70)
    print(f"脚本日志文件: {script_log_file}")


    parser = argparse.ArgumentParser(description='DiT-Mamba 训练脚本')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='配置文件路径（相对于项目根目录），默认: config.yaml'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='ditmamba',
        choices=['dit', 'ditmamba'],
        help='模型类型: dit 或 ditmamba (默认: ditmamba)'
    )
    args = parser.parse_args()


    config_path = project_root / args.config
    if config_path.exists():
        print(f"从配置文件加载: {config_path}")
        config = TrainingConfig(str(config_path))
    else:
        print(f"警告: 配置文件不存在: {config_path}")

        default_config_path = project_root / 'config.yaml'
        if default_config_path.exists():
            print(f"使用默认配置文件: {default_config_path}")
            config = TrainingConfig(str(default_config_path))
        else:
            print("使用默认配置（未找到任何配置文件）")
            config = TrainingConfig()



    base_config = config.get('base', {})
    if base_config is None or not isinstance(base_config, dict):
        base_config = {}

    batch_size = base_config.get('batch_size', 4)
    if batch_size is None:
        batch_size = 4

    total_epochs = base_config.get('total_epochs', 150)
    if total_epochs is None:
        total_epochs = 150

    gradient_accumulation_steps = base_config.get('gradient_accumulation_steps', 2)
    if gradient_accumulation_steps is None:
        gradient_accumulation_steps = 2

    if batch_size <= 0:
        print(f"错误: batch_size 必须大于0，当前值: {batch_size}")
        return
    if total_epochs <= 0:
        print(f"错误: total_epochs 必须大于0，当前值: {total_epochs}")
        return
    if gradient_accumulation_steps <= 0:
        print(f"错误: gradient_accumulation_steps 必须大于0，当前值: {gradient_accumulation_steps}")
        return


    calc_weights, calc_alpha = load_class_weights_from_calculation()
    if calc_weights and calc_alpha is not None:

        config.config['loss']['class_weights'] = calc_weights
        config.config['loss']['focal_alpha'] = calc_alpha
        print("已使用计算出的类别权重和Focal alpha")
    else:
        print("使用配置文件中的默认类别权重")


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")


    print("\n创建模型...")

    model_config = config.get('model', {})
    base_channels = model_config.get('base_channels', 64)
    num_encoder_layers = model_config.get('num_encoder_layers', 4)
    mamba_dim = model_config.get('mamba_dim', 256)

    print(f"模型类型: {args.model_type}")
    print(f"模型架构配置:")
    print(f"  基础通道数: {base_channels}")
    print(f"  编码器层数: {num_encoder_layers}")
    if args.model_type == 'ditmamba':
        print(f"  Mamba维度: {mamba_dim}")


    if args.model_type == 'dit':
        model = DiT(
            in_channels=4,
            base_channels=base_channels,
            num_encoder_layers=num_encoder_layers,
            num_classes=1,
            dropout=config.get('regularization', {}).get('dropout', 0.1)
        )
    else:
        model = DiTMamba(
            in_channels=4,
            base_channels=base_channels,
            num_encoder_layers=num_encoder_layers,
            mamba_dim=mamba_dim,
            num_classes=1,
            dropout=config.get('regularization', {}).get('dropout', 0.1)
        )


    has_nan, has_inf = model.check_parameters()
    if has_nan or has_inf:
        print("错误: 模型参数包含NaN/Inf，请检查模型初始化")
        return


    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params / 1e6:.2f}M")


    print("\n创建数据加载器...")
    try:
        train_loader, val_loader = create_data_loaders(config.config, device)
    except Exception as e:
        print(f"错误: 创建数据加载器失败: {e}")
        return


    if len(train_loader.dataset) == 0:
        print("错误: 训练集为空，无法开始训练")
        return
    if len(val_loader.dataset) == 0:
        print("警告: 验证集为空，将无法进行验证")

    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"验证集样本数: {len(val_loader.dataset)}")


    try:
        test_batch = next(iter(train_loader))
        print("数据加载器测试通过")
    except Exception as e:
        print(f"错误: 数据加载器测试失败: {e}")
        return




    base_lr = config.get('optimizer', {}).get('lr', 1e-4)
    param_groups = [
        {
            'params': [p for n, p in model.named_parameters() if 'mamba.A_log' in n],
            'lr': base_lr * 0.01,
            'weight_decay': config.get('optimizer', {}).get('weight_decay', 1e-4),
            'betas': (
                config.get('optimizer', {}).get('beta1', 0.9),
                config.get('optimizer', {}).get('beta2', 0.999)
            ),
            'eps': config.get('optimizer', {}).get('eps', 1e-8)
        },
        {
            'params': [p for n, p in model.named_parameters() if 'mamba.A_proj' in n],
            'lr': base_lr * 0.1,
            'weight_decay': config.get('optimizer', {}).get('weight_decay', 1e-4),
            'betas': (
                config.get('optimizer', {}).get('beta1', 0.9),
                config.get('optimizer', {}).get('beta2', 0.999)
            ),
            'eps': config.get('optimizer', {}).get('eps', 1e-8)
        },
        {
            'params': [p for n, p in model.named_parameters() if 'mamba.x_proj' in n],
            'lr': base_lr * 0.1,
            'weight_decay': config.get('optimizer', {}).get('weight_decay', 1e-4),
            'betas': (
                config.get('optimizer', {}).get('beta1', 0.9),
                config.get('optimizer', {}).get('beta2', 0.999)
            ),
            'eps': config.get('optimizer', {}).get('eps', 1e-8)
        },
        {
            'params': [p for n, p in model.named_parameters() if 'mamba.dt_proj' in n],
            'lr': base_lr * 0.1,
            'weight_decay': config.get('optimizer', {}).get('weight_decay', 1e-4),
            'betas': (
                config.get('optimizer', {}).get('beta1', 0.9),
                config.get('optimizer', {}).get('beta2', 0.999)
            ),
            'eps': config.get('optimizer', {}).get('eps', 1e-8)
        },
        {
            'params': [p for n, p in model.named_parameters() if not any(key in n for key in ['mamba.A_log', 'mamba.A_proj', 'mamba.x_proj', 'mamba.dt_proj'])],
            'lr': base_lr,
            'weight_decay': config.get('optimizer', {}).get('weight_decay', 1e-4),
            'betas': (
                config.get('optimizer', {}).get('beta1', 0.9),
                config.get('optimizer', {}).get('beta2', 0.999)
            ),
            'eps': config.get('optimizer', {}).get('eps', 1e-8)
        }
    ]
    optimizer = torch.optim.Adam(param_groups)


    class_weights = torch.tensor(
        config.get('loss', {}).get('class_weights', [0.5, 1.5]),
        dtype=torch.float32
    )
    criterion = DiceFocalLoss(
        dice_weight=config.get('loss', {}).get('dice_weight', 1.0),
        focal_weight=config.get('loss', {}).get('focal_weight', 1.0),
        focal_alpha=config.get('loss', {}).get('focal_alpha', 0.25),
        focal_gamma=config.get('loss', {}).get('focal_gamma', 2.0),
        class_weights=class_weights
    )


    print("\n创建训练器...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=config.config,
        checkpoint_dir=str(project_root / 'checkpoints'),
        log_dir=str(project_root / 'logs'),
        model_type=args.model_type
    )


    trainer.train()

    print("\n训练完成!")

if __name__ == '__main__':
    main()



