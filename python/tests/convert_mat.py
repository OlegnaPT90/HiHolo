#!/usr/bin/env python3
"""
将MAT文件中的hologram字段数据转换为3D HDF5数据文件
从多个MAT文件中提取hologram字段(2D double)，重组成3D数组(float)并保存为HDF5格式
"""

import numpy as np
import h5py
import scipy.io
import glob
import os
import argparse
from pathlib import Path


def load_mat_files(pattern="*.mat", directory="."):
    """
    加载匹配模式的MAT文件并提取hologram字段
    
    Args:
        pattern: MAT文件匹配模式
        directory: 搜索目录
        
    Returns:
        list: 按文件名排序的2D hologram数据列表
        list: 对应的文件名列表
    """
    # 构建完整的搜索路径
    search_pattern = os.path.join(directory, pattern)
    mat_files = glob.glob(search_pattern)
    mat_files.sort()  # 按文件名排序
    
    if len(mat_files) == 0:
        raise FileNotFoundError(f"未找到匹配模式 '{search_pattern}' 的文件")
    
    print(f"找到 {len(mat_files)} 个MAT文件:")
    for file in mat_files:
        print(f"  - {file}")
    
    holograms = []
    filenames = []
    
    for mat_file in mat_files:
        try:
            print(f"\n正在处理: {mat_file}")
            
            # 加载MAT文件
            mat_data = scipy.io.loadmat(mat_file)
            
            # 检查是否存在hologram字段
            if 'hologram' not in mat_data:
                print(f"警告: {mat_file} 中未找到 'hologram' 字段")
                print(f"可用字段: {[key for key in mat_data.keys() if not key.startswith('__')]}")
                continue
            
            # 提取hologram数据
            hologram = mat_data['hologram']
            
            # 验证数据是2D的
            if hologram.ndim != 2:
                print(f"警告: {mat_file} 中的hologram不是2D数组，实际维度: {hologram.ndim}")
                continue
            
            # 转换为float32以节省内存
            hologram_float = hologram.astype(np.float32)
            
            holograms.append(hologram_float)
            filenames.append(os.path.basename(mat_file))
            
            print(f"  成功加载: 形状 {hologram.shape}, 原始类型 {hologram.dtype}")
            print(f"  转换后: 形状 {hologram_float.shape}, 类型 {hologram_float.dtype}")
            print(f"  数值范围: {hologram_float.min():.6f} 到 {hologram_float.max():.6f}")
            
        except Exception as e:
            print(f"错误: 处理 {mat_file} 时发生错误: {e}")
            continue
    
    if len(holograms) == 0:
        raise ValueError("没有成功加载任何hologram数据")
    
    return holograms, filenames


def create_3d_hologram_array(holograms):
    """
    将2D hologram列表合并为3D数组
    
    Args:
        holograms: 2D hologram数据列表，每个形状为 (height, width)
        
    Returns:
        numpy.ndarray: 3D数组，形状为 (num_holograms, height, width)
    """
    # 检查所有hologram是否具有相同的形状
    shapes = [holo.shape for holo in holograms]
    if not all(shape == shapes[0] for shape in shapes):
        print("警告: 不是所有hologram都有相同的形状:")
        for i, shape in enumerate(shapes):
            print(f"  hologram {i}: {shape}")
        raise ValueError(f"所有hologram必须具有相同的形状，但得到: {shapes}")
    
    # 确保所有hologram都是2D的
    for i, holo in enumerate(holograms):
        if holo.ndim != 2:
            raise ValueError(f"hologram {i} 不是2D数组，形状: {holo.shape}")
    
    # 将hologram堆叠成3D数组
    data_3d = np.stack(holograms, axis=0)
    print(f"\n创建3D数组: 形状 {data_3d.shape}, 数据类型 {data_3d.dtype}")
    print(f"内存使用: {data_3d.nbytes / (1024**3):.2f} GB")
    
    # 验证结果确实是3D的
    if data_3d.ndim != 3:
        raise ValueError(f"最终数组不是3D的，实际维度: {data_3d.ndim}, 形状: {data_3d.shape}")
    
    return data_3d


def save_to_h5(data_3d, filenames, output_file="holograms_3d.h5", dataset_name="holograms"):
    """
    将3D hologram数组保存为HDF5文件
    
    Args:
        data_3d: 3D numpy数组
        filenames: 原始文件名列表
        output_file: 输出文件名
        dataset_name: 数据集名称
    """
    print(f"\n正在保存到 {output_file}...")
    
    with h5py.File(output_file, 'w') as f:
        # 创建主数据集
        dataset = f.create_dataset(dataset_name, data=data_3d, 
                                 compression='gzip', compression_opts=9,
                                 dtype=np.float32)
        
        # 添加属性信息
        dataset.attrs['description'] = '从MAT文件提取的hologram数据重组为3D数组'
        dataset.attrs['shape'] = data_3d.shape
        dataset.attrs['dtype'] = str(data_3d.dtype)
        dataset.attrs['num_holograms'] = data_3d.shape[0]
        dataset.attrs['height'] = data_3d.shape[1]
        dataset.attrs['width'] = data_3d.shape[2]
        dataset.attrs['data_range_min'] = float(data_3d.min())
        dataset.attrs['data_range_max'] = float(data_3d.max())
        
        # 保存文件名信息
        filename_group = f.create_group('metadata')
        filename_dataset = filename_group.create_dataset('source_files', 
                                                       data=[name.encode('utf-8') for name in filenames])
        filename_dataset.attrs['description'] = '源MAT文件名列表'
        
        # 保存每个hologram的统计信息
        stats_group = filename_group.create_group('statistics')
        for i, (filename, holo_slice) in enumerate(zip(filenames, data_3d)):
            holo_group = stats_group.create_group(f'hologram_{i:02d}')
            holo_group.attrs['filename'] = filename
            holo_group.attrs['min'] = float(holo_slice.min())
            holo_group.attrs['max'] = float(holo_slice.max())
            holo_group.attrs['mean'] = float(holo_slice.mean())
            holo_group.attrs['std'] = float(holo_slice.std())
    
    print(f"成功保存到 {output_file}")
    print(f"数据集名称: {dataset_name}")
    print(f"数据形状: {data_3d.shape}")
    print(f"数据类型: {data_3d.dtype}")
    print(f"压缩后文件大小: {os.path.getsize(output_file) / (1024**2):.2f} MB")


def verify_h5_file(h5_file, dataset_name="holograms"):
    """
    验证生成的HDF5文件
    
    Args:
        h5_file: HDF5文件路径
        dataset_name: 数据集名称
    """
    print(f"\n正在验证 {h5_file}...")
    
    try:
        with h5py.File(h5_file, 'r') as f:
            # 检查数据集
            if dataset_name not in f:
                print(f"错误: 数据集 '{dataset_name}' 不存在")
                return False
            
            dataset = f[dataset_name]
            print(f"验证成功:")
            print(f"  数据形状: {dataset.shape}")
            print(f"  数据类型: {dataset.dtype}")
            print(f"  压缩方式: {dataset.compression}")
            
            # 检查属性
            print(f"  属性:")
            for key, value in dataset.attrs.items():
                print(f"    {key}: {value}")
            
            # 检查元数据
            if 'metadata' in f:
                print(f"  源文件数量: {len(f['metadata/source_files'])}")
                print(f"  源文件列表:")
                for i, filename in enumerate(f['metadata/source_files']):
                    print(f"    {i}: {filename.decode('utf-8')}")
            
            return True
            
    except Exception as e:
        print(f"验证失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='将MAT文件中的hologram数据转换为3D HDF5文件')
    parser.add_argument('--pattern', default='*.mat', 
                       help='MAT文件匹配模式 (默认: *.mat)')
    parser.add_argument('--directory', default='.', 
                       help='搜索目录 (默认: 当前目录)')
    parser.add_argument('--output', default='holograms_3d.h5', 
                       help='输出HDF5文件名 (默认: holograms_3d.h5)')
    parser.add_argument('--dataset', default='holograms', 
                       help='HDF5数据集名称 (默认: holograms)')
    parser.add_argument('--verify', action='store_true', 
                       help='转换完成后验证输出文件')
    
    args = parser.parse_args()
    
    try:
        print("=== MAT到HDF5转换工具 ===")
        print(f"搜索模式: {args.pattern}")
        print(f"搜索目录: {args.directory}")
        print(f"输出文件: {args.output}")
        print(f"数据集名称: {args.dataset}")
        
        # 加载MAT文件
        print("\n1. 加载MAT文件...")
        holograms, filenames = load_mat_files(args.pattern, args.directory)
        
        # 创建3D数组
        print("\n2. 创建3D数组...")
        data_3d = create_3d_hologram_array(holograms)
        
        # 保存为HDF5文件
        print("\n3. 保存为HDF5文件...")
        save_to_h5(data_3d, filenames, args.output, args.dataset)
        
        # 验证输出文件
        if args.verify:
            print("\n4. 验证输出文件...")
            verify_h5_file(args.output, args.dataset)
        
        print("\n=== 转换完成! ===")
        
    except Exception as e:
        print(f"\n错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
