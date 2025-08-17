import numpy as np
import h5py
import matplotlib.pyplot as plt
import hiholo

def display_phase(phase, title="Phase"):
    """Display phase image"""
    plt.figure(figsize=(8, 8))
    plt.imshow(phase, cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.pause(3)
    plt.close()

def test_ctf_reconstruction():
    """Test CTF holographic reconstruction with hiholo"""
    
    # Input/output files
    input_file = "/home/hug/Downloads/HoloTomo_Data/holo_purephase.h5"
    input_dataset = "holodata"
    output_file = "/home/hug/Downloads/HoloTomo_Data/purephase_ctf_result.h5"
    output_dataset = "phasedata"
    
    # CTF reconstruction parameters
    fresnel_numbers = [[1.6667e-3], [8.3333e-4], [4.83333e-4], [2.66667e-4]]  # wing of dragonfly
    low_freq_lim = 1e-3      # 低频正则化参数
    high_freq_lim = 1e-1     # 高频正则化参数  
    beta_delta_ratio = 0.1   # β/δ比值（吸收与相位偏移的比值）
    
    # Padding parameters (optional)
    pad_size = [50, 50]    # 填充大小
    pad_type = hiholo.PaddingType.Replicate  # 填充类型
    pad_value = 0.0          # 填充值
    
    try:
        # Read hologram data
        print("读取全息图数据...")
        with h5py.File(input_file, 'r') as f:
            # 直接读取为numpy数组，保持原始维度
            holo_data = np.array(f[input_dataset], dtype=np.float32)
        
        # 确保数据是2D的（单张图像）或3D的（多张图像）
        if holo_data.ndim == 2:
            print(f"单张图像重建，尺寸: {holo_data.shape}")
        elif holo_data.ndim == 3:
            print(f"多张图像重建，图像数量: {holo_data.shape[0]}，单张尺寸: {holo_data.shape[1:3]}")
        
        # 执行CTF重建
        print("开始CTF重建...")
        result = hiholo.reconstruct_ctf(
            holograms=holo_data,           # 直接传入numpy数组
            fresnelNumbers=fresnel_numbers,
            lowFreqLim=low_freq_lim,
            highFreqLim=high_freq_lim,
            betaDeltaRatio=beta_delta_ratio,
            padSize=pad_size,
            padType=pad_type,
            padValue=pad_value
        )
        
        print("CTF重建完成！")
                
        # 显示结果
        print("显示重建结果...")
        display_phase(result, "CTF Reconstructed Phase")
        
        # 保存结果到HDF5文件
        print(f"保存结果到: {output_file}")
        with h5py.File(output_file, 'w') as f:
            f.create_dataset(output_dataset, data=result, dtype=np.float32)
        
        # 保存为图像文件
        plt.imsave("ctf_phase.png", result, cmap='viridis')        
        
    except FileNotFoundError:
        print(f"错误: 找不到输入文件 {input_file}")
        print("请检查文件路径是否正确")
    except KeyError as e:
        print(f"错误: 数据集 '{input_dataset}' 在文件中不存在")
        print(f"键错误: {e}")
    except Exception as e:
        print(f"重建过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 测试单张图像CTF重建
    test_ctf_reconstruction() 