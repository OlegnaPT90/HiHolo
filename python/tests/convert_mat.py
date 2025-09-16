import numpy as np
import h5py

# 输入输出文件
input_file = "/home/hug/Downloads/HoloTomo_Data/holo_200angles_simu_format.h5"
phase_file = "/home/hug/Downloads/HoloTomo_Data/result.h5"
output_file = "/home/hug/Downloads/HoloTomo_Data/holo_200angles_phase.h5"
back_dataset = "backgrounds"
dark_dataset = "darks"
phase_dataset = "phasedata"
angles = 200
distance = 4

# back_data = np.ones((distance, 500, 500)).astype(np.float32)
# dark_data = np.random.rand(1, 500, 500).astype(np.float32) * 1.0/50
# print(dark_data[0])

# with h5py.File(input_file, 'a') as f:
#     if back_dataset in f:
#         del f[back_dataset]
#     f.create_dataset(back_dataset, data=back_data)
#     if dark_dataset in f:
#         del f[dark_dataset]
#     f.create_dataset(dark_dataset, data=dark_data)

phase_data = None
with h5py.File(phase_file, 'r') as f:
    phase_data = np.array(f[phase_dataset], dtype=np.float32)

# 将2D相位数据扩展为3D，形状为(angles, H, W)
phase_data_3d = np.repeat(phase_data[np.newaxis, :, :], angles, axis=0)
print(f"扩展后的phase_data_3d形状: {phase_data_3d.shape}")

with h5py.File(output_file, 'w') as f:
    f.create_dataset(phase_dataset, data=phase_data_3d, dtype=np.float32)
print(f"已在{output_file}中写入数据集：{phase_dataset}，形状为{phase_data_3d.shape}")
