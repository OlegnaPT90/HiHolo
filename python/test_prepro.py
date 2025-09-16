import numpy as np
import h5py
import matplotlib.pyplot as plt
import mytools

# Input/output files
input_file = "/home/hug/Downloads/HoloTomo_Data/holo_200angles_simu_format.h5"
datasets = "holodata_distance_0,holodata_distance_1,holodata_distance_2,holodata_distance_3"
back_dataset = "backgrounds"
dark_dataset = "darks"

angle = 127
data_angle = mytools.get_angle_data(input_file, datasets, angle)
print(data_angle.shape)
distances = data_angle.shape[0]

back_data = mytools.read_h5_to_float(input_file, back_dataset)
dark_data = mytools.read_h5_to_float(input_file, dark_dataset)

mytools.remove_outliers(data_angle)
mytools.remove_stripes(data_angle)

mytools.remove_outliers(back_data)
mytools.remove_outliers(dark_data)
dark_data = np.repeat(dark_data, distances, axis=0)
print(dark_data.shape)
print(dark_data[0])

holo_data = (data_angle - dark_data) / (back_data - dark_data)

holo_data, translations = mytools.register_images(holo_data)

display_data = mytools.scale_display_data(holo_data[1])
plt.figure(figsize=(8, 8))
plt.imshow(display_data, cmap='viridis')
plt.colorbar()
plt.title("holo_data first frame")
plt.show()
