import hiholo
import mytools

# Read holograms
input_file = "data_shift.h5"
output_file = "data_regist.h5"
dataset = "holodata"
holo_data = mytools.read_h5_to_float(input_file, dataset)

print(f"Original holo_data shape: {holo_data.shape}")

# Test registerImages interface
# registered_images: aligned hologram images
# translations: [[x1, y1], [x2, y2], ...], translation offsets for each image
registered_images, translations = hiholo.registerImages(holo_data)

print(f"Registered images shape: {registered_images.shape}")
print("Translations:")
for i, translation in enumerate(translations):
    print(f"  Image {i}: x={translation[0]}, y={translation[1]}")

# Save registered images
mytools.save_h5_from_float(output_file, dataset, registered_images)
print(f"Registered images saved to: {output_file}")

# Display comparison of original vs registered images
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Image Registration Results: Original (Top) vs Registered (Bottom)', fontsize=14)

for i in range(4):
    # Original images (top row)
    axes[0, i].imshow(holo_data[i], cmap='gray')
    axes[0, i].set_title(f'Original Image {i}')
    axes[0, i].axis('off')
    
    # Registered images (bottom row)
    axes[1, i].imshow(registered_images[i], cmap='gray')
    axes[1, i].set_title(f'Registered Image {i}\nTranslation: ({translations[i][0]}, {translations[i][1]})')
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()