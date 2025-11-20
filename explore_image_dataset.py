import os

dataset_path = 'soil-data'  # Path to your main dataset folder

soil_types = os.listdir(dataset_path)
print("Soil types (folder names):")
print(soil_types)

for soil_type in soil_types:
    folder_path = os.path.join(dataset_path, soil_type)
    num_images = len(os.listdir(folder_path))
    print(f"{soil_type}: {num_images} images")
