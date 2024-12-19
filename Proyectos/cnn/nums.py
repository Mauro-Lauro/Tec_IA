import os
import re

def rename_and_renumber_images(folder_path, prefix="syclone_", extension=".jpg"):

    files = [f for f in os.listdir(folder_path) if f.startswith(prefix) and f.endswith(extension)]

    numbered_files = []
    for file in files:
        match = re.search(r"(\d+)", file)
        if match:
            number = int(match.group(1))
            numbered_files.append((number, file))
    

    numbered_files.sort()

    # Renombrar los archivos con números consecutivos
    for i, (_, file) in enumerate(numbered_files, start=1):
        new_name = f"{prefix}{i:05d}{extension}"
        old_path = os.path.join(folder_path, file)
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {file} -> {new_name}")


rename_and_renumber_images("src/dataset/syclone")