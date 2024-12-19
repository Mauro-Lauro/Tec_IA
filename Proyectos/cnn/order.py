import os

def rename_images(directory):
    # Obtener una lista de todos los archivos que terminan en ".jpg"
    files = [f for f in os.listdir(directory) if f.endswith(".jpg")]
    
    # Ordenar los archivos alfabéticamente para mantener el orden
    files.sort()
    
    for index, file in enumerate(files, start=1):  # Comenzar desde 1
        # Crear el nuevo nombre del archivo
        new_name = f"{index}.jpg"  # Nombres consecutivos desde 1
        
        # Ruta completa para el archivo actual y el nuevo nombre
        old_path = os.path.join(directory, file)
        new_path = os.path.join(directory, new_name)
        
        # Renombrar el archivo
        os.rename(old_path, new_path)
        print(f"Renamed {file} -> {new_name}")

# Ruta al directorio con las imágenes
directory_path = "src/dataset/Senna"  # Reemplaza con tu directorio
rename_images(directory_path)