import os
import random

def reduce_images_to_sample(directory, target_count=5600):
    files = [f for f in os.listdir(directory) if f.endswith(".jpg")]
    
    current_count = len(files)
    if current_count <= target_count:
        print(f"No se necesita eliminar archivos. Actualmente hay {current_count} fotos.")
        return
    
    files_to_remove_count = current_count - target_count
    print(f"Actualmente hay {current_count} fotos. Se eliminarán {files_to_remove_count} fotos.")

    files_to_remove = random.sample(files, files_to_remove_count)
    
    for file in files_to_remove:
        file_path = os.path.join(directory, file)
        os.remove(file_path)
        print(f"Eliminado: {file}")
    
    print(f"Reducción completada. Ahora hay {target_count} fotos.")

directory_path = "src/dataset/Fiat500"  
reduce_images_to_sample(directory_path, target_count=5300)