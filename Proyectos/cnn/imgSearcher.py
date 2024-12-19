import cv2
import os
import re
import numpy as np
from bing_image_downloader import downloader

def process_and_detect_cars(query_string, output_dir, yolo_weights, yolo_cfg, coco_names, limit=5, image_size=(80, 80)):

    os.makedirs(output_dir, exist_ok=True)

    # Obtener el último índice en la carpeta
    existing_files = [f for f in os.listdir(output_dir) if f.startswith("CANAM_") and f.endswith(".jpg")]
    if existing_files:
        last_index = max([int(re.search(r"(\d+)", f).group()) for f in existing_files])
    else:
        last_index = -1 

    frame_count = last_index + 1

    # Descargar las imágenes con bing_image_downloader
    downloader.download(
        query_string,
        limit=limit,
        output_dir="temp",
        adult_filter_off=True,
        force_replace=False,
        timeout=120,
        verbose=True
    )

    # Configurar YOLO
    net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Cargar las clases del modelo
    with open(coco_names, "r") as f:
        classes = f.read().strip().split("\n")

    # Procesar cada imagen descargada
    downloaded_dir = os.path.join("temp", query_string)
    for file_name in os.listdir(downloaded_dir):
        image_path = os.path.join(downloaded_dir, file_name)
        image = cv2.imread(image_path)
        if image is None:
            continue

        height, width, _ = image.shape
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward(output_layers)

        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = int(scores.argmax())
                confidence = scores[class_id]

                if classes[class_id] == "car" and confidence > 0.5:
                    # Obtener coordenadas del bounding box
                    center_x, center_y, w, h = (obj[0:4] * [width, height, width, height]).astype("int")
                    x = max(0, int(center_x - w / 2))
                    y = max(0, int(center_y - h / 2))
                    w = min(w, width - x)
                    h = min(h, height - y)

                    # Recortar la imagen al coche
                    cropped_car = image[y:y+h, x:x+w]
                    resized_car = cv2.resize(cropped_car, image_size)

                    # Guardar la imagen original y las rotaciones
                    for angle in range(0, 360, 30):
                        M = cv2.getRotationMatrix2D((image_size[0] // 2, image_size[1] // 2), angle, 1.0)
                        rotated_car = cv2.warpAffine(resized_car, M, image_size)

                        frame_name = os.path.join(output_dir, f"CANAM_{frame_count:05d}.jpg")
                        cv2.imwrite(frame_name, rotated_car)
                        print(f"Saved {frame_name}")
                        frame_count += 1

    # Eliminar la carpeta temporal
    for file_name in os.listdir(downloaded_dir):
        os.remove(os.path.join(downloaded_dir, file_name))
    os.rmdir(downloaded_dir)
    print(f"Processed and saved {frame_count - (last_index + 1)} car images to {output_dir}")

# Ejemplo de uso
process_and_detect_cars(
    query_string="CAN AM MAVERICK",
    output_dir="src/dataset/CAN AM MAVERICK2/",
    yolo_weights="src/yolov3.weights", 
    yolo_cfg="src/yolov3.cfg",        
    coco_names="src/coco.names",     
    limit=80,                           
    image_size=(80, 80),              
)
