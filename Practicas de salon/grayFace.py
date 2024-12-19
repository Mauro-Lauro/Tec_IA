import numpy as np
import cv2

# Configuración inicial
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
video_capture = cv2.VideoCapture(0)
frame_counter = 0
previous_white_pixels = 0  # Para comparar la cantidad de píxeles blancos

def process_face_region(face_region, frame_counter, previous_white_pixels):
    """
    Procesa la región del rostro detectado: redimensiona, convierte a binario,
    cuenta píxeles blancos y guarda la imagen binaria.
    """
    # Redimensionar el rostro a 100x100 píxeles
    resized_face = cv2.resize(face_region, (100, 100), interpolation=cv2.INTER_AREA)

    # Convertir a escala de grises
    gray_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)

    # Aplicar umbral para obtener una imagen binaria
    _, binary_face = cv2.threshold(gray_face, 127, 255, cv2.THRESH_BINARY)

    # Contar píxeles blancos
    current_white_pixels = cv2.countNonZero(binary_face)

    # Calcular la diferencia con respecto a la cantidad anterior de píxeles blancos
    if frame_counter > 0:
        difference = current_white_pixels - previous_white_pixels
        print(f'Píxeles blancos actuales: {current_white_pixels}, Diferencia: {difference}')
    else:
        print(f'Píxeles blancos actuales: {current_white_pixels}')

    # Guardar la imagen binaria
    cv2.imwrite(f'capturas/FaceBinary_{frame_counter}.jpg', binary_face)

    # Mostrar la imagen binaria
    cv2.imshow('Imagen Binaria del Rostro', binary_face)

    return current_white_pixels

while True:
    # Capturar el cuadro de video
    ret, frame = video_capture.read()
    if not ret:
        print("Error al capturar el video. Saliendo...")
        break

    # Convertir el cuadro a escala de grises
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extraer la región del rostro
        face_region = frame[y:y+h, x:x+w]

        # Procesar la región del rostro
        previous_white_pixels = process_face_region(face_region, frame_counter, previous_white_pixels)

    # Mostrar la imagen original con rostros detectados
    cv2.imshow('Rostros Detectados', frame)

    frame_counter += 1

    # Salir al presionar 'Esc'
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Liberar recursos
video_capture.release()
cv2.destroyAllWindows()
