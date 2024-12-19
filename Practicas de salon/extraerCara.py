import numpy as np
import cv2 as cv
import os

# Cargar el clasificador en cascada para detección de rostros
rostro = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')

# Iniciar la captura de video desde la cámara
cap = cv.VideoCapture(0)
i = 0

# Asegurarse de que la carpeta 'assets' exista
output_folder = 'assets'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convertir el frame a escala de grises para la detección de rostros
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Detectar rostros en la imagen
    rostros = rostro.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in rostros:
        # Recortar la región del rostro detectado
        frame2 = frame[y:y+h, x:x+w]
        
        # Redimensionar la imagen del rostro a 100x100 píxeles
        frame2 = cv.resize(frame2, (100, 100), interpolation=cv.INTER_AREA)
        
        # Guardar la imagen del rostro en la carpeta 'assets'
        cv.imwrite(f'{output_folder}/Face{i}.jpg', frame2)
        
        # Mostrar la imagen del rostro detectado
        cv.imshow('rostro', frame2)

    # Mostrar la imagen original con los rostros detectados
    cv.imshow('rostros', frame)
    
    # Incrementar el contador de imágenes
    i += 1
    
    # Esperar por la tecla 'Esc' para salir
    k = cv.waitKey(1)
    if k == 27:  # 'Esc' para salir
        break

# Liberar la captura y cerrar todas las ventanas
cap.release()
cv.destroyAllWindows()
