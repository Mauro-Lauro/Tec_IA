import cv2
import numpy as np

# Iniciar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

# Definir los rangos de color en HSV para rastrear el color rojo
color_lower_bound_1 = np.array([0, 150, 50])  # Parte inferior del espectro rojo
color_upper_bound_1 = np.array([10, 255, 255])

color_lower_bound_2 = np.array([170, 150, 50])  # Parte superior del espectro rojo
color_upper_bound_2 = np.array([180, 255, 255])

def preprocess_mask(hsv_frame):
    """
    Genera y procesa una máscara binaria para detectar el color rojo en un rango HSV.
    """
    # Crear máscaras para ambas partes del espectro rojo
    mask1 = cv2.inRange(hsv_frame, color_lower_bound_1, color_upper_bound_1)
    mask2 = cv2.inRange(hsv_frame, color_lower_bound_2, color_upper_bound_2)
    
    # Combinar las máscaras
    combined_mask = cv2.bitwise_or(mask1, mask2)
    
    # Aplicar erosión y dilatación para eliminar ruido
    processed_mask = cv2.erode(combined_mask, None, iterations=2)
    processed_mask = cv2.dilate(processed_mask, None, iterations=2)
    
    return processed_mask

while True:
    # Capturar frame por frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convertir el frame de BGR a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Preprocesar la máscara para detectar el color rojo
    mask = preprocess_mask(hsv)
    
    # Encontrar contornos en la máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Si se encuentra al menos un contorno, seguir el objeto
    if contours:
        # Tomar el contorno más grande
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Encontrar el centro del contorno usando un círculo mínimo que lo rodee
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
        
        # Dibujar el círculo y el centro en el frame original si el radio es mayor que un umbral
        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), -1)
    
    # Mostrar el frame
    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
