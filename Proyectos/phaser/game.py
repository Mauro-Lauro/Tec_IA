import pygame
import random
import matplotlib.pyplot as plt
import numpy as np
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import os
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree

# Inicializar Pygame
pygame.init()
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# Dimensiones de la pantalla
w, h = 800, 400
pantalla = pygame.display.set_mode((w, h))
pygame.display.set_caption("Saltarin")

# Colores
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
AZUL = (0, 0, 255)

# Variables del jugador, bala, nave, fondo, etc.
jugador = None
bala = None
fondo = None
nave = None
menu = None

# Variables de salto
salto = False
salto_altura = 15  # Velocidad inicial de salto
gravedad = 1
en_suelo = True

# Variables de pausa y menú
pausa = False
fuente = pygame.font.SysFont('Arial', 24)
menu_activo = True
modo_auto = False  # Indica si el modo de juego es automático
modo_graf = False

# Lista para guardar los datos de velocidad, distancia y salto (target)
datos_modelo = []
modo = ""

# Cargar las imágenes
jugador_frames = [
    pygame.image.load('./assets/sprites/mono_frame_1.png'),
    pygame.image.load('./assets/sprites/mono_frame_2.png'),
    pygame.image.load('./assets/sprites/mono_frame_3.png'),
    pygame.image.load('./assets/sprites/mono_frame_4.png')
]

bala_img = pygame.image.load('./assets/sprites/purple_ball.png')
fondo_img = pygame.image.load('./assets/game/fondo2.png')
nave_img = pygame.image.load('./assets/game/ufo.png')
menu_img = pygame.image.load('./assets/game/menu.png')

# Escalar la imagen de fondo para que coincida con el tamaño de la pantalla
fondo_img = pygame.transform.scale(fondo_img, (w, h))

# Crear el rectángulo del jugador y de la bala
jugador = pygame.Rect(50, h - 100, 32, 48)
bala = pygame.Rect(w - 50, h - 90, 16, 16)
nave = pygame.Rect(w - 100, h - 100, 64, 64)
menu_rect = pygame.Rect(w // 2 - 135, h // 2 - 90, 270, 180)

# Variables para la animación del jugador
current_frame = 0
frame_speed = 10  # Cuántos frames antes de cambiar a la siguiente imagen
frame_count = 0

# Variables para la bala
velocidad_bala = -10  # Velocidad de la bala hacia la izquierda
bala_disparada = False

# Variables para el fondo en movimiento
fondo_x1 = 0
fondo_x2 = w

def disparar_bala():
    global bala_disparada, velocidad_bala
    if not bala_disparada:
        velocidad_bala = random.randint(-8, -3)  # Velocidad aleatoria negativa para la bala
        bala_disparada = True

def reset_bala():
    global bala, bala_disparada
    bala.x = w - 50  # Reiniciar la posición de la bala
    bala_disparada = False

def manejar_salto():
    global jugador, salto, salto_altura, gravedad, en_suelo

    if salto:
        jugador.y -= salto_altura  # Mover al jugador hacia arriba
        salto_altura -= gravedad  # Aplicar gravedad (reduce la velocidad del salto)

        # Si el jugador llega al suelo, detener el salto
        if jugador.y >= h - 100:
            jugador.y = h - 100
            salto = False
            salto_altura = 15  # Restablecer la velocidad de salto
            en_suelo = True

def update():
    global bala, velocidad_bala, current_frame, frame_count, fondo_x1, fondo_x2

    # Mover el fondo
    fondo_x1 -= 1
    fondo_x2 -= 1

    # Si el primer fondo sale de la pantalla, lo movemos detrás del segundo
    if fondo_x1 <= -w:
        fondo_x1 = w

    # Si el segundo fondo sale de la pantalla, lo movemos detrás del primero
    if fondo_x2 <= -w:
        fondo_x2 = w

    # Dibujar los fondos
    pantalla.blit(fondo_img, (fondo_x1, 0))
    pantalla.blit(fondo_img, (fondo_x2, 0))

    # Animación del jugador
    frame_count += 1
    if frame_count >= frame_speed:
        current_frame = (current_frame + 1) % len(jugador_frames)
        frame_count = 0

    # Dibujar el jugador con la animación
    pantalla.blit(jugador_frames[current_frame], (jugador.x, jugador.y))

    # Dibujar la nave
    pantalla.blit(nave_img, (nave.x, nave.y))

    # Mover y dibujar la bala
    if bala_disparada:
        bala.x += velocidad_bala

    # Si la bala sale de la pantalla, reiniciar su posición
    if bala.x < 0:
        reset_bala()

    pantalla.blit(bala_img, (bala.x, bala.y))

    # Colisión entre la bala y el jugador
    if jugador.colliderect(bala):
        print("Colisión detectada!")
        reiniciar_juego()  # Terminar el juego y mostrar el menú

def guardar_datos():
    global jugador, bala, velocidad_bala, salto
    distancia = abs(jugador.x - bala.x)
    salto_hecho = 1 if salto else 0  
    datos_modelo.append((velocidad_bala, distancia, salto_hecho))

def pausa_juego():
    global pausa, menu_activo
    pausa = not pausa
    if pausa:
        menu_activo = True
        print("Juego pausado. Datos registrados hasta ahora:", datos_modelo)
        mostrar_menu() 
    else:
        menu_activo = False
        print("Juego reanudado.")
        
def training(modo):
    global model_rn, arbol_clf
    if modo == "ad":
        print("Entrenando Decision Tree...")
        arbol_data = [tuple(map(int, row)) for row in datos_modelo]
        xa = np.array([d[:2] for d in arbol_data])
        ya = np.array([d[2] for d in arbol_data]).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            xa, ya, test_size=0.2, random_state=42)
        arbol_clf = DecisionTreeClassifier(random_state=42, max_depth=1)
        arbol_clf.fit(X_train, y_train)
        y_pred = arbol_clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")
        predict = arbol_clf.predict(np.array([[-5, 660]]))[0]
        print(predict)
        plt.figure(figsize=(10, 8))
        plot_tree(
            arbol_clf,
            filled=True,
            feature_names=["Velocidad", "Distancia"],
            class_names=["No Saltó", "Saltó"],
            rounded=True,
        )

    elif modo == "rn":
        print("Entrenando Neuronal Network...")
        red_data = [tuple(map(int, row)) for row in datos_modelo]
        xr = np.array([d[:2] for d in red_data])
        yr = np.array([d[2] for d in red_data]).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
        xr, yr, test_size=0.2, random_state=42
        )
        model_rn = Sequential(
            [
                Dense(4, input_dim=2, activation="relu"),
                Dense(1, activation="sigmoid"), 
            ]
        )
        model_rn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        model_rn.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)
        loss, accuracy = model_rn.evaluate(X_test, y_test, verbose=0)
        print(f"Accuracy: {accuracy:.2f}%")
        predict = model_rn.predict(X_test)
        for i in range(len(predict)):
            print(f"Real: {y_test[i]}, Prediction: {int(np.round(predict[i][0]))}")
        predict = model_rn.predict(np.array([[-8, 52]]), verbose=0)[0][0]
        print(predict)     
        
def predict_salto():
    global jugador, bala, velocidad_bala, dt_model, selected_model, model_rn, arbol_clf
    distancia = abs(jugador.x - bala.x)
    formatted = np.array([[velocidad_bala, distancia]])
    if selected_model == "rn": 
        res = model_rn.predict(formatted, verbose=0)[0][0]
        rounded = int(np.round(res))
        print(f"Prediccion: ",res)
        return True if rounded == 1 else False
    if selected_model == "ad":
        res = arbol_clf.predict(formatted)[0]
        print(f"Prediccion: ",res)
        return True if res == 1 else False

def mostrar_menu():
    global menu_activo, modo_auto, selected_model, pausa
    pantalla.fill(NEGRO)

    botones = [
        {"texto": "Manual (M)", "key": pygame.K_m},
        {"texto": "Red Neuronal (N)", "key": pygame.K_n},
        {"texto": "Árbol de Decisión (T)", "key": pygame.K_t},
        {"texto": "Pausa (P)"},
        {"texto": "Salir (Borrar)", "key": pygame.K_BACKSPACE}
    ]

    y_offset = h // 2 - (len(botones) * 40) // 2
    for boton in botones:
        # Crear rectángulo del botón
        boton_rect = pygame.Rect(w // 2 - 100, y_offset, 200, 30)
        # Dibujar el borde del botón
        pygame.draw.rect(pantalla, AZUL, boton_rect, 2)
        
        # Renderizar texto del botón
        texto = fuente.render(boton["texto"], True, BLANCO)
        texto_rect = texto.get_rect(center=(w // 2, y_offset + 15))
        pantalla.blit(texto, texto_rect)
        
        y_offset += 40

    pygame.display.flip()

    while menu_activo:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                exit()
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_n or evento.key == pygame.K_t:
                    selected_model = "ad" if evento.key == pygame.K_t else "rn"
                    training(selected_model)
                    modo_auto = True
                    pausa = False
                    menu_activo = False
                elif evento.key == pygame.K_m:
                    datos_modelo.clear()
                    print(f"nuevos datos: ",datos_modelo)
                    modo_auto = False
                    print("Modo manual seleccionado.")
                    pausa = False
                    menu_activo = False
                elif evento.key == pygame.K_BACKSPACE:
                    print("Juego terminado. Datos recopilados:", datos_modelo)
                    pygame.quit()
                    exit()

def reiniciar_juego():
    global menu_activo, jugador, bala, nave, bala_disparada, salto, en_suelo
    menu_activo = True  # Activar de nuevo el menú
    jugador.x, jugador.y = 50, h - 100  # Reiniciar posición del jugador
    bala.x = w - 50  # Reiniciar posición de la bala
    nave.x, nave.y = w - 100, h - 100  # Reiniciar posición de la nave
    bala_disparada = False
    salto = False
    en_suelo = True
    print("Datos recopilados para el modelo: ", datos_modelo)
    mostrar_menu()

def main():
    global salto, en_suelo, bala_disparada, velocidad_jugador, menu_activo

    reloj = pygame.time.Clock()
    mostrar_menu()
    correr = True
    
    while correr:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                correr = False
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_w and en_suelo and not pausa:
                    salto = True
                    en_suelo = False
                if evento.key == pygame.K_p:
                    pausa_juego()
                if evento.key == pygame.K_BACKSPACE:
                    print("Juego terminado. Datos recopilados:", datos_modelo)
                    pygame.quit()
                    exit()

        if not pausa:
            # Modo manual: el jugador controla el salto
            if not modo_auto:
                if salto:
                    manejar_salto()
                # Guardar los datos si estamos en modo manual
                guardar_datos()
            # Modo automático: el salto se activa automáticamente
            if modo_auto:
                if predict_salto():
                    salto = True
                    en_suelo = False
                if salto:
                    manejar_salto()

            # Actualizar el juego
            if not bala_disparada:
                disparar_bala()
            update()

        # Actualizar la pantalla
        pygame.display.flip()
        reloj.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()