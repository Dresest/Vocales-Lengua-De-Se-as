import cv2
import mediapipe as mp
import numpy as np

# Inicializar mediapipe para detección de manos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=2)

# Función para detectar vocales según la posición de los dedos
def detectar_vocal(puntos_mano):
    dedos_levantados = []

    # Pulgar doblado o no
    if puntos_mano[4][0] < puntos_mano[3][0]:
        dedos_levantados.append('pulgar')

    # Índice levantado o no
    if puntos_mano[8][1] < puntos_mano[6][1]:
        dedos_levantados.append('indice')

    # Medio levantado o no
    if puntos_mano[12][1] < puntos_mano[10][1]:
        dedos_levantados.append('medio')

    # Anular levantado o no
    if puntos_mano[16][1] < puntos_mano[14][1]:
        dedos_levantados.append('anular')

    # Meñique levantado o no
    if puntos_mano[20][1] < puntos_mano[18][1]:
        dedos_levantados.append('meñique')

    # Reglas para detectar cada vocal
    if 'pulgar' in dedos_levantados and len(dedos_levantados) == 1:
        return 'A'
    elif len(dedos_levantados) == 0:
        return 'E'
    elif 'meñique' in dedos_levantados and len(dedos_levantados) == 1:
        return 'I'
    elif 'pulgar' in dedos_levantados and 'indice' in dedos_levantados and 'medio' in dedos_levantados and 'anular' in dedos_levantados and 'meñique' in dedos_levantados:
        return 'O'
    elif 'indice' in dedos_levantados and 'medio' in dedos_levantados and len(dedos_levantados) == 2:
        return 'U'

    return None

# Función para cargar imagen de la vocal
def vocalImage(vocal):
    if vocal == 'A': return cv2.imread('./imagenes_vocales/a.jpg')
    if vocal == 'E': return cv2.imread('./imagenes_vocales/e.jpg')
    if vocal == 'I': return cv2.imread('./imagenes_vocales/i.jpg')
    if vocal == 'O': return cv2.imread('./imagenes_vocales/o.jpg')
    if vocal == 'U': return cv2.imread('./imagenes_vocales/u.jpg')
    return cv2.imread('./imagenes_vocales/relog.jpg')  

# Función para dibujar una caja de texto con fondo y borde
def dibujar_texto_fondo(frame, texto, pos, tamaño=1, color=(0, 255, 0), grosor=2, fondo_color=(0, 0, 0), padding=5):
    (text_width, text_height), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, tamaño, grosor)
    x, y = pos
    # Dibujar rectángulo de fondo
    cv2.rectangle(frame, (x - padding, y - text_height - padding), (x + text_width + padding, y + padding), fondo_color, -1)
    # Dibujar el texto
    cv2.putText(frame, texto, (x, y), cv2.FONT_HERSHEY_SIMPLEX, tamaño, color, grosor)

# Captura de video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir imagen a RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar la imagen para detección de manos
    resultado = hands.process(frame_rgb)

    # Usar el frame original para mostrar las manos y las vocales
    nFrame = frame.copy()

    # Comprobar si se detectan manos
    if resultado.multi_hand_landmarks:
        for idx, mano in enumerate(resultado.multi_hand_landmarks):
            # Comprobar la mano dominante
            mano_dominante = resultado.multi_handedness[idx].classification[0].label
            if mano_dominante == 'Right':  # Solo procesar la mano izquierda, se utiliza el efecto espejo 
                color = (0, 255, 0)  # Color verde para la mano izquierda
                mp_drawing.draw_landmarks(nFrame, mano, mp_hands.HAND_CONNECTIONS, 
                                          mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=3),
                                          mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))

                # Obtener coordenadas de los puntos clave de la mano
                alto, ancho, _ = frame.shape
                puntos_mano = [(int(p.x * ancho), int(p.y * alto)) for p in mano.landmark]

                # Detectar la vocal en base a los puntos
                vocal = detectar_vocal(puntos_mano)
                if vocal:
                    # Dibujar la vocal detectada con fondo y borde
                    dibujar_texto_fondo(nFrame, f'Vocal: {vocal}', (50, 50), tamaño=1, color=color, grosor=2)

                    # Cargar la imagen de la vocal detectada
                    image = vocalImage(vocal)
                    if image is not None:  # Asegurarse de que se cargue una imagen válida
                        nFrame = cv2.hconcat([nFrame, image])
    else:
        # Si no se detecta ninguna mano, mostrar la imagen de reloj
        image_relog = cv2.imread('./imagenes_vocales/relog.jpg')
        if image_relog is not None:
            nFrame = cv2.hconcat([nFrame, image_relog])  # Concatenar la imagen de reloj

    # Mostrar el video con las líneas de las manos y la imagen concatenada
    cv2.imshow('Detección de señas para vocales', nFrame)

    # Salir con la tecla 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
