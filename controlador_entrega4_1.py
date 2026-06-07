from controller import Display, Keyboard, Robot, Camera
from vehicle import Driver
import numpy as np
import cv2
from datetime import datetime
import os
import json

# Suprimir mensajes informativos de TensorFlow en consola (solo mostrar errores)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["KERAS_BACKEND"] = "tensorflow"

import time
import tensorflow as tf
from tensorflow import keras


# =============================================================================
# CONFIGURACIÓN DEL VEHÍCULO
# =============================================================================
DEBOUNCE_TIME = 0.1  # Tiempo mínimo (s) entre pulsaciones de la misma tecla
MAX_ANGLE     = 0.3  # Ángulo máximo de giro izquierda/derecha (radianes)
MAX_SPEED     = 180  # Velocidad máxima hacia adelante (km/h)
MAX_REVERSE   = -30  # Velocidad máxima en reversa (km/h, valor negativo)
SPEED_INCR    = 5    # Incremento de velocidad por pulsación de tecla (km/h)

# =============================================================================
# PARÁMETROS DE DETECCIÓN
# =============================================================================
INFER_EVERY       = 3    # Ejecutar el modelo cada N fotogramas (reduce carga de CPU/GPU)
CONF_UMBRAL       = 0.85 # Confianza mínima para registrar una detección (aumentado para reducir falsos positivos)
ENTROPY_MAX       = 1.5  # Entropía máxima permitida (baja = predición concentrada = señal real)
FRAMES_REQUERIDOS = 3    # Fotogramas consecutivos con la misma clase para confirmar detección
COOLDOWN_SENALES  = 2    # Segundos mínimos entre detecciones de la misma clase

# =============================================================================
# OBJETIVO DE LA SESIÓN
# =============================================================================
# Hay 16 señales en el mundo Webots; la meta es detectar al menos el 50% (8)
TOTAL_SENALES = 16
META_SENALES  = TOTAL_SENALES // 2  # 8 señales únicas

# =============================================================================
SPEED_LIMIT_CANONICAL = {
    18: 3,   # 40 km/h -> clase canónica 3
    19: 4,   # 50 km/h -> clase canónica 4
}


def load_model():
    #Carga model.keras y sus metadatos (descripciones de clase, número de clases).
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model.keras")
    meta_path  = os.path.join(script_dir, "checkpoint2", "meta2.json")

    # Si no está en la carpeta raíz del script, buscar en la subcarpeta checkpoint2
    if not os.path.exists(model_path):
        model_path = os.path.join(script_dir, "checkpoint2", "model.keras")

    if not os.path.exists(model_path):
        print(f"ERROR: Modelo no encontrado en {model_path}")
        print("  Ejecuta train2.py primero para crear el modelo")
        raise FileNotFoundError(model_path)

    model = keras.models.load_model(model_path)

    # Cargar metadatos si existen; si no, usar valores predeterminados
    class_descriptions = {}
    num_classes = 58
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            class_descriptions = meta.get("class_descriptions", {})
            num_classes        = meta.get("NUM_CLASSES", 58)

    # Detectar si hay GPU disponible para mostrarlo en el resumen de inicio
    gpus = tf.config.list_physical_devices('GPU')
    dispositivo = f"GPU ({gpus[0].name})" if gpus else "CPU"

    print("=" * 50)
    print("  Modelo de Señales v2 (58 Clases)")
    print("=" * 50)
    print(f"  Backend:      TensorFlow {tf.__version__}")
    print(f"  Dispositivo:  {dispositivo}")
    print(f"  Modelo:       {model_path}")
    print(f"  Forma entrada:{model.input_shape}")
    print(f"  Clases:       {num_classes}")
    print(f"  Parámetros:   {model.count_params():,}")
    print("=" * 50)

    return model, class_descriptions, num_classes


def predict_sign(model, frame, num_classes):
    """Predice la clase de señal de tráfico a partir de un fotograma de cámara.

    Retorna:
        class_idx  : Índice de la clase predicha
        confidence : Probabilidad máxima (confianza) entre todas las clases
        entropy    : Entropía de la distribución de probabilidad
                     - Entropía alta -> probabilidad dispersa -> probablemente fondo
                     - Entropía baja -> predicción concentrada -> probable señal real
    """
    # Fotograma vacío o inválido: devolver valores nulos
    if frame is None or frame.size == 0:
        return -1, 0.0, 999.0

    # Redimensionar al tamaño de entrada esperado por el modelo y normalizar [0, 1]
    _, h, w, _ = model.input_shape
    img = cv2.resize(frame, (w, h))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # Agregar dimensión de batch: (1, H, W, 3)

    probs      = model.predict(img, verbose=0)[0]
    class_idx  = int(np.argmax(probs))
    confidence = float(np.max(probs))

    # Entropía de Shannon: H = -sum(p * log(p))
    # Se recorta a 1e-10 para evitar log(0) = -inf
    probs_clipped = np.clip(probs, 1e-10, 1.0)
    entropy = float(-np.sum(probs_clipped * np.log(probs_clipped)))

    return class_idx, confidence, entropy


def get_image(camera):
    """Convierte la imagen cruda de la cámara Webots a un array NumPy en formato BGRA."""
    raw_image = camera.getImage()
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image


def display_image(display, image):
    """Envía una imagen OpenCV al Display de Webots.

    Si la imagen es en escala de grises (2D), la convierte a RGB replicando el canal.
    """
    # Webots Display requiere formato RGB; duplicar canal si la imagen es monocromática
    if len(image.shape) == 2:
        image_rgb = np.dstack((image, image, image))
    else:
        image_rgb = image

    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    # Pegar en posición (0,0) sin transparencia
    display.imagePaste(image_ref, 0, 0, False)


def main():
    # Estado inicial del vehículo
    speed      = 0    # Velocidad de crucero actual (km/h)
    angle      = 0.0  # Ángulo de dirección actual (radianes)
    last_press = {}   # Diccionario {tecla: tiempo_ultima_pulsacion} para debounce

    model, class_descriptions, num_classes = load_model()

    # Carpeta donde se guardan las imágenes de señales detectadas
    img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img_detectadas_v2")
    os.makedirs(img_dir, exist_ok=True)

    # Inicializar el driver de Webots (Driver hereda de Car que hereda de Robot)
    driver    = Driver()
    timestep  = int(driver.getBasicTimeStep())

    # Habilitar cámara, display y teclado con el mismo timestep de la simulación
    camera    = driver.getDevice("camera")
    camera.enable(timestep)
    display_img = Display("display_image")
    keyboard    = Keyboard()
    keyboard.enable(timestep)

    # -------------------------------------------------------------------------
    # Estado del sistema de detección
    # -------------------------------------------------------------------------
    sign_class          = -1     # Última clase predicha (-1 = sin detección)
    sign_conf           = 0.0    # Confianza de la última predicción [0, 1]
    sign_entropy        = 999.0  # Entropía de la última predicción (999 = sin datos)
    signs_detectadas    = set()  # Clases únicas confirmadas en esta sesión
    total_detecciones   = 0      # Contador total de detecciones (incluyendo repetidas)
    last_seen_time      = {}     # {clase: timestamp} para controlar el cooldown por clase
    frame_count         = 0      # Contador de fotogramas para ejecutar inferencia cada N frames
    last_detected_frame = None   # Copia del fotograma en el que se confirmó la última señal

    # Confirmación multi-fotograma: se requieren FRAMES_REQUERIDOS frames consecutivos
    # de la misma clase antes de aceptar la detección como válida. Esto evita que un
    # único fotograma con ruido dispare una detección falsa.
    candidate_class = -1  # Clase que está acumulando frames consecutivos
    candidate_count = 0   # Cuántos frames consecutivos lleva el candidato actual

    # Dimensiones del display para calcular posiciones del HUD
    dw = display_img.getWidth()
    dh = display_img.getHeight()

    print(f"\n  Carpeta de salida: {img_dir}")
    print(f"  Meta: Detectar {META_SENALES}+ de {TOTAL_SENALES} señales únicas")
    print("  Controles: Flechas=conducir, Espacio=freno, A=captura\n")

    # =========================================================================
    # BUCLE PRINCIPAL DE SIMULACIÓN
    # driver.step() avanza la simulación un timestep; retorna -1 al terminar
    # =========================================================================
    while driver.step() != -1:

        # Capturar fotograma de cámara y convertir de BGRA a RGB para OpenCV/modelo
        image     = get_image(camera)
        cam_frame = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        # =====================================================================
        # BLOQUE DE INFERENCIA
        # Solo se ejecuta cada INFER_EVERY fotogramas para reducir la carga
        # computacional sin perder capacidad de detección a velocidades normales
        # =====================================================================
        frame_count += 1
        if frame_count % INFER_EVERY == 0:
            sign_class, sign_conf, sign_entropy = predict_sign(model, cam_frame, num_classes)

            # Normalizar clases duplicadas de límites de velocidad a su clase canónica
            sign_class = SPEED_LIMIT_CANONICAL.get(sign_class, sign_class)

            # Filtro doble:
            #   1. Confianza: descarta predicciones débiles (clase poco probable)
            #   2. Entropía:  descarta fotogramas de fondo donde la probabilidad
            #                 está distribuida uniformemente entre muchas clases
            is_valid = sign_conf >= CONF_UMBRAL and sign_entropy <= ENTROPY_MAX

            if is_valid:
                if sign_class == candidate_class:
                    # Misma clase que el fotograma anterior: acumular contador
                    candidate_count += 1
                else:
                    # Clase nueva: reiniciar el contador para esta nueva candidata
                    candidate_class = sign_class
                    candidate_count = 1

                # Una vez alcanzado el mínimo de frames consecutivos, confirmar
                if candidate_count >= FRAMES_REQUERIDOS:
                    ahora = time.time()
                    # Cooldown por clase: evita contar varias veces la misma señal
                    # mientras el vehículo pasa lentamente frente a ella
                    tiempo_desde_ultima = ahora - last_seen_time.get(sign_class, 0)
                    if tiempo_desde_ultima >= COOLDOWN_SENALES:
                        signs_detectadas.add(sign_class)
                        total_detecciones += 1
                        last_seen_time[sign_class] = ahora
                        last_detected_frame = cam_frame.copy()

                        # Guardar imagen: convertir de RGB a BGR porque cv2.imwrite espera BGR
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        fname = os.path.join(img_dir, f"clase{sign_class}_{timestamp}.png")
                        cv2.imwrite(fname, cv2.cvtColor(cam_frame, cv2.COLOR_RGB2BGR))

                        desc = class_descriptions.get(str(sign_class), f"Clase {sign_class}")
                        print(f"  [DETECTADA] Clase {sign_class}: {desc}")
                        print(f"              Guardada: {fname}")

                    # Reiniciar contador tras el intento de confirmación (exitoso o bloqueado por cooldown)
                    candidate_count = 0
            else:
                # Fotograma rechazado por el filtro: limpiar candidato para no
                # acumular frames de distintas clases (o fondo) como si fueran consecutivos
                candidate_class = -1
                candidate_count = 0

        # =====================================================================
        # Heads-Up Display
        # Se dibuja sobre una imagen negra y se envía al Display de Webots.
        #
        # Columna izquierda (telemetría):
        #   - Velocidad actual en km/h (azul si va en reversa)
        #   - Ángulo de dirección en radianes
        #   - Versión del modelo y número de clases
        #   - Clase detectada y si pasó o no el filtro de confianza/entropía
        #   - Confianza de la predicción (%) y entropía de Shannon
        #   - Barra de confianza con línea amarilla marcando el umbral mínimo
        #   - Contador de señales únicas detectadas vs. total, y meta del 50%
        #   - Barra de progreso con marcador en el 50%
        #   - Historial de clases ya detectadas (numeradas, en verde)
        #
        # Columna derecha:
        #   - Miniatura de la última señal detectada con borde cian
        # =====================================================================
        hud    = np.zeros((dh, dw, 3), dtype=np.uint8)
        fuente = cv2.FONT_HERSHEY_SIMPLEX
        col    = dw // 2  # Columna central que divide telemetría y miniatura

        # -- Miniatura de la última señal detectada (esquina superior derecha) --
        if last_detected_frame is not None:
            thumb_w = dw // 2
            thumb_h = int(thumb_w * last_detected_frame.shape[0] / last_detected_frame.shape[1])
            thumb   = cv2.resize(last_detected_frame, (thumb_w, thumb_h))
            hud[0:thumb_h, dw - thumb_w:dw] = thumb
            # Borde naranja alrededor de la miniatura
            cv2.rectangle(hud, (dw - thumb_w, 0), (dw - 1, thumb_h - 1), (0, 180, 255), 1)

        # -- Telemetría del vehículo --
        # Velocidad en rojo/azul si va en reversa, blanco si avanza
        vel_color = (0, 80, 255) if speed < 0 else (255, 255, 255)
        vel_label = f"Vel:  {speed} km/h  [R]" if speed < 0 else f"Vel:  {speed} km/h"
        cv2.putText(hud, vel_label,                        (5, 20),  fuente, 0.45, vel_color,       1)
        cv2.putText(hud, f"Giro: {angle:.2f} rad",        (5, 38),  fuente, 0.45, (255, 255, 255), 1)
        cv2.putText(hud, f"Modelo: v2 ({num_classes} cls)",(5, 56),  fuente, 0.40, (100, 100, 100), 1)

        cv2.line(hud, (0, 65), (col, 65), (60, 60, 60), 1)  # Separador horizontal

        # -- Información de la detección actual --
        # is_valid refleja el valor del último fotograma de inferencia (persiste entre frames)
        if sign_class >= 0:
            # Cian si pasa el filtro; gris si fue rechazada
            color_senal = (0, 255, 255) if is_valid else (120, 120, 120)
            status      = "" if is_valid else " [REC]"  # REC = rechazada
            cv2.putText(hud, f"Senal: {sign_class}{status}", (5, 82),      fuente, 0.45, color_senal,    1)
            cv2.putText(hud, f"Conf:  {sign_conf*100:.1f}%", (5, 100),     fuente, 0.45, color_senal,    1)
            # Entropía: verde si está bajo el umbral (señal probable), rojo si está alta (fondo)
            ent_color = (0, 255, 100) if sign_entropy <= ENTROPY_MAX else (255, 100, 100)
            cv2.putText(hud, f"Entr:  {sign_entropy:.2f}",   (col//2, 100), fuente, 0.40, ent_color,     1)

            # Barra de confianza con marcador en la posición del umbral
            bar_x, bar_y, bar_h = 5, 105, 6
            bar_max_w = col - 10
            bar_fill  = int(bar_max_w * sign_conf)
            cv2.rectangle(hud, (bar_x, bar_y), (bar_x + bar_max_w, bar_y + bar_h), (50, 50, 50),  -1)
            cv2.rectangle(hud, (bar_x, bar_y), (bar_x + bar_fill,  bar_y + bar_h), color_senal,   -1)
            # Línea amarilla vertical que indica dónde está el umbral de confianza
            umbral_x = bar_x + int(bar_max_w * CONF_UMBRAL)
            cv2.line(hud, (umbral_x, bar_y - 1), (umbral_x, bar_y + bar_h + 1), (255, 255, 0), 1)
        else:
            # Sin detección activa
            cv2.putText(hud, "Senal: --", (5, 82), fuente, 0.45, (80, 80, 80), 1)

        cv2.line(hud, (0, 118), (col, 118), (60, 60, 60), 1)  # Separador

        # -- Progreso hacia la meta --
        unicas     = len(signs_detectadas)
        # Verde si ya se alcanzó la meta; naranja-azul si aún no
        meta_color = (0, 255, 0) if unicas >= META_SENALES else (0, 180, 255)
        cv2.putText(hud, f"Unicas: {unicas}/{TOTAL_SENALES}", (5, 134), fuente, 0.45, meta_color,    1)
        cv2.putText(hud, f"Meta 50%: {META_SENALES}",          (5, 152), fuente, 0.40, (100,100,100), 1)

        # Barra de progreso; el marcador amarillo indica el punto del 50% (meta)
        prog_x, prog_y, prog_h = 5, 158, 8
        prog_max_w = col - 10
        prog_fill  = int(prog_max_w * min(unicas / TOTAL_SENALES, 1.0))
        cv2.rectangle(hud, (prog_x, prog_y), (prog_x + prog_max_w, prog_y + prog_h), (50, 50, 50), -1)
        cv2.rectangle(hud, (prog_x, prog_y), (prog_x + prog_fill,  prog_y + prog_h), meta_color,   -1)
        meta_x = prog_x + prog_max_w // 2
        cv2.line(hud, (meta_x, prog_y - 2), (meta_x, prog_y + prog_h + 2), (255, 255, 0), 1)

        # -- Historial de señales detectadas --
        cv2.line(hud, (0, 172), (col, 172), (60, 60, 60), 1)
        cv2.putText(hud, "Historial:", (5, 186), fuente, 0.40, (150, 150, 150), 1)
        y_hist = 200
        for clase in sorted(signs_detectadas):
            if y_hist > dh - 12:
                # No caben más entradas; indicar que hay más con "..."
                cv2.putText(hud, "...", (5, y_hist), fuente, 0.35, (100, 100, 100), 1)
                break
            cv2.putText(hud, f"  Clase {clase}", (5, y_hist), fuente, 0.35, (0, 220, 120), 1)
            y_hist += 14

        # Enviar el HUD al display de Webots
        display_image(display_img, hud)

        # =====================================================================
        # MANEJO DE TECLADO
        # =====================================================================
        current_time = time.time()
        key          = keyboard.getKey()

        # Dirección: aplicar ángulo fijo mientras la tecla esté presionada;
        # al soltar, volver al centro (0.0 radianes)
        if key == keyboard.LEFT:
            angle = -MAX_ANGLE
        elif key == keyboard.RIGHT:
            angle = MAX_ANGLE
        else:
            angle = 0.0

        # Freno de emergencia: detener el vehículo inmediatamente (sin debounce)
        if key == ord(' '):
            speed = 0

        # Velocidad y captura manual: con debounce para evitar incrementos múltiples
        # por mantener la tecla presionada
        if key not in last_press or (current_time - last_press.get(key, 0) >= DEBOUNCE_TIME):
            if key == keyboard.UP:
                if speed < MAX_SPEED:
                    speed += SPEED_INCR
                last_press[key] = current_time
            elif key == keyboard.DOWN:
                if speed > MAX_REVERSE:
                    speed -= SPEED_INCR
                last_press[key] = current_time
            elif key == ord('A'):
                # Captura manual: útil para documentar el entorno o depurar detecciones
                current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                file_name = os.path.join(img_dir, f"manual_{current_datetime}.png")
                cv2.imwrite(file_name, cv2.cvtColor(cam_frame, cv2.COLOR_RGB2BGR))
                print(f"  [MANUAL] Captura guardada: {file_name}")
                last_press[key] = current_time

        # Aplicar controles al vehículo en Webots
        driver.setSteeringAngle(angle)
        driver.setCruisingSpeed(speed)


if __name__ == "__main__":
    main()
