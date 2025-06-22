# Importa las bibliotecas necesarias
import serial  # Para comunicación con el puerto serial (LiDAR y ESP32)
from CalcLidarData import CalcLidarData  # Función personalizada para interpretar los datos del LiDAR
import math  # Para convertir entre radianes y grados
import time
import sys # Para salir del programa de forma limpia
import threading # Para la implementación de hilos

# Importaciones específicas para el Control (YOLO + DepthAI)
import cv2
import depthai as dai
import numpy as np
from ultralytics import YOLO

# ---
## Variables Globales de Sincronización y Estado del Sistema

# Evento para señalizar a todos los hilos que deben detenerse
stop_event = threading.Event()
# Bandera para indicar si la alarma está activa.
alarm_active = False
# Bloqueo para proteger el acceso a 'alarm_active' y otras variables compartidas
alarm_lock = threading.Lock()

# ---
## Configuración Serial ESP32 (Control de Motores, Relevador y Lectura de Botón)

# Define el puerto COM al que está conectada la ESP32
esp32_com_port = "COM3"
esp32_baud_rate = 115200

# Objeto serial de la ESP32 (global, compartido entre hilos que lo necesiten)
ser_esp32_global = None

try:
    # Intenta establecer la conexión serial con la ESP32 al inicio del programa principal
    ser_esp32_global = serial.Serial(esp32_com_port, esp32_baud_rate, timeout=0.1)
    time.sleep(2)  # Dar tiempo a la ESP32 para inicializarse completamente
    print(f"Main: Conexión serial con ESP32 en {esp32_com_port} establecida.")
except serial.SerialException as e:
    print(f"Main: Error al abrir el puerto serial {esp32_com_port} para ESP32: {e}")
    print("Main: Asegúrate de que la ESP32 esté conectada y que el puerto COM sea correcto.")
    sys.exit(1) # Salir del programa si no se puede establecer la conexión con la ESP32

# ---
## Configuración del LiDAR

# Define el puerto COM al que está conectado el LiDAR
lidar_com_port = "COM4"

# Objeto serial del LiDAR (gestionado por el hilo del LiDAR)
ser_lidar = None

# Función para abrir la conexión serial con el LiDAR
def open_lidar_serial():
    global ser_lidar
    if ser_lidar and ser_lidar.is_open:
        return True # Ya está abierto
    try:
        ser_lidar = serial.Serial(
            port=lidar_com_port,
            baudrate=230400,
            timeout=0.1, 
            bytesize=8,
            parity='N',
            stopbits=1
        )
        print(f"LiDAR Func: Comunicación serial con LiDAR en {lidar_com_port} establecida/reestablecida.")
        return True
    except serial.SerialException as e:
        print(f"LiDAR Func: Error al abrir/reabrir el puerto serial del LiDAR {lidar_com_port}: {e}")
        ser_lidar = None
        return False

# Función para cerrar la conexión serial con el LiDAR
def close_lidar_serial():
    global ser_lidar
    if ser_lidar and ser_lidar.is_open:
        try:
            ser_lidar.close()
            print("LiDAR Func: Comunicación serial del LiDAR CERRADA.")
        except Exception as e:
            print(f"LiDAR Func: Error al cerrar el puerto serial del LiDAR: {e}")
        finally:
            ser_lidar = None
    # else:
        # print("LiDAR Func: Comunicación serial del LiDAR ya está cerrada.") # Descomentar para depurar

# ---
## Configuración del Subsistema de Control (YOLO + DepthAI)

# Ruta al modelo YOLOv8 (asegúrate de que esta ruta sea correcta en tu sistema)
MODELO_PATH = "C:/Users/JohnChris/Desktop/TT/YoloV8n_ArUco.pt"
try:
    modelo = YOLO(MODELO_PATH) # Carga el modelo de detección de objetos YOLOv8
except Exception as e:
    print(f"Main: Error al cargar el modelo YOLOv8 desde {MODELO_PATH}: {e}")
    sys.exit(1) # Termina el programa si no se puede cargar el modelo

# Variables globales para el dispositivo DepthAI y sus colas (gestionadas por el hilo de la cámara)
device = None
cola_rgb = None
cola_depth = None

# Función para configurar y abrir el sistema de cámara DepthAI
def open_camera_system():
    global device, cola_rgb, cola_depth
    if device is not None: # Si ya está abierto, no hacer nada
        return True
    try:
        # El pipeline debe crearse cada vez que se abre el dispositivo
        pipeline = dai.Pipeline()

        # Configuración de las cámaras y el nodo de profundidad
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setInterleaved(False)
        cam_rgb.setIspScale(1, 1)

        cam_left = pipeline.create(dai.node.MonoCamera)
        cam_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

        cam_right = pipeline.create(dai.node.MonoCamera)
        cam_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setConfidenceThreshold(200)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        cam_left.out.link(stereo.left)
        cam_right.out.link(stereo.right)

        # Configuración de las salidas de datos
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam_rgb.video.link(xout_rgb.input)

        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)

        device = dai.Device(pipeline) # Abre el dispositivo DepthAI
        cola_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
        cola_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)
        print("Camera Func: Sistema de cámara DepthAI activado/reactivado.")
        return True
    except RuntimeError as e:
        print(f"Camera Func: Error al activar el sistema de cámara DepthAI: {e}")
        device = None
        cola_rgb = None
        cola_depth = None
        return False

# Función para cerrar el sistema de cámara DepthAI
def close_camera_system():
    global device, cola_rgb, cola_depth
    if device is not None:
        try:
            device.close() # Cierra el dispositivo DepthAI
            print("Camera Func: Sistema de cámara DepthAI cerrado.")
        except Exception as e:
            print(f"Camera Func: Error al cerrar el sistema de cámara DepthAI: {e}")
        finally:
            device = None
            cola_rgb = None
            cola_depth = None
    # else:
        # print("Camera Func: Sistema de cámara DepthAI ya está cerrado.") # Descomentar para depurar

# Parámetros de control PID (utilizados y reiniciados en el hilo de control)
distancia_objetivo = 1.75
Kp_dist = 150.0
Ki_dist = 30.0
Kd_dist = 50.0

Kp_giro_izquierdo = 100
Kd_giro_izquierdo = 250
Ki_giro_izquierdo = 0

Kp_giro_derecho = 100
Kd_giro_derecho = 230
Ki_giro_derecho = 0

# Variables PID globales para que el hilo de control pueda acceder a ellas
# Se reiniciarán dentro del hilo de control al salir del estado de alarma.
_error_giro_anterior = 0.0
_error_dist_anterior = 0.0
_integral_error = 0.0
_integral_giro = 0.0
_tiempo_anterior = 0.0

# ---
## Hilo de Procesamiento del LiDAR

def lidar_thread_function(esp32_serial_port):
    global ser_lidar, alarm_active

    tmpString_lidar = "" # Cadena temporal local para los datos del paquete LiDAR
    
    # Asegurarse de que el LiDAR esté abierto al iniciar el hilo
    if not open_lidar_serial():
        stop_event.set() # Si el LiDAR falla al iniciar, señalizar para detener todo el programa
        print("LiDAR Thread: Error crítico al abrir el LiDAR. Deteniendo sistema.")
        return

    print("LiDAR Thread: Iniciado y esperando datos.")

    while not stop_event.is_set(): # Bucle principal del hilo LiDAR
        current_alarm_state = False
        with alarm_lock: # Acceso seguro a la bandera de alarma compartida
            current_alarm_state = alarm_active

        # Manejo del estado del botón de la ESP32 (lectura)
        button_state = ""
        try:
            if esp32_serial_port.in_waiting > 0:
                button_state = esp32_serial_port.readline().decode('utf-8').strip()
                # print(f"LiDAR Thread: Estado del botón de ESP32: {button_state}") # Descomentar para depurar
        except serial.SerialException as e:
            print(f"LiDAR Thread: Error al leer del puerto serial de la ESP32: {e}")
            time.sleep(0.5) # Pequeña pausa antes de reintentar
            continue # Continuar para intentar leer de nuevo

        if current_alarm_state: # SI LA ALARMA ESTÁ ACTIVA
            # Asegurarse de que el relevador esté ENCENDIDO (el hilo de control detendrá los motores)
            try:
                esp32_serial_port.write(b"1\n")
            except serial.SerialException as e:
                print(f"LiDAR Thread: Error al enviar '1' a ESP32 durante alarma: {e}")

            # Lógica para desactivar la alarma con el botón
            if button_state == "BUTTON_PRESSED":
                print("LiDAR Thread: ¡Botón de desactivación de alarma presionado!")
                with alarm_lock: # Proteger el cambio de estado de la alarma
                    alarm_active = False # Desactivar la alarma
                try:
                    esp32_serial_port.write(b"0\n") # Apagar el relevador
                    print("LiDAR Thread: Señal '0' enviada a ESP32 (Relevador OFF - Alarma desactivada)")
                except serial.SerialException as e:
                    print(f"LiDAR Thread: Error al enviar '0' a ESP32 al desactivar alarma: {e}")
                esp32_serial_port.reset_input_buffer() # Limpiar el buffer para evitar relecturas

                # Reabrir la comunicación del LiDAR (el hilo de la cámara también reabrirá la cámara)
                if not open_lidar_serial():
                    stop_event.set() # Si el LiDAR no se puede reabrir, detener todo
                    print("LiDAR Thread: Error crítico al reabrir el LiDAR. Deteniendo sistema.")
                    return
            time.sleep(0.1) # Pequeña pausa para evitar el uso excesivo de CPU mientras espera el botón
            continue # Volver a la parte superior del bucle del hilo LiDAR

        # SI LA ALARMA NO ESTÁ ACTIVA, procesar datos del LiDAR
        # Asegurarse de que el puerto del LiDAR esté abierto para la lectura
        if ser_lidar is None or not ser_lidar.is_open:
            print("LiDAR Thread: Error: Puerto serial del LiDAR no está abierto. Intentando reabrir...")
            if not open_lidar_serial():
                time.sleep(1) # Esperar antes de reintentar si falla la reapertura
                continue # Saltar a la siguiente iteración del hilo LiDAR

        loopFlag = True
        flag2c = False
        tmpString_lidar = "" # Reiniciar para cada intento de paquete LiDAR

        # Bucle interno para leer un paquete completo del LiDAR
        while loopFlag and not stop_event.is_set():
            try:
                b = ser_lidar.read()  # Lee un byte
                tmpInt = int.from_bytes(b, 'big') # Convierte a entero
            except serial.SerialException as e:
                print(f"LiDAR Thread: Error de lectura serial del LiDAR: {e}")
                close_lidar_serial() # Cerrar el puerto si hay un error
                break # Salir del bucle interno
            except TypeError: # Esto ocurre si 'b' está vacío (timeout)
                break # Salir del bucle interno si no hay datos

            if tmpInt == 0x54:  # Primer byte del encabezado
                tmpString_lidar += b.hex() + " "
                flag2c = True
                continue
            elif tmpInt == 0x2c and flag2c:  # Segundo byte del encabezado
                tmpString_lidar += b.hex()
                if not len(tmpString_lidar[0:-5].replace(' ', '')) == 90: # Validar longitud
                    tmpString_lidar = ""
                    loopFlag = False
                    flag2c = False
                    continue

                lidarData = CalcLidarData(tmpString_lidar[0:-5]) # Procesar datos
                objeto_lidar_detectado = False
                for angle_rad, distance in zip(lidarData.Angle_i, lidarData.Distance_i):
                    angle_deg = math.degrees(angle_rad)
                    angle_deg = angle_deg if angle_deg >= 0 else angle_deg + 360

                    # Sectores de detección de obstáculos
                    if (180 <= angle_deg <= 240 and 0.1 <= distance <= 0.6) or \
                       (300 <= angle_deg <= 360 and 0.1 <= distance <= 0.6) or \
                       (241 <= angle_deg <= 299 and 0.1 <= distance <= 1.0):
                        print(f"LiDAR Thread: !!! OBSTÁCULO DETECTADO por LiDAR en {angle_deg:.1f}° a {distance:.2f}m !!!")
                        objeto_lidar_detectado = True
                        break # Salir del bucle 'for' al detectar el primer objeto
                
                if objeto_lidar_detectado:
                    with alarm_lock: # Proteger el cambio de estado de la alarma
                        if not alarm_active: # Solo activar si no estaba ya activa
                            alarm_active = True
                            try:
                                esp32_serial_port.write(b"1\n") # Activar relevador
                                print("LiDAR Thread: Señal '1' enviada a ESP32 (Relevador ON - ALARMA ACTIVADA)")
                            except serial.SerialException as e:
                                print(f"LiDAR Thread: Error al enviar '1' a ESP32: {e}")
                            close_lidar_serial() # Cerrar el LiDAR al activar la alarma
                else: # No se detectó objeto, asegurar que el relevador esté APAGADO si la alarma no está activa
                    if not current_alarm_state: # Solo apagar si NO hay alarma activa
                        try:
                            esp32_serial_port.write(b"0\n")
                            # print("LiDAR Thread: Señal '0' enviada a ESP32 (Relevador OFF - Despejado)") # Descomentar para depurar
                        except serial.SerialException as e:
                            print(f"LiDAR Thread: Error al enviar '0' a ESP32 (despejado): {e}")

                tmpString_lidar = "" # Reiniciar la cadena temporal
                loopFlag = False # Salir del bucle interno para procesar el siguiente paquete
            else:
                tmpString_lidar += b.hex() + " " # Acumular bytes
            flag2c = False # Reiniciar bandera de encabezado
    print("LiDAR Thread: Terminando.")
    close_lidar_serial() # Asegurar cierre al terminar el hilo

# ---
## Hilo de Control de la Cámara DepthAI

def camera_thread_function():
    global device, cola_rgb, cola_depth, alarm_active

    # Asegurarse de que la cámara esté abierta al iniciar el hilo
    if not open_camera_system():
        stop_event.set() # Si la cámara falla al iniciar, señalizar para detener todo
        print("Camera Thread: Error crítico al abrir la cámara. Deteniendo sistema.")
        return

    print("Camera Thread: Iniciado.")

    while not stop_event.is_set():
        current_alarm_state = False
        with alarm_lock:
            current_alarm_state = alarm_active

        if current_alarm_state: # SI LA ALARMA ESTÁ ACTIVA, cerrar la cámara
            if device is not None:
                close_camera_system() # Cierra la cámara si está abierta
            
            # Muestra un frame en blanco con mensaje de alarma activa
            blank_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(blank_frame, "!!! ALARMA ACTIVA !!!", (200, 350),
                                 cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
            cv2.putText(blank_frame, "Presione el boton para desactivar", (100, 450),
                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("YOLOv8 + Profundidad + Lidar", blank_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set() # Señalizar para detener todos los hilos
            
            time.sleep(0.1) # Pausa para evitar bucle de espera ocupado
            continue # Ir a la siguiente iteración del bucle de la cámara

        # SI LA ALARMA NO ESTÁ ACTIVA, asegurar que la cámara esté abierta y procesar frames
        if device is None: # Si la cámara no está abierta despues de una alarma
            if not open_camera_system(): # Intentar reabrirla
                print("Camera Thread: No se pudo reabrir la cámara. Intentando de nuevo...")
                time.sleep(0.5) # Pausa antes de reintentar
                continue # Saltar a la siguiente iteración

        # Intentar obtener frames
        try:
            in_rgb = cola_rgb.tryGet()
            in_depth = cola_depth.tryGet()
        except RuntimeError as e:
            # Captura errores de comunicación del DepthAI en este hilo
            print(f"Camera Thread: RuntimeError during frame acquisition: {e}")
            close_camera_system() # Intenta cerrar y forzar una reapertura en el siguiente ciclo
            time.sleep(0.5)
            continue

        if in_rgb is None or in_depth is None:
            # print("Camera Thread: No frames available.") # Descomentar para depurar
            time.sleep(0.01) # Pequeña pausa
            continue

        frame_rgb = in_rgb.getCvFrame() # Convertir a formato OpenCV
        frame_depth = in_depth.getFrame() # Obtener mapa de profundidad

        # Realizar inferencia con YOLOv8 para dibujar en el frame
        input_frame = cv2.resize(frame_rgb, (640, 360))
        resultados = modelo.predict(
            source=input_frame,
            conf=0.8, iou=0.3, imgsz=640, verbose=False
        )
        frame_dibujado = resultados[0].plot() # Dibuja las detecciones en el frame

        # Dibuja la información de control PID en el frame 
        HFOV = 69.0
        ancho_imagen = frame_rgb.shape[1]
        
        # Obtener datos de PID. 
        display_dist = 0.0
        display_error_giro = 0.0

        detecciones_para_display = resultados[0].boxes
        if detecciones_para_display is not None and len(detecciones_para_display) > 0:
            confianzas_display = detecciones_para_display.conf
            indice_mejor_display = confianzas_display.argmax().item()
            det_display = detecciones_para_display.xyxy[indice_mejor_display]

            x1_d, y1_d, x2_d, y2_d = map(int, det_display[:4])
            cx_d = int((x1_d + x2_d) / 2)
            cy_d = int((y1_d + y2_d) / 2)

            escala_x_d = frame_rgb.shape[1] / 640
            escala_y_d = frame_rgb.shape[0] / 360
            cx_d = int(cx_d * escala_x_d)
            cy_d = int(cy_d * escala_y_d)
            x1_d = int(x1_d * escala_x_d)
            y1_d = int(y1_d * escala_y_d)

            if 0 <= cx_d < frame_depth.shape[1] and 0 <= cy_d < frame_depth.shape[0]:
                depth_mm_d = frame_depth[cy_d, cx_d]
                if 1500 < depth_mm_d < 8000:
                    display_dist = depth_mm_d / 1000.0
                    centro_x_imagen_d = frame_rgb.shape[1] // 2
                    error_giro_px_d = centro_x_imagen_d - cx_d
                    display_error_giro = (error_giro_px_d / ancho_imagen) * HFOV

        # Visualización de información en el frame
        texto_dist = f"Distancia: {display_dist:.2f} m"
        texto_error_giro = f"Error Giro: {display_error_giro:.2f}°"
        texto_error_distancia = f"Error Distancia: {display_error_giro:.2f}°"
        # Usamos variables PID globales para la visualización
        texto_vel = f"VI: {int(np.clip(_error_giro_anterior, -130, 130))} | VD: {int(np.clip(_error_dist_anterior, -130, 130))}"

        cv2.putText(frame_dibujado, texto_dist, (x1_d if 'x1_d' in locals() else 50, y1_d-10 if 'y1_d' in locals() else 50),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame_dibujado, texto_error_giro, (x1_d if 'x1_d' in locals() else 50, y2_d+25 if 'y2_d' in locals() else 80),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2)
        cv2.putText(frame_dibujado, texto_error_distancia, (x1_d if 'x1_d' in locals() else 50, y2_d+25 if 'y2_d' in locals() else 80),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 50, 0), 2)
        cv2.putText(frame_dibujado, texto_vel, (x1_d if 'x1_d' in locals() else 50, y2_d+50 if 'y2_d' in locals() else 110),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        frame_mostrar = cv2.resize(frame_dibujado, (1280, 720)) # Redimensionar para visualización
        cv2.imshow("YOLOv8 + Profundidad + Lidar", frame_mostrar) # MOSTRAR VENTANA
        
        if cv2.waitKey(1) & 0xFF == ord('q'): # Permitir salir con 'q'
            stop_event.set() # Señalizar para detener todos los hilos
    
    print("Camera Thread: Terminando.")
    close_camera_system() # Asegurar cierre de cámara al terminar el hilo

# ---
## Hilo de Control (Visión y Motores)

def control_thread_function(esp32_serial_port):
    global _error_giro_anterior, _error_dist_anterior, _integral_error, _integral_giro, _tiempo_anterior
    global alarm_active # Para leer el estado de la alarma

    # Esta bandera controla el reinicio de los PID cuando la alarma se desactiva
    prev_alarm_state_for_pid_reset = True 

    print("Control Thread: Iniciado.")

    while not stop_event.is_set(): # Bucle principal del hilo de control
        current_alarm_state = False
        with alarm_lock: # Acceso seguro a la bandera de alarma
            current_alarm_state = alarm_active

        # Reiniciar variables PID si la alarma acaba de desactivarse
        if prev_alarm_state_for_pid_reset and not current_alarm_state:
            print("Control Thread: Alarma desactivada, reiniciando variables PID.")
            _error_giro_anterior = 0.0
            _error_dist_anterior = 0.0
            _integral_error = 0.0
            _integral_giro = 0.0
            _tiempo_anterior = time.time() # Restablece el tiempo para un cálculo correcto del 'dt'
            # Añadir un pequeño retardo para permitir que el hardware de la cámara se reinicie
            time.sleep(1.0) # Espera 1 segundo para la cámara
            prev_alarm_state_for_pid_reset = False

        prev_alarm_state_for_pid_reset = current_alarm_state # Actualizar el estado anterior para el siguiente ciclo

        if current_alarm_state: # SI LA ALARMA ESTÁ ACTIVA
            try:
                esp32_serial_port.write(b"0,0\n") # Detener motores (hilo LiDAR maneja el relevador)
            except serial.SerialException as e:
                print(f"Control Thread: Error al enviar '0,0' a ESP32 durante alarma: {e}")
    
            time.sleep(0.1) # Pequeña pausa
            continue # Saltar el resto del bucle de control

        # SI LA ALARMA NO ESTÁ ACTIVA, realizar la lógica de control normal
        if device is None or cola_rgb is None or cola_depth is None:
            print("Control Thread: Sistema de cámara inactivo o no inicializado. Deteniendo motores.")
            try:
                esp32_serial_port.write(b"0,0\n") # Asegura que los motores estén detenidos
            except serial.SerialException as e:
                print(f"Control Thread: Error al enviar '0,0' a ESP32 por cámara inactiva: {e}")
            time.sleep(0.01)
            continue # Saltar a la siguiente iteración

        # Intentar obtener frames de las colas. Capturar RuntimeError por si falla la cámara.
        try:
            in_rgb = cola_rgb.tryGet() # Intenta obtener un frame RGB
            in_depth = cola_depth.tryGet() # Intenta obtener un frame de profundidad
        except RuntimeError as e:
            print(f"Control Thread: RuntimeError during camera frame acquisition: {e}")
            try:
                esp32_serial_port.write(b"0,0\n")
            except serial.SerialException as e_ser:
                print(f"Control Thread: Error sending stop command to ESP32: {e_ser}")
            time.sleep(0.1) # Dar una pausa antes de reintentar
            continue # Salta a la siguiente iteración del bucle de control

        if in_rgb is None or in_depth is None:
            try:
                esp32_serial_port.write(b"0,0\n") # Detener motores si no hay frames
            except serial.SerialException as e:
                print(f"Control Thread: Error al enviar '0,0' a ESP32 por falta de frames: {e}")
            time.sleep(0.01)
            continue

        frame_rgb = in_rgb.getCvFrame() # Convertir a formato OpenCV
        frame_depth = in_depth.getFrame() # Obtener mapa de profundidad

        HFOV = 69.0 # Campo de visión horizontal de la cámara
        ancho_imagen = frame_rgb.shape[1] # Ancho de la imagen
        input_frame = cv2.resize(frame_rgb, (640, 360)) # Redimensionar para YOLO

        # Realizar inferencia con YOLOv8
        resultados = modelo.predict(
            source=input_frame,
            conf=0.8, iou=0.3, imgsz=640, verbose=False
        )

        detecciones = resultados[0].boxes # Obtener cajas delimitadoras
        
        vel_izquierda = 0 # Velocidad por defecto
        vel_derecha = 0   # Velocidad por defecto

        if detecciones is not None and len(detecciones) > 0:
            confianzas = detecciones.conf
            indice_mejor = confianzas.argmax().item()
            det = detecciones.xyxy[indice_mejor]

            x1, y1, x2, y2 = map(int, det[:4])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            escala_x = frame_rgb.shape[1] / 640
            escala_y = frame_rgb.shape[0] / 360
            cx = int(cx * escala_x)
            cx_depth_mapping = int(cx * (frame_depth.shape[1] / frame_rgb.shape[1])) # Escalar cx para el frame de profundidad
            cy = int(cy * escala_y)
            cy_depth_mapping = int(cy * (frame_depth.shape[0] / frame_rgb.shape[0])) # Escalar cy para el frame de profundidad

            if 0 <= cx_depth_mapping < frame_depth.shape[1] and 0 <= cy_depth_mapping < frame_depth.shape[0]:
                depth_mm = frame_depth[cy_depth_mapping, cx_depth_mapping] # Profundidad en milímetros
                
                distancia_valida_para_control = 1750 < depth_mm < 8000 # Rango de control

                if distancia_valida_para_control:
                    tiempo_actual = time.time()
                    dt = tiempo_actual - _tiempo_anterior if _tiempo_anterior != 0.0 else 1.0
                    _tiempo_anterior = tiempo_actual # Actualiza _tiempo_anterior para el siguiente cálculo de 'dt'

                    centro_x_imagen = frame_rgb.shape[1] // 2
                    error_giro_px = centro_x_imagen - cx
                    error_giro_deg = (error_giro_px / ancho_imagen) * HFOV

                    if abs(error_giro_deg) > 5: # Priorizar giro si el error es grande
                        derivada_giro = (error_giro_deg - _error_giro_anterior) / dt
                        _error_giro_anterior = error_giro_deg
                        _integral_giro += error_giro_deg * dt

                        ajuste_giro_izquierdo = - (Kp_giro_izquierdo * error_giro_deg + Kd_giro_izquierdo * derivada_giro + Ki_giro_izquierdo * _integral_giro)
                        ajuste_giro_derecho = Kp_giro_derecho * error_giro_deg + Kd_giro_derecho * derivada_giro + Ki_giro_derecho * _integral_giro

                        vel_izquierda = int(np.clip(ajuste_giro_izquierdo, -130, 130))
                        vel_derecha = int(np.clip(ajuste_giro_derecho, -130, 130))
                    else: # Control de distancia
                        distancia_m = depth_mm / 1000.0
                        error_dist = distancia_m - distancia_objetivo
                        _integral_error += error_dist * dt
                        derivada_error = (error_dist - _error_dist_anterior) / dt if dt > 0 else 0

                        vel_lineal = (Kp_dist * error_dist + Ki_dist * _integral_error + Kd_dist * derivada_error)
                        vel_lineal = int(np.clip(vel_lineal, -130, 130))

                        _error_dist_anterior = error_dist

                        vel_izquierda = vel_derecha = vel_lineal
                    
                    # Información para imprimir en consola
                    texto_dist = f"Distancia: {depth_mm/1000.0:.2f} m"
                    texto_error_giro = f"Error Giro: {error_giro_deg:.2f}°"
                    texto_error_distancia = f"Error Distancia: {error_giro_deg:.2f}°"
                    texto_vel = f"VI: {vel_izquierda} | VD: {vel_derecha}"
                    print(f"Control Thread: {texto_dist} | {texto_error_giro} | {texto_error_distancia} | {texto_vel}")
                else: # Si la distancia del marcador está fuera del rango de control válido
                    vel_izquierda = vel_derecha = 0 # Detiene los motores
                    print("Control Thread: Distancia del marcador fuera de rango de control. Motores detenidos.")
            else: # Si el centro del marcador (cx, cy) está fuera de los límites del frame de profundidad
                vel_izquierda = vel_derecha = 0 # Detiene los motores
                print("Control Thread: Marcador detectado fuera del rango de profundidad. Motores detenidos.")
        else: # Si no se detectó ningún marcador en el frame
            vel_izquierda = vel_derecha = 0 # Detiene los motores
            # print("Control Thread: Ningún marcador detectado. Motores detenidos.") # Descomentar para depurar

        # Enviar las velocidades calculadas a la ESP32 (solo si la alarma NO está activa)
        try:
            esp32_serial_port.write(f"{vel_izquierda},{vel_derecha}\n".encode())
        except serial.SerialException as e:
            print(f"Control Thread: Error al enviar velocidades a ESP32: {e}")
        
        time.sleep(0.01) # Pequeña pausa para no saturar CPU

    print("Control Thread: Terminando.")

# ---
## Función Principal de Ejecución

if __name__ == "__main__":
    print("Main: Iniciando programa principal con hilos...")

    # Activar la alarma por 1 segundo al inicio del programa
    if ser_esp32_global:
        with alarm_lock:
            alarm_active = True
            try:
                ser_esp32_global.write(b"1\n") # Activa el relevador para la alarma
                print("Main: Alarma inicial activada por 1 segundo.")
            except serial.SerialException as e:
                print(f"Main: Error al activar la alarma inicial en ESP32: {e}")
        time.sleep(1) # Espera 1 segundo
        with alarm_lock:
            alarm_active = False
            try:
                ser_esp32_global.write(b"0\n") # Desactiva el relevador
                print("Main: Alarma inicial desactivada. Sistema operando con normalidad.")
            except serial.SerialException as e:
                print(f"Main: Error al desactivar la alarma inicial en ESP32: {e}")
    else:
        print("Main: No se pudo establecer comunicación con ESP32, omitiendo activación de alarma inicial.")

    # Crear los hilos, pasando el objeto serial de la ESP32
    lidar_thread = threading.Thread(target=lidar_thread_function, args=(ser_esp32_global,))
    camera_thread = threading.Thread(target=camera_thread_function)
    control_thread = threading.Thread(target=control_thread_function, args=(ser_esp32_global,))

    # Iniciar los hilos
    lidar_thread.start()
    camera_thread.start()
    control_thread.start()

    try:
        # Esperar a que los hilos terminen (o hasta que se presione Ctrl+C)
        lidar_thread.join()
        camera_thread.join()
        control_thread.join()
    except KeyboardInterrupt:
        print("\nMain: Señal de interrupción (Ctrl+C) recibida. Deteniendo todos los hilos...")
        stop_event.set() # Señaliza a todos los hilos para que se detengan

    # Esperar un tiempo prudente para que los hilos se detengan limpiamente
    print("Main: Esperando a que los hilos terminen...")
    lidar_thread.join(timeout=2)
    camera_thread.join(timeout=2)
    control_thread.join(timeout=2)

    # Asegurarse de que el puerto serial de la ESP32 esté cerrado al finalizar
    if ser_esp32_global and ser_esp32_global.is_open:
        try:
            ser_esp32_global.write(b"0,0\n") # Enviar comando de detener motores
            time.sleep(0.1)
            ser_esp32_global.write(b"0\n") # Asegurar que el relevador esté apagado
            time.sleep(0.1)
            ser_esp32_global.close()
            print("Main: Conexión serial con ESP32 CERRADA.")
        except serial.SerialException as e:
            print(f"Main: Error al cerrar el puerto serial de ESP32: {e}")
    
    # Destruir todas las ventanas de OpenCV
    cv2.destroyAllWindows()
    print("Main: Programa finalizado.")