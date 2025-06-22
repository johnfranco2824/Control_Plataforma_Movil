import math  # Importa la biblioteca matemática para usar funciones como pi

# Clase que encapsula los datos obtenidos del LiDAR
class LidarData:
    def __init__(self, FSA, LSA, CS, Speed, TimeStamp, Degree_angle, Angle_i, Distance_i):
        self.FSA = FSA                  # First Sample Angle (primer ángulo del paquete)
        self.LSA = LSA                  # Last Sample Angle (último ángulo del paquete)
        self.CS = CS                    # Checksum (verificación del paquete)
        self.Speed = Speed              # Velocidad de rotación del LiDAR
        self.TimeStamp = TimeStamp      # Marca de tiempo del paquete
        self.Degree_angle = Degree_angle  # Lista de ángulos en grados
        self.Angle_i = Angle_i          # Lista de ángulos en radianes
        self.Distance_i = Distance_i    # Lista de distancias medidas

# Función para calcular los datos del LiDAR a partir de una cadena hexadecimal
def CalcLidarData(str):
    str = str.replace(' ', '')  # Elimina los espacios de la cadena recibida

    # Extrae la velocidad de rotación (bytes 0-3), intercambiando el orden de bytes (little-endian)
    Speed = int(str[2:4] + str[0:2], 16) / 100

    # Extrae el FSA (First Sample Angle)
    FSA = float(int(str[6:8] + str[4:6], 16)) / 100

    # Extrae el LSA (Last Sample Angle)
    LSA = float(int(str[-8:-6] + str[-10:-8], 16)) / 100

    # Extrae la marca de tiempo del paquete
    TimeStamp = int(str[-4:-2] + str[-6:-4], 16)

    # Extrae el checksum del paquete (últimos 2 caracteres hexadecimales)
    CS = int(str[-2:], 16)

    # Inicializa listas para ángulos y distancias
    Degree_angle = []  # Ángulos en grados
    Angle_i = []       # Ángulos en radianes
    Distance_i = []    # Distancias en metros

    # Calcula el paso angular entre muestras (hay 12 muestras por paquete)
    if (LSA - FSA > 0):
        angleStep = float(LSA - FSA) / 12
    else:
        # Si se cruzó el ángulo 360° al rotar
        angleStep = float((LSA + 360) - FSA) / 12

    counter = 0  # Contador de muestras

    # Función para ajustar ángulos mayores o iguales a 360°
    circle = lambda deg: deg - 360 if deg >= 360 else deg

    # Procesa cada una de las 12 muestras del paquete (cada una ocupa 6 caracteres hexadecimales)
    for i in range(0, 6 * 12, 6):
        # Extrae la distancia (en mm) de cada muestra, convierte a metros
        distance = int(str[8+i+2 : 8+i+4] + str[8+i : 8+i+2], 16) / 1000

        # Calcula el ángulo correspondiente a esta muestra
        degree = circle(angleStep * counter + FSA)

        # Filtra los datos para que solo incluyan la mitad del círculo (de 180° a 360°)
        if 180 <= degree <= 360:
            Degree_angle.append(degree)
            Angle_i.append(degree * math.pi / 180.0)  # Convierte a radianes
            Distance_i.append(distance)  # Agrega la distancia

        counter += 1  # Incrementa el contador de muestra

    # Devuelve una instancia de la clase LidarData con todos los datos extraídos
    return LidarData(FSA, LSA, CS, Speed, TimeStamp, Degree_angle, Angle_i, Distance_i)
