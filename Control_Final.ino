#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_ADXL345_U.h>

// --- Definiciones para el Control de Motores y Relevador ---
#define PWM_IZQ 5    // Pin GPIO para el PWM del motor izquierdo
#define PWM_DER 18   // Pin GPIO para el PWM del motor derecho

#define CH_IZQ 0     // Canal PWM para el motor izquierdo
#define CH_DER 1     // Canal PWM para el motor derecho

#define RELE_PIN 27  // Pin GPIO al que está conectado el relevador (¡Este pin es compartido!)
#define BOTON_PIN 25 // Pin GPIO al que está conectado el botón

// Variable para almacenar el estado anterior del botón (para detectar flancos)
int lastButtonState = HIGH;

// --- Definiciones para el Acelerómetro ---
Adafruit_ADXL345_Unified accel = Adafruit_ADXL345_Unified(12345);

// Umbrales de inclinación en grados
const float UMBRAL_MAXIMO = 1.0;
const float UMBRAL_MINIMO = -1.0;

// --- Parámetros para el filtro promediador ---
const int NUM_MUESTRAS = 250;
float muestrasPitch[NUM_MUESTRAS];
int indiceMuestra = 0;
bool bufferLleno = false; // Bandera para saber cuándo el buffer tiene al menos NUM_MUESTRAS

void setup() {
  Serial.begin(115200); // Inicializa la comunicación serial

  // --- Configuración de Motores ---
  ledcSetup(CH_IZQ, 5000, 8);
  ledcAttachPin(PWM_IZQ, CH_IZQ);
  ledcSetup(CH_DER, 5000, 8);
  ledcAttachPin(PWM_DER, CH_DER);

  // --- Configuración de Relevador ---
  pinMode(RELE_PIN, OUTPUT);
  digitalWrite(RELE_PIN, LOW); // Asegura que el relevador esté inicialmente APAGADO

  // --- Configuración del Botón ---
  pinMode(BOTON_PIN, INPUT_PULLUP);

  // --- Inicialización del Acelerómetro ---
  if (!accel.begin()) {
    Serial.println("No se detectó el ADXL345. Verifica la conexión.");
    while (1); // Detiene la ejecución si el sensor no se encuentra
  }
  accel.setRange(ADXL345_RANGE_2_G); // ±2g para mayor sensibilidad
  Serial.println("Sensor ADXL345 listo.");

  // Inicializar el buffer de muestras a cero
  for (int i = 0; i < NUM_MUESTRAS; i++) {
    muestrasPitch[i] = 0.0;
  }

  Serial.println("ESP32 lista. Motores, relevador y acelerómetro configurados. Esperando señales.");
}

void loop() {
  // --- Control del Relevador y Motores desde Python ---
  bool releActivadoPorSerial = false; // Bandera para saber si el relé fue activado por serial

  if (Serial.available()) {
    String comando = Serial.readStringUntil('\n');
    comando.trim();

    if (comando == "1") {
      digitalWrite(RELE_PIN, HIGH);
      releActivadoPorSerial = true; // El relé fue activado por comando serial
      // Serial.println("Relevador ENCENDIDO por Serial"); // Descomentar para depuración
    } else if (comando == "0") {
      digitalWrite(RELE_PIN, LOW);
      releActivadoPorSerial = false; // El relé fue desactivado por comando serial
      // Serial.println("Relevador APAGADO por Serial"); // Descomentar para depuración
    } else {
      // Lógica de control de motores
      int commaIndex = comando.indexOf(',');
      if (commaIndex != -1) {
        String velIzquierdaStr = comando.substring(0, commaIndex);
        String velDerechaStr = comando.substring(commaIndex + 1);

        int velIzquierda = velIzquierdaStr.toInt();
        int velDerecha = velDerechaStr.toInt();

        velIzquierda = constrain(velIzquierda, 0, 255);
        velDerecha = constrain(velDerecha, 0, 255);

        ledcWrite(CH_IZQ, velIzquierda);
        ledcWrite(CH_DER, velDerecha);
      }
    }
  }

  // --- Control del Relevador por Acelerómetro ---
  // Solo se evalúa el acelerómetro si el relevador no fue activado por serial en esta iteración
  if (!releActivadoPorSerial) {
    sensors_event_t event;
    accel.getEvent(&event);

    float x = event.acceleration.x;
    float y = event.acceleration.y;
    float z = event.acceleration.z;

    float pitchActual = atan2(x, sqrt(y * y + z * z)) * 180.0 / PI;

    // Almacenar la nueva muestra en el buffer
    muestrasPitch[indiceMuestra] = pitchActual; // Usamos el valor absoluto del pitch
    indiceMuestra++;

    // Reiniciar el índice si llega al final del buffer
    if (indiceMuestra >= NUM_MUESTRAS) {
      indiceMuestra = 0;
      bufferLleno = true; // El buffer ya tiene al menos NUM_MUESTRAS
    }

    // Calcular el promedio solo si el buffer está lleno o si ya tenemos suficientes muestras
    float promedioPitch = 0.0;
    int muestrasDisponibles = bufferLleno ? NUM_MUESTRAS : indiceMuestra;

    for (int i = 0; i < muestrasDisponibles; i++) {
      promedioPitch += muestrasPitch[i];
    }
    promedioPitch /= muestrasDisponibles;

    Serial.print("Inclinación (pitch) actual: ");
    Serial.print(pitchActual);
    Serial.print(" grados, Promedio (ultimas ");
    Serial.print(muestrasDisponibles);
    Serial.print(" muestras): ");
    Serial.print(promedioPitch);
    Serial.println(" grados");

    // Verificar si el promedio supera el umbral
    if (promedioPitch > UMBRAL_MINIMO && promedioPitch < UMBRAL_MAXIMO) {
      digitalWrite(RELE_PIN, HIGH); // Activar relevador
      Serial.println(">>> Relevador ACTIVADO por Acelerómetro (promedio) <<<");
    } else {
      digitalWrite(RELE_PIN, LOW); // Apagar relevador
      // Serial.println("Relevador apagado por Acelerómetro (promedio) o sin acción"); // Descomentar para depuración
    }
  }

  // --- Lectura del Botón para Python ---
  int currentButtonState = digitalRead(BOTON_PIN);

  if (currentButtonState == LOW && lastButtonState == HIGH) {
    // delay(50); // Debounce
    currentButtonState = digitalRead(BOTON_PIN);
    if (currentButtonState == LOW) {
      Serial.println("BUTTON_PRESSED");
    }
  }
  lastButtonState = currentButtonState;

  delay(10); // Pequeño retardo para estabilidad
}