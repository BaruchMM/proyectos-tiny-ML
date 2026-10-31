// TinyML 2026 | sen(x) con modelo M5 | Resolver: MicroMutableOpResolver
// Placas: Arduino Nano 33 BLE Sense / Rev2. Biblioteca: Chirale_TensorFlowLite.
#include <Arduino.h>
#include <math.h>
#include <stdint.h>
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model.h"

// 0 = Serial Plotter (real y pred), 1 = cronometrar inferencias.
#define MEDIR_TIEMPO 1
// Poner en 0 si la biblioteca no tiene arena_used_bytes().
#define USE_ARENA_USED 1

constexpr int kTensorArenaSize = 24576;
constexpr int kNumInferencias = 100;
constexpr float kDosPi = 6.28318530718f;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* net = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

void fatal(const char* message) {
  // Se repite porque el monitor serie puede abrirse despues del error
  while (true) {
    Serial.print("[error] modelo=M5 resolver=MicroMutableOpResolver ");
    Serial.println(message);
    delay(2000);
  }
}

void set_input(float x) {
  if (input_tensor->type == kTfLiteInt8) {
    // q = round(x / scale) + zero_point, saturada a [-128, 127]
    int q = (int)lroundf(x / input_tensor->params.scale) + input_tensor->params.zero_point;
    if (q < -128) q = -128;
    if (q >  127) q =  127;
    input_tensor->data.int8[0] = (int8_t)q;
  } else {
    input_tensor->data.f[0] = x;
  }
}

float get_output() {
  if (output_tensor->type == kTfLiteInt8) {
    return (float)(output_tensor->data.int8[0] - output_tensor->params.zero_point)
           * output_tensor->params.scale;
  }
  return output_tensor->data.f[0];
}

void setup() {
  Serial.begin(115200);
  unsigned long inicio = millis();
  while (!Serial && millis() - inicio < 3000) { delay(10); }

  net = tflite::GetModel(g_model);
  if (net->version() != TFLITE_SCHEMA_VERSION) {
    fatal("ERROR: version de schema incompatible.");
  }

  static tflite::MicroMutableOpResolver<2> resolver;
  resolver.AddFullyConnected();
  resolver.AddTanh();
  static tflite::MicroInterpreter static_interpreter(
      net, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    fatal("ERROR: AllocateTensors. Revisar arena, operadores o versiones.");
  }
  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);

  Serial.print("modelo: "); Serial.println("M5");
  Serial.print("tipo entrada (1=float32, 9=int8): "); Serial.println((int)input_tensor->type);
  Serial.print("tipo salida: "); Serial.println((int)output_tensor->type);
  Serial.print("scale/zero entrada: "); Serial.print(input_tensor->params.scale, 8);
  Serial.print(" / "); Serial.println(input_tensor->params.zero_point);
  Serial.print("scale/zero salida: "); Serial.print(output_tensor->params.scale, 8);
  Serial.print(" / "); Serial.println(output_tensor->params.zero_point);
  Serial.print("kTensorArenaSize: "); Serial.println(kTensorArenaSize);
#if USE_ARENA_USED
  Serial.print("arena_used_bytes: "); Serial.println((unsigned long)interpreter->arena_used_bytes());
#endif

  // Inferencia de calentamiento
  set_input(0.0f);
  if (interpreter->Invoke() != kTfLiteOk) { fatal("ERROR: fallo de inferencia."); }
}

void loop() {
#if MEDIR_TIEMPO
  static uint32_t tiempos[kNumInferencias];
  float suma_cuad = 0.0f, suma_abs = 0.0f;
  for (int i = 0; i < kNumInferencias; i++) {
    float x = kDosPi * (float)i / (float)(kNumInferencias - 1);
    set_input(x);
    uint32_t t_ini = micros();
    TfLiteStatus st = interpreter->Invoke();
    uint32_t t_fin = micros();
    if (st != kTfLiteOk) { fatal("ERROR: fallo de inferencia."); }
    tiempos[i] = t_fin - t_ini;
    float e = sinf(x) - get_output();
    suma_cuad += e * e;
    suma_abs += fabsf(e);
  }
  uint32_t minimo = tiempos[0], maximo = tiempos[0];
  uint64_t suma = 0;
  for (int i = 0; i < kNumInferencias; i++) {
    if (tiempos[i] < minimo) minimo = tiempos[i];
    if (tiempos[i] > maximo) maximo = tiempos[i];
    suma += tiempos[i];
  }
  float media = (float)suma / (float)kNumInferencias;
  float var = 0.0f;
  for (int i = 0; i < kNumInferencias; i++) {
    float d = (float)tiempos[i] - media;
    var += d * d;
  }
  float desv = sqrtf(var / (float)kNumInferencias);
  Serial.print("[tiempo] modelo=M5 resolver=MicroMutableOpResolver n="); Serial.print(kNumInferencias);
  Serial.print(" min_us="); Serial.print(minimo);
  Serial.print(" max_us="); Serial.print(maximo);
  Serial.print(" prom_us="); Serial.print(media, 3);
  Serial.print(" desv_us="); Serial.print(desv, 3);
  Serial.print(" rmse="); Serial.print(sqrtf(suma_cuad / (float)kNumInferencias), 5);
  Serial.print(" mae="); Serial.print(suma_abs / (float)kNumInferencias, 5);
#if USE_ARENA_USED
  Serial.print(" arena_usada="); Serial.print((unsigned long)interpreter->arena_used_bytes());
#endif
  Serial.println();
  delay(3000);
#else
  static uint16_t paso = 0;
  constexpr uint16_t kPasos = 128;
  float x = kDosPi * (float)(paso % kPasos) / (float)(kPasos - 1);
  paso++;
  set_input(x);
  if (interpreter->Invoke() != kTfLiteOk) { fatal("ERROR: fallo de inferencia."); }
  Serial.print("real:"); Serial.print(sinf(x), 4);
  Serial.print(",pred:"); Serial.println(get_output(), 4);
  delay(80);
#endif
}