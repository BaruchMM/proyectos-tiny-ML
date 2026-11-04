#%% [markdown]
# # TinyML 2026: sen(x) en Arduino Nano 33 BLE Sense
#
# Los bloques 0.x generan lo compartido: datos, modelos y sketches. Cada bloque numerado
# corresponde a un apartado del reporte y solo analiza y guarda sus resultados.
# Los apartados 9 (cuestionario) y 10 (conclusiones) no llevan codigo.
#
# Salidas: resultados/<apartado>_<tema>/ y arduino/.
# 3_tab_arduino.csv y 4_tab_tiempos.csv se llenan a mano: el script no las sobrescribe.

#%% 0.1 Configuracion
import contextlib, csv, io, os, sys, zipfile
from collections import Counter
from pathlib import Path

os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')  # desactivado en la corrida que reproduce a Colab
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

SEED = 2026
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# Rutas
try:
    ROOT = Path(__file__).resolve().parent
except NameError:  # celdas interactivas
    ROOT = Path.cwd()
OUTPUT = ROOT / 'resultados'
ARDUINO = ROOT / 'arduino'
ZIP_FILE = ROOT / 'TinyML_Proyecto_1.zip'
ZIP_EXTRAS = ['Proyecto_1.py', 'Proyecto_1_final.ipynb', 'compilar_sketches.py', 'requirements.txt']
ARMAR_ZIP = False   # activar solo en la entrega, cuando existan todas las evidencias

APARTADOS = {
    1: '1_modelo',
    2: '2_precision',
    3: '3_memoria',
    4: '4_rendimiento',
    5: '5_operadores',
    6: '6_generalizacion',
    7: '7_cuantizacion',
    8: '8_calibracion',
}

# Datos
N_TRAIN = 1600
N_VAL = 320
N_TEST = 256
TRAIN_NOISE = 0.015
INTERVALO_A = (0.0, 2 * np.pi)
INTERVALO_B = (-np.pi, np.pi)
INTERVALO_ANCHO = (-2 * np.pi, 2 * np.pi)
N_ANCHO = 512

# Entrenamiento
UNITS = 16
UNITS_ANCHA = 64
LEARNING_RATE = 0.005
BATCH_SIZE = 64
EPOCHS = 200
EPOCHS_B = 400   # con 200 el modelo B llega al tope sin que EarlyStopping lo detenga
PATIENCE = 20

# Cuantizacion: nombre de archivo y descripcion por estrategia
N_REP = 256      # mismo numero de muestras en todos los modelos INT8
# El kernel FULLY_CONNECTED int8 de Chirale 2.0.0 usa una sola escala de pesos por capa
# (filter->params.scale). Con una escala por neurona la placa predice mal sin dar error.
DENSE_POR_CANAL = False
MODELOS = {
    'M1': ('fp32', 'FP32 sin cuantizar'),
    'M2': ('rango_dinamico', 'rango dinamico'),
    'M3': ('float16', 'Float16'),
    'M4': ('int8_es_float32', 'entero con E/S float32'),
    'M5': ('int8', 'INT8 completo'),
}
CALIB = {
    'C1': ('[0, 2pi]', '0_2pi', (0.0, 2 * np.pi)),
    'C2': ('[0, pi]', '0_pi', (0.0, np.pi)),
    'C3': ('[pi, 2pi]', 'pi_2pi', (np.pi, 2 * np.pi)),
    'C4': ('[0, pi/2]', '0_pi2', (0.0, np.pi / 2)),
}
REGIONES = {
    '[0, pi/2]': (0.0, np.pi / 2),
    '[pi/2, pi]': (np.pi / 2, np.pi),
    '[pi, 3pi/2]': (np.pi, 3 * np.pi / 2),
    '[3pi/2, 2pi]': (3 * np.pi / 2, 2 * np.pi),
}

# Arduino
TENSOR_ARENA_BYTES = 24 * 1024
N_TIMING_RUNS = 100
MEDIR_TIEMPO_SKETCH = 1   # 1 = cronometrar y evaluar en la placa, 0 = Serial Plotter
RESOLVERS = ('allops', 'minops')
PENDIENTE = 'pendiente'

# Figuras
MOSTRAR_FIGURAS = 'ipykernel' in sys.modules   # en terminal solo se guardan
DPI_FIG = 200
SUPERFICIE = '#ffffff'
TINTA = '#0b0b0b'
TINTA_2 = '#52514e'
GRIS = '#898781'
REJILLA = '#e1e0d9'
EJE = '#c3c2b7'
BANDA = '#f0efec'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
COLOR_MODELO = dict(zip(['M5', 'M4', 'M3', 'M2', 'M1'], SERIES))
COLOR_CALIB = dict(zip(CALIB, SERIES))   # C1 calibra igual que M5 y comparte su color
COLOR_FP32 = GRIS
COLOR_INT8 = COLOR_MODELO['M5']
COLOR_DTYPE = {'float32': COLOR_FP32, 'float16': COLOR_MODELO['M3'], 'int32': SERIES[1], 'int8': COLOR_INT8}

plt.rcParams.update({
    'figure.facecolor': SUPERFICIE, 'axes.facecolor': SUPERFICIE, 'savefig.facecolor': SUPERFICIE,
    'axes.edgecolor': EJE, 'axes.labelcolor': TINTA_2, 'axes.titlecolor': TINTA,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'axes.grid.axis': 'y', 'axes.axisbelow': True,
    'grid.color': REJILLA, 'grid.linewidth': 0.6,
    'xtick.color': TINTA_2, 'ytick.color': TINTA_2,
    'legend.frameon': False, 'lines.linewidth': 1.6,
})

print('Python:', sys.version.split()[0])
print('TensorFlow:', tf.__version__)
print('NumPy:', np.__version__)
print('Archivos en:', OUTPUT)

#%% 0.2 Funciones
def carpeta(apartado, *sub):
    ruta = OUTPUT.joinpath(APARTADOS[apartado], *sub)
    ruta.mkdir(parents=True, exist_ok=True)
    return ruta


def ruta_tab(apartado, nombre):
    return carpeta(apartado) / f'{apartado}_tab_{nombre}.csv'


def fmt_cell(value):
    return '%.6g' % value if isinstance(value, (float, np.floating)) else str(value)


def fmt_table(headers, rows):
    cols = list(zip(*([headers] + [[str(c) for c in r] for r in rows])))
    w = [max(len(c) for c in col) for col in cols]
    print('  '.join(h.ljust(k) for h, k in zip(headers, w)))
    print('-' * (sum(w) + 2 * (len(w) - 1)))
    for r in rows:
        print('  '.join(str(c).ljust(k) for c, k in zip(r, w)))


def escribir_csv(ruta, headers, rows):
    with open(ruta, 'w', newline='', encoding='ascii') as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)


def guardar_tab(apartado, nombre, headers, rows, mostrar=True):
    ruta = ruta_tab(apartado, nombre)
    escribir_csv(ruta, headers, rows)
    print('\nTabla:', ruta.relative_to(ROOT))
    if mostrar:
        fmt_table(headers, [[fmt_cell(c) for c in r] for r in rows])


def crear_plantilla(apartado, nombre, headers, rows):
    # Tablas que se llenan a mano: solo se crean si no existen
    ruta = ruta_tab(apartado, nombre)
    if ruta.exists():
        print('\nTabla de mediciones conservada:', ruta.relative_to(ROOT))
    else:
        escribir_csv(ruta, headers, rows)
        print('\nPlantilla creada:', ruta.relative_to(ROOT))


def leer_tab(apartado, nombre):
    with open(ruta_tab(apartado, nombre), newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def valor_medido(filas, modelo, resolver, columna):
    for fila in filas:
        if fila['modelo'] == modelo and fila['resolver'] == resolver:
            return fila[columna]
    return PENDIENTE


def diferencia(a, b):
    try:
        return float(a) - float(b)
    except ValueError:
        return PENDIENTE


def guardar_fig(apartado, nombre, fig=None):
    fig = plt.gcf() if fig is None else fig
    ruta = carpeta(apartado) / f'{apartado}_fig_{nombre}.png'
    fig.savefig(ruta, dpi=DPI_FIG, bbox_inches='tight')
    print('Figura:', ruta.relative_to(ROOT))
    if MOSTRAR_FIGURAS:
        plt.show()
    else:
        plt.close(fig)


def guardar_modelo(apartado, nombre, blob):
    ruta = carpeta(apartado, 'modelos') / f'{nombre}.tflite'
    ruta.write_bytes(blob)
    print(f'{nombre}: {len(blob)} bytes -> {ruta.relative_to(ROOT)}')


def make_data(interval, seed):
    rng = np.random.default_rng(seed)
    X_tr = rng.uniform(*interval, size=(N_TRAIN, 1)).astype(np.float32)
    y_tr = (np.sin(X_tr) + rng.normal(0, TRAIN_NOISE, size=(N_TRAIN, 1))).astype(np.float32)
    X_v = rng.uniform(*interval, size=(N_VAL, 1)).astype(np.float32)
    y_v = np.sin(X_v).astype(np.float32)
    return X_tr, y_tr, X_v, y_v


def build_model(name, units=UNITS):
    keras_model = tf.keras.Sequential([
        tf.keras.Input(shape=(1,), dtype=tf.float32),
        tf.keras.layers.Dense(units, activation='tanh'),
        tf.keras.layers.Dense(units, activation='tanh'),
        tf.keras.layers.Dense(1),
    ], name=name)
    keras_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mae'],
    )
    return keras_model


def train_model(keras_model, X_tr, y_tr, X_v, y_v, epochs=EPOCHS):
    return keras_model.fit(
        X_tr, y_tr,
        validation_data=(X_v, y_v),
        epochs=epochs, batch_size=BATCH_SIZE, verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=PATIENCE, restore_best_weights=True
        )],
    )


def error_metrics(reference, prediction):
    return {
        'RMSE': float(np.sqrt(np.mean((reference - prediction)**2))),
        'MAE': float(np.mean(np.abs(reference - prediction))),
        'Error_max': float(np.max(np.abs(reference - prediction))),
    }


def make_rep_fn(lo, hi, n=N_REP):
    xs = np.linspace(lo, hi, n, dtype=np.float32)
    def gen():
        for v in xs:
            yield [np.array([[v]], dtype=np.float32)]
    return gen


def make_converter(keras_model, strategy, rep_fn=None, por_canal=DENSE_POR_CANAL):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    if strategy in ('M4', 'M5'):  # atributo privado del conversor (TF 2.21)
        converter._experimental_disable_per_channel_quantization_for_dense_layers = not por_canal
    if strategy == 'M1':      # sin cuantizar, FP32
        pass
    elif strategy == 'M2':    # rango dinamico
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif strategy == 'M3':    # float16
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif strategy == 'M4':    # entera, entrada y salida float32
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep_fn
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    elif strategy == 'M5':    # entera completa, entrada y salida int8
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep_fn
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    return converter


def convert_model(keras_model, strategy, rep_fn=None, por_canal=DENSE_POR_CANAL):
    with contextlib.redirect_stdout(io.StringIO()):   # silencia el log de conversion
        return make_converter(keras_model, strategy, rep_fn, por_canal).convert()


def analyzer_text(blob):
    salida = io.StringIO()
    with contextlib.redirect_stdout(salida):
        tf.lite.experimental.Analyzer.analyze(model_content=blob)
    return salida.getvalue()


def collect_tensors(blob):
    it = tf.lite.Interpreter(model_content=blob)
    it.allocate_tensors()
    ops = it._get_ops_details()  # API privada
    produced = {int(o) for op in ops for o in op['outputs']}
    used = {int(i) for op in ops for i in op['inputs'] if int(i) >= 0}
    in_idx = {d['index'] for d in it.get_input_details()}
    out_idx = {d['index'] for d in it.get_output_details()}

    rows = []
    for t in it.get_tensor_details():
        i = t['index']
        if i in in_idx:
            role = 'entrada'
        elif i in out_idx:
            role = 'salida'
        elif i in produced:
            role = 'activacion'
        elif i in used:
            role = 'constante'
        else:
            role = '?'
        dt = np.dtype(t['dtype'])
        qp = t['quantization_parameters']
        scales = np.asarray(qp['scales'], dtype=np.float64)
        zps = np.asarray(qp['zero_points'], dtype=np.int64)
        n_el = int(np.prod(t['shape']))
        s_max = float(scales.max()) if scales.size else 0.0
        zp0 = int(zps[0]) if zps.size else 0
        if dt == np.int8 and scales.size:
            lo, hi = (-128 - zp0) * s_max, (127 - zp0) * s_max
        else:
            lo = hi = float('nan')
        rows.append({
            'idx': i, 'rol': role, 'dtype': dt.name,
            'shape': [int(v) for v in t['shape']], 'n_el': n_el,
            'bytes': n_el * dt.itemsize,
            'n_scales': int(scales.size), 'scale_max': s_max,
            'scale_min': float(scales.min()) if scales.size else 0.0,
            'zp': zp0, 'lo': lo, 'hi': hi,
            'es_bias': role == 'constante' and len(t['shape']) < 2,
        })
    return rows, [op['op_name'] for op in ops if op['op_name'] != 'DELEGATE']


def predict_tflite(blob, x_values, verbose=True, name=''):
    lite = tf.lite.Interpreter(model_content=blob)
    lite.allocate_tensors()
    inp = lite.get_input_details()[0]
    out = lite.get_output_details()[0]
    in_scale, in_zero = inp['quantization']
    out_scale, out_zero = out['quantization']
    in_int = inp['dtype'] == np.int8
    out_int = out['dtype'] == np.int8
    if verbose:
        print('%s | entrada: %s %s scale=%s zero=%s | salida: %s %s scale=%s zero=%s' % (
            name, np.dtype(inp['dtype']).name, inp['shape'].tolist(), in_scale, in_zero,
            np.dtype(out['dtype']).name, out['shape'].tolist(), out_scale, out_zero))

    predictions = []
    for value in np.asarray(x_values).reshape(-1):
        if in_int:
            q = int(np.clip(np.rint(float(value) / in_scale + in_zero), -128, 127))
            lite.set_tensor(inp['index'], np.array([[q]], dtype=np.int8))
        else:
            lite.set_tensor(inp['index'], np.array([[value]], dtype=np.float32))
        lite.invoke()
        result = lite.get_tensor(out['index']).reshape(-1)[0]
        if out_int:
            predictions.append((int(result) - out_zero) * out_scale)
        else:
            predictions.append(float(result))
    return np.asarray(predictions, dtype=np.float32)


def expected_in_params(lo, hi):
    # Teoria: rango asimetrico int8; se extiende el rango para incluir el 0.0
    lo2, hi2 = min(lo, 0.0), max(hi, 0.0)
    scale = (hi2 - lo2) / 255.0
    zp = int(np.round(-128 - lo2 / scale))
    return scale, zp


def n_saturated(x, scale, zp):
    q = np.rint(np.asarray(x).reshape(-1) / scale) + zp
    return int(np.sum((q < -128) | (q > 127)))


def dtypes_of(rows, cond):
    return ','.join(sorted({r['dtype'] for r in rows if cond(r)})) or '-'


def forma(shape):
    return 'x'.join(str(v) for v in shape)


#%% 0.3 Datos y entrenamiento
X_train, y_train, X_val, y_val = make_data(INTERVALO_A, SEED)
X_test = np.linspace(*INTERVALO_A, N_TEST, dtype=np.float32).reshape(-1, 1)
y_test = np.sin(X_test).astype(np.float32)
print('Formas:', X_train.shape, y_train.shape, X_val.shape, X_test.shape)

model = build_model('seno_tinyml')
history = train_model(model, X_train, y_train, X_val, y_val)
print('Epocas ejecutadas:', len(history.history['loss']))

#%% 0.4 Conversion y evaluacion de M1 a M5
rep_fn_A = make_rep_fn(*INTERVALO_A)
tflite_models = {}
for key, (archivo, _) in MODELOS.items():
    tflite_models[key] = convert_model(model, key, rep_fn_A)
    guardar_modelo(7, f'{key}_{archivo}', tflite_models[key])

pred_fp32 = model.predict(X_test, verbose=0).reshape(-1)
y_true = y_test.reshape(-1)
metric_fp32 = error_metrics(y_true, pred_fp32)

preds, metrics, all_rows, all_ops = {}, {}, {}, {}
for key, blob in tflite_models.items():
    preds[key] = predict_tflite(blob, X_test, name=key)
    metrics[key] = error_metrics(y_true, preds[key])
    all_rows[key], all_ops[key] = collect_tensors(blob)

#%% 0.5 Exportar model.h y sketches
def tflite_to_header(blob, symbol='g_model', desc=''):
    body = '\n'.join(
        '  ' + ', '.join(f'0x{b:02x}' for b in blob[i:i+12]) + ','
        for i in range(0, len(blob), 12)
    )
    return (f'#pragma once\n#include <stdint.h>\n\n'
            f'// Modelo {desc} generado por Proyecto_1.py. No editar.\n'
            f'alignas(16) const unsigned char {symbol}[] = {{\n{body}\n}};\n'
            f'const unsigned int {symbol}_len = sizeof({symbol});\n')


SKETCH = r'''// TinyML 2026 | sen(x) con modelo @@NAME@@ | Resolver: @@RESOLVER_KIND@@
// Placas: Arduino Nano 33 BLE Sense / Rev2. Biblioteca: Chirale_TensorFlowLite.
#include <Arduino.h>
#include <math.h>
#include <stdint.h>
#include <Chirale_TensorFlowLite.h>
@@RESOLVER_INCLUDE@@
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model.h"

// 0 = Serial Plotter (real y pred), 1 = cronometrar inferencias.
#define MEDIR_TIEMPO @@MEDIR@@
// Poner en 0 si la biblioteca no tiene arena_used_bytes().
#define USE_ARENA_USED 1

constexpr int kTensorArenaSize = @@ARENA@@;
constexpr int kNumInferencias = @@NRUNS@@;
constexpr float kDosPi = 6.28318530718f;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* net = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

void fatal(const char* message) {
  // Se repite porque el monitor serie puede abrirse despues del error
  while (true) {
    Serial.print("[error] modelo=@@NAME@@ resolver=@@RESOLVER_KIND@@ ");
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

@@RESOLVER_DECL@@
  static tflite::MicroInterpreter static_interpreter(
      net, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    fatal("ERROR: AllocateTensors. Revisar arena, operadores o versiones.");
  }
  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);

  Serial.print("modelo: "); Serial.println("@@NAME@@");
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
  Serial.print("[tiempo] modelo=@@NAME@@ resolver=@@RESOLVER_KIND@@ n="); Serial.print(kNumInferencias);
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
}'''

OP_ADD = {
    'FULLY_CONNECTED': 'AddFullyConnected',
    'TANH': 'AddTanh',
    'QUANTIZE': 'AddQuantize',
    'DEQUANTIZE': 'AddDequantize',
}


def build_sketch(key, blob, ops, min_resolver):
    kind = 'MicroMutableOpResolver' if min_resolver else 'AllOpsResolver'
    folder_name = f'Sketch_{key}_{"minops" if min_resolver else "allops"}'
    folder = ARDUINO / folder_name
    folder.mkdir(parents=True, exist_ok=True)

    if min_resolver:
        unique_ops = sorted(set(ops))
        lines = [f'  static tflite::MicroMutableOpResolver<{len(unique_ops)}> resolver;']
        for op in unique_ops:
            lines.append(f'  resolver.{OP_ADD[op]}();' if op in OP_ADD
                         else f'  // TODO: operador sin mapeo: {op}')
        decl = '\n'.join(lines)
        include = '#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"'
    else:
        decl = '  static tflite::AllOpsResolver resolver;'
        include = '#include "tensorflow/lite/micro/all_ops_resolver.h"'

    code = (SKETCH.replace('@@NAME@@', key)
                  .replace('@@RESOLVER_KIND@@', kind)
                  .replace('@@RESOLVER_INCLUDE@@', include)
                  .replace('@@RESOLVER_DECL@@', decl)
                  .replace('@@ARENA@@', str(TENSOR_ARENA_BYTES))
                  .replace('@@NRUNS@@', str(N_TIMING_RUNS))
                  .replace('@@MEDIR@@', str(MEDIR_TIEMPO_SKETCH)))
    (folder / f'{folder_name}.ino').write_text(code, encoding='ascii')
    (folder / 'model.h').write_text(tflite_to_header(blob, desc=MODELOS[key][1]), encoding='ascii')
    return folder


for key, blob in tflite_models.items():
    for min_res in (False, True):
        folder = build_sketch(key, blob, all_ops[key], min_res)
        print('Sketch:', folder.name, '| modelo:', len(blob), 'bytes')

#%% 1 Modelo
guardar_tab(1, 'entorno', ['Componente', 'Version'], [
    ['Python', sys.version.split()[0]],
    ['TensorFlow', tf.__version__],
    ['Keras', tf.keras.__version__],
    ['NumPy', np.__version__],
])

guardar_tab(1, 'entrenamiento', ['Parametro', 'Valor'], [
    ['Arquitectura', '1-%d-%d-1' % (UNITS, UNITS)],
    ['Activacion en capas ocultas', 'tanh'],
    ['Optimizador', 'Adam'],
    ['Learning rate', LEARNING_RATE],
    ['Tamano de lote', BATCH_SIZE],
    ['Perdida', 'mse'],
    ['Epocas maximas', EPOCHS],
    ['Paciencia de EarlyStopping', PATIENCE],
    ['Epocas ejecutadas', len(history.history['loss'])],
])

guardar_tab(1, 'conjuntos', ['Conjunto', 'Muestras', 'Intervalo', 'Muestreo', 'Etiqueta'], [
    ['Entrenamiento', N_TRAIN, '[0, 2pi]', 'uniforme aleatorio', 'sen(x) + ruido %g' % TRAIN_NOISE],
    ['Validacion', N_VAL, '[0, 2pi]', 'uniforme aleatorio', 'sen(x)'],
    ['Prueba', N_TEST, '[0, 2pi]', 'rejilla regular', 'sen(x)'],
])

rows_a1 = []
total = 0
for layer in model.layers:
    w, b = layer.get_weights()
    n = int(w.size + b.size)
    total += n
    rows_a1.append([layer.name, '%dx%d + %d' % (w.shape[0], w.shape[1], b.size), n, n * 4])
rows_a1.append(['TOTAL', '', total, total * 4])
guardar_tab(1, 'parametros', ['Capa', 'W + b', 'Parametros', 'Bytes FP32'], rows_a1)

plt.figure(figsize=(9, 3.6))
plt.scatter(X_train[::5, 0], y_train[::5, 0], s=9, alpha=0.35, color=SERIES[0], label='Entrenamiento (muestra)')
plt.plot(X_test[:, 0], y_test[:, 0], color=TINTA, linewidth=2, label='sen(x) real')
plt.xlabel('x [rad]'); plt.ylabel('y'); plt.title('Conjunto de entrenamiento')
plt.legend()
guardar_fig(1, 'datos')

plt.figure(figsize=(9, 3.4))
plt.plot(history.history['loss'], color=SERIES[0], label='MSE entrenamiento')
plt.plot(history.history['val_loss'], color=SERIES[1], label='MSE validacion')
plt.yscale('log'); plt.xlabel('Epoca'); plt.ylabel('MSE (log)')
plt.title('Curvas de aprendizaje'); plt.legend()
guardar_fig(1, 'curvas_aprendizaje')

#%% 2 Precision
guardar_tab(2, 'errores', ['Comparacion', 'RMSE', 'MAE', 'Error_max'], [
    [name, m['RMSE'], m['MAE'], m['Error_max']] for name, m in {
        'Keras float32 contra sen(x)': metric_fp32,
        'M5 INT8 contra sen(x)': metrics['M5'],
        'M5 INT8 contra Keras float32': error_metrics(pred_fp32, preds['M5']),
    }.items()
])

plt.figure(figsize=(9, 3.6))
plt.plot(X_test[:, 0], y_true, color=TINTA, linewidth=2.2, label='sen(x) real')
plt.plot(X_test[:, 0], pred_fp32, '--', color=COLOR_FP32, label='Keras float32')
plt.plot(X_test[:, 0], preds['M5'], color=COLOR_INT8, label='M5 INT8')
plt.xlabel('x [rad]'); plt.ylabel('Salida'); plt.title('Predicciones float32 e INT8')
plt.legend()
guardar_fig(2, 'predicciones')

plt.figure(figsize=(9, 3.2))
plt.plot(X_test[:, 0], np.abs(y_true - pred_fp32), '--', color=COLOR_FP32, label='Keras float32')
plt.plot(X_test[:, 0], np.abs(y_true - preds['M5']), color=COLOR_INT8, label='M5 INT8')
plt.xlabel('x [rad]'); plt.ylabel('|error|'); plt.title('Error absoluto contra sen(x)')
plt.legend()
guardar_fig(2, 'error_absoluto')

#%% 3 Memoria
bytes_fp32 = model.count_params() * 4
rows = []
for key, blob in tflite_models.items():
    constantes = sum(r['bytes'] for r in all_rows[key] if r['rol'] == 'constante')
    rows.append([key, len(blob), bytes_fp32, len(blob) - bytes_fp32, constantes, len(blob) - constantes])
guardar_tab(3, 'tflite_vs_fp32',
            ['Modelo', 'Bytes tflite', 'Pesos FP32 teoricos', 'Diferencia', 'Bytes constantes', 'Bytes resto'], rows)

crear_plantilla(3, 'arduino',
                ['modelo', 'resolver', 'bytes_tflite', 'flash_bytes', 'ram_estatica_bytes',
                 'arena_bytes', 'arena_usada_bytes', 'compila', 'funciona'],
                [[key, res, len(blob), PENDIENTE, PENDIENTE, TENSOR_ARENA_BYTES, PENDIENTE, PENDIENTE, PENDIENTE]
                 for key, blob in tflite_models.items() for res in RESOLVERS])

#%% 4 Rendimiento
crear_plantilla(4, 'tiempos',
                ['modelo', 'resolver', 'n', 'min_us', 'max_us', 'prom_us', 'desv_us', 'rmse_placa', 'mae_placa'],
                [[key, res, N_TIMING_RUNS] + [PENDIENTE] * 6
                 for key in tflite_models for res in RESOLVERS])

#%% 5 Operadores
rows = []
for key, ops in all_ops.items():
    conteo = sorted(Counter(ops).items())
    rows.append([key, len(ops), len(conteo), ' '.join('%s x%d' % (op, n) for op, n in conteo)])
guardar_tab(5, 'operadores', ['Modelo', 'Nodos', 'N resolver', 'Operadores'], rows)

med_memoria = leer_tab(3, 'arduino')
med_tiempo = leer_tab(4, 'tiempos')
rows = []
for key in tflite_models:
    for metrica, filas, columna in (('Flash (bytes)', med_memoria, 'flash_bytes'),
                                    ('RAM estatica (bytes)', med_memoria, 'ram_estatica_bytes'),
                                    ('Tiempo promedio (us)', med_tiempo, 'prom_us'),
                                    ('RMSE en placa', med_tiempo, 'rmse_placa')):
        a = valor_medido(filas, key, 'allops', columna)
        b = valor_medido(filas, key, 'minops', columna)
        rows.append([key, metrica, a, b, diferencia(a, b)])
guardar_tab(5, 'resolvers', ['Modelo', 'Metrica', 'AllOpsResolver', 'MicroMutableOpResolver', 'Diferencia'], rows)

#%% 6 Generalizacion
X_train_B, y_train_B, X_val_B, y_val_B = make_data(INTERVALO_B, SEED + 1)

tf.keras.utils.set_random_seed(SEED)
model_B = build_model('seno_B')
history_B = train_model(model_B, X_train_B, y_train_B, X_val_B, y_val_B, EPOCHS_B)
print('Epocas modelo B:', len(history_B.history['loss']), 'de', EPOCHS_B)

blob_B = convert_model(model_B, 'M5', make_rep_fn(*INTERVALO_B))
guardar_modelo(6, 'B_int8', blob_B)

X_wide = np.linspace(*INTERVALO_ANCHO, N_ANCHO, dtype=np.float32).reshape(-1, 1)
y_wide = np.sin(X_wide).reshape(-1)

cases = {
    'A [0, 2pi]': (model, tflite_models['M5'], INTERVALO_A, history),
    'B [-pi, pi]': (model_B, blob_B, INTERVALO_B, history_B),
}
res = {}
for name, (kmodel, blob, (lo, hi), hist) in cases.items():
    p_fp = kmodel.predict(X_wide, verbose=0).reshape(-1)
    p_q = predict_tflite(blob, X_wide, verbose=False)
    rows, _ = collect_tensors(blob)
    r_in = [r for r in rows if r['rol'] == 'entrada'][0]
    inside = (X_wide[:, 0] >= lo) & (X_wide[:, 0] <= hi)
    res[name] = {
        'fp': p_fp, 'q': p_q, 'interval': (lo, hi),
        'epocas': len(hist.history['loss']),
        'sat': n_saturated(X_wide, r_in['scale_max'], r_in['zp']),
        'fp_in': error_metrics(y_wide[inside], p_fp[inside])['RMSE'],
        'fp_out': error_metrics(y_wide[~inside], p_fp[~inside])['RMSE'],
        'q_in': error_metrics(y_wide[inside], p_q[inside])['RMSE'],
        'q_out': error_metrics(y_wide[~inside], p_q[~inside])['RMSE'],
    }

guardar_tab(6, 'dentro_fuera',
            ['Modelo', 'Epocas', 'RMSE float32 dentro', 'RMSE float32 fuera',
             'RMSE INT8 dentro', 'RMSE INT8 fuera', 'Entradas saturadas'],
            [[name, r['epocas'], r['fp_in'], r['fp_out'], r['q_in'], r['q_out'], r['sat']]
             for name, r in res.items()])

fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
for ax, (name, r) in zip(axes, res.items()):
    ax.axvspan(*r['interval'], color=BANDA, label='intervalo de entrenamiento')
    ax.plot(X_wide[:, 0], y_wide, color=TINTA, linewidth=2.2, label='sen(x) real')
    ax.plot(X_wide[:, 0], r['fp'], '--', color=COLOR_FP32, label='float32')
    ax.plot(X_wide[:, 0], r['q'], color=COLOR_INT8, label='INT8')
    ax.set_title('Modelo ' + name); ax.set_ylabel('Salida')
axes[1].set_xlabel('x [rad]')
fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=4)
plt.tight_layout(rect=(0, 0.05, 1, 1))
guardar_fig(6, 'modelos_a_b', fig)

#%% 7 Estrategias de cuantizacion
evidencias = carpeta(7, 'evidencias')
for key, blob in tflite_models.items():
    (evidencias / f'7_analyzer_{key}.txt').write_text(analyzer_text(blob), encoding='ascii')

csv_rows = []
for key, rows in all_rows.items():
    for r in rows:
        csv_rows.append([key, r['idx'], r['rol'], r['dtype'], forma(r['shape']),
                         r['n_el'], r['bytes'], r['n_scales'], r['scale_min'], r['scale_max'],
                         r['zp'], r['lo'], r['hi']])
guardar_tab(7, 'tensores',
            ['modelo', 'idx', 'rol', 'dtype', 'forma', 'elementos', 'bytes',
             'n_scales', 'scale_min', 'scale_max', 'zero_point', 'rango_lo', 'rango_hi'],
            csv_rows, mostrar=False)

summary = []
for key, rows in all_rows.items():
    summary.append([
        key, len(tflite_models[key]), len(rows), len(all_ops[key]),
        dtypes_of(rows, lambda r: r['rol'] == 'constante' and not r['es_bias']),
        dtypes_of(rows, lambda r: r['es_bias']),
        dtypes_of(rows, lambda r: r['rol'] == 'activacion'),
        dtypes_of(rows, lambda r: r['rol'] == 'entrada'),
        dtypes_of(rows, lambda r: r['rol'] == 'salida'),
    ])
guardar_tab(7, 'resumen', ['Modelo', 'Bytes', 'Tensores', 'Ops', 'Pesos', 'Sesgos', 'Activ.', 'Entrada', 'Salida'], summary)

# Tabla: tamano, reduccion, tipo de E/S, RMSE y MAE
size_m1 = len(tflite_models['M1'])
rows = []
for key, blob in tflite_models.items():
    tipos = '%s/%s' % (dtypes_of(all_rows[key], lambda r: r['rol'] == 'entrada'),
                       dtypes_of(all_rows[key], lambda r: r['rol'] == 'salida'))
    rows.append([key, len(blob), round((size_m1 - len(blob)) / size_m1 * 100, 1), tipos,
                 metrics[key]['RMSE'], metrics[key]['MAE']])
guardar_tab(7, 'comparacion', ['Modelo', 'Bytes', 'Reduccion %', 'Entrada/Salida', 'RMSE', 'MAE'], rows)

# Escala de pesos por neurona contra una por capa: el FlatBuffer cambia y la placa solo soporta la segunda
rows = []
for por_canal in (True, False):
    blob = tflite_models['M5'] if por_canal == DENSE_POR_CANAL else convert_model(model, 'M5', rep_fn_A, por_canal)
    if por_canal != DENSE_POR_CANAL:
        guardar_modelo(7, 'M5_int8_' + ('por_canal' if por_canal else 'por_capa'), blob)
    filas = collect_tensors(blob)[0]
    pesos = [r for r in filas if r['rol'] == 'constante' and not r['es_bias']]
    m = error_metrics(y_true, predict_tflite(blob, X_test, verbose=False))
    rows.append(['por neurona' if por_canal else 'por capa', len(blob),
                 ' '.join(str(r['n_scales']) for r in sorted(pesos, key=lambda r: r['idx'])),
                 'si' if por_canal == DENSE_POR_CANAL else 'no', m['RMSE'], m['MAE']])
guardar_tab(7, 'escala_pesos', ['Escala de pesos (M5)', 'Bytes', 'Escalas por tensor de pesos',
                                'Usado en arduino/', 'RMSE', 'MAE'], rows)

# Compromiso entre tamano, precision y costo (se completa con las mediciones de Arduino)
# La placa evalua N_TIMING_RUNS puntos: se repite esa malla en Python para comparar sin sesgo
X_placa = np.linspace(*INTERVALO_A, N_TIMING_RUNS, dtype=np.float32)
y_placa = np.sin(X_placa)
rows = []
for key, blob in tflite_models.items():
    rows.append([key, len(blob), metrics[key]['RMSE'],
                 error_metrics(y_placa, predict_tflite(blob, X_placa, verbose=False))['RMSE'],
                 valor_medido(med_tiempo, key, 'minops', 'rmse_placa'),
                 valor_medido(med_memoria, key, 'minops', 'compila'),
                 valor_medido(med_memoria, key, 'minops', 'funciona'),
                 valor_medido(med_memoria, key, 'minops', 'flash_bytes'),
                 valor_medido(med_memoria, key, 'minops', 'ram_estatica_bytes'),
                 valor_medido(med_tiempo, key, 'minops', 'prom_us')])
guardar_tab(7, 'despliegue',
            ['Modelo', 'Bytes', 'RMSE', 'RMSE malla placa (Python)', 'RMSE en placa', 'Compila', 'Funciona',
             'Flash (bytes)', 'RAM estatica (bytes)', 'Tiempo promedio (us)'],
            rows)

# Predicciones por estrategia
plt.figure(figsize=(9.5, 4.2))
plt.plot(X_test[:, 0], y_true, color=TINTA, label='sen(x) real', linewidth=2.2)
plt.plot(X_test[:, 0], pred_fp32, '--', color=COLOR_FP32, label='Keras float32')
# M1 y M2 coinciden, igual que M4 y M5: la linea ancha deja ver a la punteada encima
styles = {'M1': ('-', 2.8), 'M2': ('--', 1.2), 'M3': ('-.', 1.2), 'M4': ('-', 2.8), 'M5': ('--', 1.2)}
for key in tflite_models:
    plt.plot(X_test[:, 0], preds[key], styles[key][0], color=COLOR_MODELO[key], linewidth=styles[key][1], label=key)
plt.xlabel('x [rad]'); plt.ylabel('Salida')
plt.title('Validacion de inferencia por estrategia')
plt.legend(ncol=3)
guardar_fig(7, 'predicciones')

# Barras de tamano y de error
keys = list(tflite_models)
x_pos = np.arange(len(keys))

plt.figure(figsize=(8, 3.6))
barras = plt.bar(keys, [len(tflite_models[k]) for k in keys], width=0.6, color=SERIES[0])
plt.bar_label(barras, padding=2, color=TINTA_2)
plt.ylabel('Bytes'); plt.title('Tamano del archivo .tflite')
guardar_fig(7, 'tamanos')

plt.figure(figsize=(8, 3.6))
plt.bar(x_pos - 0.2, [metrics[k]['RMSE'] for k in keys], width=0.38, color=SERIES[0], label='RMSE')
plt.bar(x_pos + 0.2, [metrics[k]['MAE'] for k in keys], width=0.38, color=SERIES[1], label='MAE')
plt.xticks(x_pos, keys); plt.ylabel('Error')
plt.title('Error por estrategia de cuantizacion')
plt.legend()
guardar_fig(7, 'errores')

# Bytes de constantes por tipo de dato
dtypes = sorted({r['dtype'] for rows in all_rows.values() for r in rows if r['rol'] == 'constante'})
bottom = np.zeros(len(keys))
plt.figure(figsize=(8, 4))
for dt in dtypes:
    vals = np.array([sum(r['bytes'] for r in all_rows[k] if r['rol'] == 'constante' and r['dtype'] == dt)
                     for k in keys], dtype=float)
    plt.bar(keys, vals, bottom=bottom, width=0.6, color=COLOR_DTYPE[dt], edgecolor=SUPERFICIE, linewidth=1, label=dt)
    bottom += vals
plt.ylabel('Bytes de pesos y sesgos')
plt.title('Bytes de constantes por tipo de dato')
plt.legend()
guardar_fig(7, 'bytes_por_tipo')

# Rango representable de los tensores int8
int_models = [k for k in keys if any(r['dtype'] == 'int8' and r['n_scales'] for r in all_rows[k])]
fig, axes = plt.subplots(len(int_models), 1, figsize=(9, 3.2 * len(int_models)), squeeze=False)
for ax, key in zip(axes[:, 0], int_models):
    rows = [r for r in all_rows[key] if r['dtype'] == 'int8' and r['n_scales']]
    labels = ['%d %s %s' % (r['idx'], r['rol'], r['shape']) for r in rows]
    ypos = np.arange(len(rows))
    ax.barh(ypos, [r['hi'] - r['lo'] for r in rows], left=[r['lo'] for r in rows], height=0.6, color=SERIES[0])
    for y, r in zip(ypos, rows):
        ax.text(r['hi'], y, ' scale=%.3g zp=%d' % (r['scale_max'], r['zp']), va='center', fontsize=7, color=TINTA_2)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color=EJE, linewidth=0.6)
    ax.set_title('%s: rango representable por tensor int8' % key)
    ax.set_xlabel('Valor real')
    ax.grid(axis='x'); ax.grid(False, axis='y')
plt.tight_layout()
guardar_fig(7, 'rangos_int8', fig)

# Rango dinamico en una red mas grande (1-64-64-1)
tf.keras.utils.set_random_seed(SEED)
model_ancha = build_model('seno_ancha', UNITS_ANCHA)
history_ancha = train_model(model_ancha, X_train, y_train, X_val, y_val)
print('Epocas red ancha:', len(history_ancha.history['loss']))

filas_tam, filas_const = [], []
for key in ('M1', 'M2'):
    blob = convert_model(model_ancha, key)
    guardar_modelo(7, f'Ancha_{key}_{MODELOS[key][0]}', blob)
    m = error_metrics(y_true, predict_tflite(blob, X_test, verbose=False))
    rows, _ = collect_tensors(blob)
    filas_tam.append([key, len(blob), sum(r['bytes'] for r in rows if r['rol'] == 'constante'), m['RMSE'], m['MAE']])
    filas_const += [[key, forma(r['shape']), r['n_el'], r['dtype'], r['bytes']]
                    for r in sorted(rows, key=lambda r: r['n_el']) if r['rol'] == 'constante']
guardar_tab(7, 'red_ancha_tamanos', ['Modelo', 'Bytes', 'Bytes constantes', 'RMSE', 'MAE'], filas_tam)
guardar_tab(7, 'red_ancha_constantes', ['Modelo', 'Forma', 'Elementos', 'Tipo', 'Bytes'], filas_const)

#%% 8 Calibracion
c_blobs, c_rows, c_preds, c_metrics, c_sat = {}, {}, {}, {}, {}
for key, (_, archivo, (lo, hi)) in CALIB.items():
    blob = convert_model(model, 'M5', make_rep_fn(lo, hi))
    c_blobs[key] = blob
    guardar_modelo(8, f'{key}_{archivo}', blob)
    rows, _ = collect_tensors(blob)
    c_rows[key] = rows
    c_preds[key] = predict_tflite(blob, X_test, verbose=False)
    c_metrics[key] = error_metrics(y_true, c_preds[key])
    r_in = [r for r in rows if r['rol'] == 'entrada'][0]
    c_sat[key] = n_saturated(X_test, r_in['scale_max'], r_in['zp'])

# Parametros de entrada, salida y tamano
rows = []
for key, (texto, _, (lo, hi)) in CALIB.items():
    r_in = [r for r in c_rows[key] if r['rol'] == 'entrada'][0]
    r_out = [r for r in c_rows[key] if r['rol'] == 'salida'][0]
    th_s, th_z = expected_in_params(lo, hi)
    rows.append([key, texto, r_in['scale_max'], r_in['zp'], th_s, th_z, r_in['lo'], r_in['hi'],
                 r_out['scale_max'], r_out['zp'], r_out['lo'], r_out['hi'], len(c_blobs[key])])
guardar_tab(8, 'parametros',
            ['Modelo', 'Calibracion', 'scale_in', 'zp_in', 'scale_in_teo', 'zp_teo', 'in_lo', 'in_hi',
             'scale_out', 'zp_out', 'out_lo', 'out_hi', 'Bytes'], rows)

# Activaciones intermedias, en orden
rows = []
for key in CALIB:
    acts = [r for r in c_rows[key] if r['rol'] == 'activacion']
    rows += [[key, n, r['scale_max'], r['zp'], r['lo'], r['hi']] for n, r in enumerate(acts, 1)]
guardar_tab(8, 'activaciones', ['Modelo', 'Activacion', 'scale', 'zero_point', 'rango_lo', 'rango_hi'], rows)

# Errores sobre [0, 2pi]. RMSE = contra sen(x) real. RMSE cuant. = INT8 contra float32 (error solo de cuantizar).
rows = [['Keras float32', metric_fp32['RMSE'], metric_fp32['MAE'], metric_fp32['Error_max'], '-', '-']]
for key in CALIB:
    rmse_q = float(np.sqrt(np.mean((pred_fp32 - c_preds[key]) ** 2)))
    rows.append([key, c_metrics[key]['RMSE'], c_metrics[key]['MAE'], c_metrics[key]['Error_max'], c_sat[key], rmse_q])
guardar_tab(8, 'errores', ['Modelo', 'RMSE', 'MAE', 'Error_max', 'Entradas saturadas', 'RMSE cuant.'], rows)

# RMSE por region y ubicacion del error maximo
x_test = X_test[:, 0]
rows = []
for name, pred in {'Keras float32': pred_fp32, **c_preds}.items():
    err = np.abs(y_true - pred)
    fila = [name]
    for lo, hi in REGIONES.values():
        region = (x_test >= lo) & (x_test <= hi)
        fila.append(float(np.sqrt(np.mean(err[region] ** 2))))
    rows.append(fila + [float(err.max()), float(x_test[err.argmax()])])
guardar_tab(8, 'regiones', ['Modelo'] + ['RMSE ' + r for r in REGIONES] + ['Error_max', 'x del error max'], rows)

plt.figure(figsize=(9.5, 4.2))
plt.plot(x_test, y_true, color=TINTA, linewidth=2.2, label='sen(x) real')
plt.plot(x_test, pred_fp32, '--', color=COLOR_FP32, label='Keras float32')
for key in CALIB:
    plt.plot(x_test, c_preds[key], color=COLOR_CALIB[key], label='%s %s' % (key, CALIB[key][0]))
plt.xlabel('x [rad]'); plt.ylabel('Salida'); plt.title('Predicciones segun el intervalo de calibracion')
plt.legend(ncol=3)
guardar_fig(8, 'predicciones')

plt.figure(figsize=(9.5, 3.8))
for key in CALIB:
    plt.plot(x_test, np.abs(y_true - c_preds[key]), color=COLOR_CALIB[key], label='%s %s' % (key, CALIB[key][0]))
plt.plot(x_test, np.abs(y_true - pred_fp32), '--', color=COLOR_FP32, label='Keras float32')
plt.xlabel('x [rad]'); plt.ylabel('|error|'); plt.title('Error absoluto contra sen(x)')
plt.legend(ncol=3)
guardar_fig(8, 'error_absoluto')

#%% 11 Archivos
if ARMAR_ZIP:
    with zipfile.ZipFile(ZIP_FILE, 'w', zipfile.ZIP_DEFLATED) as z:
        for base in (OUTPUT, ARDUINO):
            for ruta in sorted(base.rglob('*')):
                if ruta.is_file():
                    z.write(ruta, ruta.relative_to(ROOT))
        for nombre in ZIP_EXTRAS:
            if (ROOT / nombre).exists():
                z.write(ROOT / nombre, nombre)
            else:
                print('Falta para el zip:', nombre)
    print('Zip:', ZIP_FILE.name, '|', ZIP_FILE.stat().st_size, 'bytes')
