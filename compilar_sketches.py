"""Compila los sketches de arduino/ para la Nano 33 BLE. No necesita la tarjeta.

Guarda la salida del compilador en un txt y anota Flash y RAM en 3_tab_arduino.csv.
Orden cuando cambien los modelos o los sketches: Proyecto_1.py, este script y otra vez
Proyecto_1.py, que actualiza 5_tab_resolvers.csv y 7_tab_despliegue.csv con estos datos.
"""
import csv, re, subprocess, unicodedata
from pathlib import Path

ARDUINO_CLI = r'C:\Program Files\Arduino IDE\resources\app\lib\backend\resources\arduino-cli.exe'
FQBN = 'arduino:mbed_nano:nano33ble'

ROOT = Path(__file__).resolve().parent
SKETCHES = ROOT / 'arduino'
TABLA = ROOT / 'resultados' / '3_memoria' / '3_tab_arduino.csv'
LOG = ROOT / 'resultados' / '3_memoria' / 'evidencias' / '3_compilacion.txt'


def to_ascii(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')


with open(TABLA, newline='', encoding='utf-8-sig') as f:
    filas = list(csv.DictReader(f))

LOG.parent.mkdir(parents=True, exist_ok=True)
with open(LOG, 'w', newline='\n', encoding='ascii') as log:
    log.write(f'Compilacion para {FQBN} con arduino-cli\n')
    for fila in filas:
        nombre = f"Sketch_{fila['modelo']}_{fila['resolver']}"
        header = (SKETCHES / nombre / 'model.h').read_text(encoding='ascii')
        fila['bytes_tflite'] = len(re.findall(r'0x[0-9a-f]{2}', header))
        p = subprocess.run([ARDUINO_CLI, 'compile', '--fqbn', FQBN, str(SKETCHES / nombre)],
                           capture_output=True, text=True, encoding='utf-8', errors='replace')
        tam = [int(n) for n in re.findall(r'(\d+) bytes \(\d+%\)', p.stdout)]
        ok = p.returncode == 0 and len(tam) == 2
        fila['compila'] = 'si' if ok else 'no'
        fila['flash_bytes'], fila['ram_estatica_bytes'] = tam if ok else ('pendiente', 'pendiente')
        log.write(f'\n===== {nombre} =====\n{to_ascii(p.stdout + p.stderr)}\n')
        log.flush()
        print(nombre, fila['compila'], fila['flash_bytes'], fila['ram_estatica_bytes'], flush=True)

with open(TABLA, 'w', newline='', encoding='ascii') as f:
    w = csv.DictWriter(f, fieldnames=list(filas[0]))
    w.writeheader()
    w.writerows(filas)
print('Salida del compilador:', LOG.relative_to(ROOT))
