import cv2
import numpy as np
import os

# 1. Configurar la captura de la cámara web
cap = cv2.VideoCapture(0)

# Leer el primer fotograma y convertirlo a tipo float
_, img = cap.read()
averageValue1 = np.float32(img)

# 2. DEFINIR LAS CARPETAS DE SALIDA EXPLICITAMENTE
# Usamos las rutas que coinciden con la estructura de tu VS Code
folder_data = os.path.join("Open_cv", "data")
folder_out = os.path.join("Open_cv", "data", "out")

# Asegurar que las carpetas existan en el disco duro para evitar errores
if not os.path.exists(folder_data):
    os.makedirs(folder_data)
if not os.path.exists(folder_out):
    os.makedirs(folder_out)

# Contador para que las imágenes se guarden correlativamente (1, 2, 3...)
img_counter = 1

print("=== PROGRAMA INICIADO ===")
print("- Presiona la tecla ESPACIO o la tecla 'S' para capturar.")
print("- Presiona la tecla ESC para salir del programa.")

while True:
    # Capturar el siguiente fotograma
    _, img = cap.read()
    if img is None:
        break
    
    # Actualizar el modelo de fondo acumulado
    cv2.accumulateWeighted(img, averageValue1, 0.02)
    
    # Convertir de vuelta a enteros de 8 bits para poder mostrar y guardar la imagen
    resultingFrames1 = cv2.convertScaleAbs(averageValue1)

    # Mostrar ambas ventanas en tiempo real
    cv2.imshow('Original Frame', img)
    cv2.imshow('Background (Running Average)', resultingFrames1)
    
    # Capturar la tecla presionada (espera 30ms)
    key = cv2.waitKey(30) & 0xFF
    
    # 3. LÓGICA DE GUARDADO AL PRESIONAR 'S' O 'ESPACIO'
    if key == ord('s') or key == 32:  # 32 es el código ASCII del espacio
        # Definir rutas completas para cada archivo por separado
        path_original = os.path.join(folder_data, f"original_{img_counter}.jpg")
        path_resultado = os.path.join(folder_out, f"resultado_{img_counter}.jpg")
        
        # Guardar: la original se va a 'data' y la modificada a 'out'
        cv2.imwrite(path_original, img)
        cv2.imwrite(path_resultado, resultingFrames1)
        
        print(f"¡Captura {img_counter} guardada exitosamente!")
        print(f" -> Original en: {path_original}")
        print(f" -> Resultado en: {path_resultado}")
        
        img_counter += 1  # Incrementar para que la siguiente no sobrescriba la anterior
    
    # Salir del bucle si se presiona la tecla ESC (código 27)
    if key == 27:
        break

# Limpieza y cierre seguro de la cámara y ventanas
cap.release()
cv2.destroyAllWindows()
print("=== PROGRAMA FINALIZADO ===")