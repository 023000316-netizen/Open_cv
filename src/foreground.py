# Python program to illustrate foreground extraction using GrabCut algorithm
# Goal: Keep people in real colors and change the background to black

import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

# --- CONFIGURACIÓN DE RUTAS ---
image_path = os.path.join('Open_cv', 'data', 'original_foreground.jpg')
output_path = os.path.join('Open_cv', 'data', 'out', 'resultado_foreground.jpg')

# Validar que exista la imagen de origen
if not os.path.exists(image_path):
    print(f"Error: No se encontró la imagen en: {image_path}")
    exit()

# Cargar la imagen original
image = cv2.imread(image_path)
 
# Crear máscara inicial y modelos para GrabCut
mask = np.zeros(image.shape[:2], np.uint8)
backgroundModel = np.zeros((1, 65), np.float64)
foregroundModel = np.zeros((1, 65), np.float64)
 
# --- SELECCIÓN INTERACTIVA ---
print("INSTRUCCIONES:")
print("1. Dibuja un rectángulo que encierre estrechamente a las personas.")
print("   *IMPORTANTE: Deja un margen de pared blanca ARRIBA y a los LADOS por fuera del cuadro.*")
print("2. Presiona ENTER o ESPACIO para procesar.")

rectangle = cv2.selectROI("Selecciona a las personas", image, fromCenter=False, showCrosshair=False)
cv2.destroyWindow("Selecciona a las personas")

# Coordenadas: (x, y, ancho, alto)
x, y, w, h = rectangle

if w > 0 and h > 0:
    
    print("Procesando... Por favor espera unos segundos.")
    # Aplicar GrabCut
    cv2.grabCut(image, mask, rectangle, backgroundModel, foregroundModel, 10, cv2.GC_INIT_WITH_RECT)
     
    # --- CORRECCIÓN ABSOLUTA DE LA MÁSCARA ---
    # Forzamos a que el objeto (ustedes) tenga valor 1 y el fondo tenga valor 0
    # Si por la iluminación vuelve a salir al revés, cambia el 0 por 1 y el 1 por 0 en la línea de abajo:
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
     
    # Multiplicar la máscara por la imagen original
    # Resultado: Personas a color, fondo negro
    image_segmented = image * mask2[:, :, np.newaxis]
     
    # --- GUARDAR EL RECORTE EN LA CARPETA OUT ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image_segmented)
    print(f"¡Proceso terminado! Imagen guardada en: {output_path}")

    # --- MOSTRAR RESULTADOS CON MATPLOTLIB ---
    plt.figure(figsize=(10, 5))

    # Imagen Original
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # Resultado correcto: Personas intactas, fondo cambiado
    plt.subplot(1, 2, 2)
    plt.title('Segmented Image (People Intact)')
    plt.imshow(cv2.cvtColor(image_segmented, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

else:
    print("No seleccionaste un área válida. Ejecución cancelada.")