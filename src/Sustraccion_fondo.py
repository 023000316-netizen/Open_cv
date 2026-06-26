import numpy as np
import cv2

# Load video file
cap = cv2.VideoCapture('Open_cv/data/Peces.mp4')

# --- CONFIGURACIÓN PARA GUARDAR LA SALIDA ---
# Obtenemos el ancho, alto y los FPS del video original para que el de salida sea idéntico
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Definimos el códec y la ruta de salida dentro de la carpeta 'out'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# Usamos isColor=False porque la máscara de sustracción de fondo es en blanco y negro (escala de grises)
out = cv2.VideoWriter('Open_cv/out/resultado_peces.mp4', fourcc, fps, (frame_width, frame_height), isColor=False)

# Create background subtractor (MOG2 handles shadows well)
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Stop if video ends

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # --- GUARDAR EL FRAME EN EL ARCHIVO ---
    out.write(fgmask)
    # --------------------------------------

    # Show original and foreground mask side by side
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Foreground Mask', fgmask)

    # Press 'Esc' to exit
    if cv2.waitKey(30) & 0xFF == 27:
        break

# Release resources
cap.release()
out.release() # ¡Importante cerrar el archivo de salida para que se guarde bien!
cv2.destroyAllWindows()