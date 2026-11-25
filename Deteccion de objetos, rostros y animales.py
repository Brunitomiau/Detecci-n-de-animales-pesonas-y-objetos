
import cv2
import numpy as np
import os
from ultralytics import YOLO

# =====================================
# CONFIGURACION
# =====================================

BASE = os.path.dirname(os.path.abspath(__file__))

RUTA_IMAGEN = os.path.join(BASE, "prueba 2.jpg")

RUTA_RESULTADO = os.path.join(BASE, "resultado.jpg")

# Cargar modelo de inteligencia artificial
print("Cargando modelo YOLO...")

modelo = YOLO("yolov8n.pt")


# =====================================
# CLASIFICAR ANIMALES
# =====================================

ANIMALES = [
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe"
]


# =====================================
# DETECTAR PERSONAS, ANIMALES Y OBJETOS
# =====================================

def detectar_elementos(imagen):

    resultados = modelo.predict(
        source=imagen,
        conf=0.35,
        verbose=False
    )

    conteo = {
        "Personas": 0,
        "Animales": 0,
        "Objetos": 0
    }

    for resultado in resultados:

        for caja in resultado.boxes:

            # Obtener coordenadas
            x1, y1, x2, y2 = map(
                int,
                caja.xyxy[0].tolist()
            )

            # Obtener clase
            clase = int(caja.cls[0].item())

            nombre = modelo.names[clase]

            # Obtener porcentaje de confianza
            confianza = float(
                caja.conf[0].item()
            ) * 100

            # =================================
            # IDENTIFICAR CATEGORIA
            # =================================

            if nombre == "person":

                categoria = "PERSONA"
                color = (0, 255, 0)

                conteo["Personas"] += 1

            elif nombre in ANIMALES:

                categoria = "ANIMAL"
                color = (255, 0, 255)

                conteo["Animales"] += 1

            else:

                categoria = "OBJETO"
                color = (255, 150, 0)

                conteo["Objetos"] += 1

            # =================================
            # DIBUJAR RECTANGULO
            # =================================

            cv2.rectangle(
                imagen,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # =================================
            # MOSTRAR NOMBRE Y CONFIANZA
            # =================================

            texto = f"{nombre} {confianza:.1f}%"

            cv2.putText(
                imagen,
                texto,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            # Mostrar categoria
            cv2.putText(
                imagen,
                categoria,
                (x1, min(y2 + 20, imagen.shape[0] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

    return imagen, conteo


# =====================================
# MOSTRAR RESULTADOS
# =====================================

def mostrar_resultados(conteo):

    print("\n================================")
    print("       RESULTADOS DEL ANALISIS")
    print("================================")

    print("Personas detectadas:", conteo["Personas"])

    print("Animales detectados:", conteo["Animales"])

    print("Objetos detectados:", conteo["Objetos"])

    total = sum(conteo.values())

    print("--------------------------------")

    print("TOTAL DE ELEMENTOS:", total)

    print("================================")


# =====================================
# PROGRAMA PRINCIPAL
# =====================================

def main():

    print("\nIniciando reconocimiento...")

    # Comprobar si existe la imagen
    if not os.path.exists(RUTA_IMAGEN):

        print("ERROR: No se encontro la imagen.")
        print("Verifica que cara 2.jfif este en la carpeta del programa.")

        return

    # Cargar imagen
    imagen = cv2.imread(RUTA_IMAGEN)

    if imagen is None:

        print("ERROR: No se pudo cargar la imagen.")

        return

    print("Imagen cargada correctamente.")

    # Detectar elementos
    resultado, conteo = detectar_elementos(
        imagen.copy()
    )

    # Mostrar resultados
    mostrar_resultados(conteo)

    # Guardar imagen
    cv2.imwrite(
        RUTA_RESULTADO,
        resultado
    )

    print("\nImagen guardada como resultado.jpg")

    # Mostrar ventana
    cv2.imshow(
        "RECONOCIMIENTO DE PERSONAS, ANIMALES Y OBJETOS",
        resultado
    )

    print("\nPresiona cualquier tecla para cerrar.")

    cv2.waitKey(0)

    cv2.destroyAllWindows()


# =====================================
# EJECUTAR PROGRAMA
# =====================================

if __name__ == "__main__":

    main()
