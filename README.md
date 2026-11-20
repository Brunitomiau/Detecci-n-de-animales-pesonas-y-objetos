
import cv2
import numpy as np
import os


# =====================================
# CONFIGURACION
# =====================================

BASE = os.path.dirname(os.path.abspath(__file__))
RUTA_IMAGEN = os.path.join(BASE, "balon.jfif")


# =====================================
# DETECTAR FIGURAS Y ELEMENTOS
# =====================================

def detectar_elementos(imagen):

    gray = cv2.cvtColor(
        imagen,
        cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blurred,
        40,
        120
    )

    kernel = np.ones((3, 3), np.uint8)

    edges = cv2.dilate(
        edges,
        kernel,
        iterations=1
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    conteo = {
        "Triangulos": 0,
        "Cuadrados": 0,
        "Rectangulos": 0,
        "Circulos": 0,
        "Otros": 0
    }

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area < 500:
            continue

        approx = cv2.approxPolyDP(
            cnt,
            0.02 * cv2.arcLength(cnt, True),
            True
        )

        x, y, w, h = cv2.boundingRect(cnt)

        lados = len(approx)

        if lados == 3:

            figura = "Triangulo"
            color = (0, 255, 0)
            conteo["Triangulos"] += 1

        elif lados == 4:

            ratio = w / float(h)

            if 0.90 <= ratio <= 1.10:

                figura = "Cuadrado"
                conteo["Cuadrados"] += 1

            else:

                figura = "Rectangulo"
                conteo["Rectangulos"] += 1

            color = (255, 0, 0)

        elif lados >= 8:

            figura = "Circulo"
            color = (0, 165, 255)
            conteo["Circulos"] += 1

        else:

            figura = "Otro elemento"
            color = (255, 0, 255)
            conteo["Otros"] += 1

        cv2.drawContours(
            imagen,
            [cnt],
            -1,
            color,
            2
        )

        cv2.putText(
            imagen,
            figura,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return imagen, conteo


# =====================================
# DETECTAR REGIONES CON RASGOS VISUALES
# =====================================

def detectar_regiones(imagen):

    gray = cv2.cvtColor(
        imagen,
        cv2.COLOR_BGR2GRAY
    )

    hsv = cv2.cvtColor(
        imagen,
        cv2.COLOR_BGR2HSV
    )

    # Detectar regiones con variaciones de color
    mask = cv2.inRange(
        hsv,
        np.array([0, 30, 30]),
        np.array([180, 255, 255])
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    regiones = 0

    for cnt in contours:

        if cv2.contourArea(cnt) < 1000:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        regiones += 1

        cv2.rectangle(
            imagen,
            (x, y),
            (x + w, y + h),
            (255, 255, 0),
            2
        )

    return regiones


# =====================================
# PROGRAMA PRINCIPAL
# =====================================

def main():

    if not os.path.exists(RUTA_IMAGEN):

        print("Error: no se encontro la imagen.")
        return

    imagen = cv2.imread(RUTA_IMAGEN)

    if imagen is None:

        print("Error: no se pudo cargar la imagen.")
        return

    # Detectar figuras
    resultado, conteo = detectar_elementos(
        imagen.copy()
    )

    # Detectar regiones
    regiones = detectar_regiones(
        resultado
    )

    # Mostrar resultados
    print("\n===== RESULTADOS =====")

    print("Triangulos:", conteo["Triangulos"])
    print("Cuadrados:", conteo["Cuadrados"])
    print("Rectangulos:", conteo["Rectangulos"])
    print("Circulos:", conteo["Circulos"])
    print("Otros elementos:", conteo["Otros"])

    print("Regiones detectadas:", regiones)

    print("======================")

    # Guardar imagen
    cv2.imwrite(
        os.path.join(BASE, "resultado.jpg"),
        resultado
    )

    # Mostrar imagen
    cv2.imshow(
        "Deteccion de elementos",
        resultado
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
