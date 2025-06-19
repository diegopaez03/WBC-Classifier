from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# utils.py
class_names = [
    "Neutrófilo inmaduro", "Basófilo", "Blasto", "Eosinófilo",
    "Linfocito", "Monocito", "Neutrófilo", "Promielocito",
    "Mielocito", "Metamielocito"
]

def load_model(model_path):
    """Carga un modelo YOLO entrenado desde un archivo .pt/.pth"""
    model = YOLO(model_path)
    return model

def predict_image(model, image_path, conf_threshold=0.25):
    """
    Ejecuta la predicción sobre una imagen.
    Retorna la imagen original, resultados de predicción y el tensor de detecciones.
    """
    results = model(image_path, conf=conf_threshold)
    result = results[0]  # Tomar la primer predicción del batch
    return result.orig_img, result, result.boxes

def plot_predictions(image, boxes, class_names=class_names, color=(0, 255, 0), thickness=4):
    """
    Dibuja las bounding boxes y clases sobre una imagen usando OpenCV.
    """
    img = image.copy()

    if isinstance(img, np.ndarray) is False:
        img = np.array(img)

    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy().astype(int)  # (x1, y1, x2, y2)
        cls_id = int(box.cls[0].cpu().numpy())
        conf = float(box.conf[0].item())
        label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        label += f" {conf:.2f}"

        # Dibujar rectángulo
        cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, thickness)

        # Dibujar texto
        font_scale = 2.0
        font_thickness = 2
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        cv2.rectangle(
            img,
            (xyxy[0], xyxy[1] - text_size[1] - 10),
            (xyxy[0] + text_size[0], xyxy[1]),
            color,
            -1,
        )
        cv2.putText(
            img,
            label,
            (xyxy[0], xyxy[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),  # Texto en negro
            font_thickness,
        )

    return img


def count_detected_classes(boxes, class_names=class_names):
    """
    Devuelve un diccionario con la cantidad de detecciones por clase.

    Parámetros:
    - boxes: resultado de result.boxes
    - class_names: lista de nombres de clases

    Retorna:
    - dict: {"NombreClase": cantidad, ...}
    """
    if boxes is None or len(boxes) == 0:
        return {}

    class_ids = [int(cls_id.item()) for cls_id in boxes.cls]
    class_counts = Counter(class_ids)

    # Traducir a nombres si están disponibles
    result = {
        class_names[cls_id] if cls_id < len(class_names) else str(cls_id): count
        for cls_id, count in class_counts.items()
    }

    return result

def build_class_summary_table(pred_result, class_names):
    """
    Crea una tabla con: clase, cantidad, porcentaje y confianza promedio.
    También devuelve el total de objetos detectados.
    """
    if len(pred_result.boxes) == 0:
        return pd.DataFrame(columns=["Clase", "Cantidad", "Porcentaje", "Confianza Promedio"]), 0

    class_ids = [int(cls.item()) for cls in pred_result.boxes.cls]
    confidences = [float(c.item()) for c in pred_result.boxes.conf]

    total = len(class_ids)
    data = {}

    for cls_id, conf in zip(class_ids, confidences):
        class_name = class_names[cls_id]
        if class_name not in data:
            data[class_name] = {"count": 0, "confidences": []}
        data[class_name]["count"] += 1
        data[class_name]["confidences"].append(conf)

    rows = []
    for class_name, values in data.items():
        count = values["count"]
        avg_conf = sum(values["confidences"]) / count
        percentage = (count / total) * 100
        rows.append({
            "Clase": class_name,
            "Cantidad": count,
            "Porcentaje": f"{percentage:.1f}%",
            "Confianza Promedio": f"{avg_conf:.2f}"
        })

    df = pd.DataFrame(rows)
    return df, total

def show_class_distribution(predictions, class_names):
    class_ids = [int(cls.item()) for cls in predictions.boxes.cls]
    class_counts = Counter(class_ids)
    
    classes = [class_names[i] for i in class_counts]
    counts = list(class_counts.values())

    tab1, tab2 = st.tabs(["📊 Gráfico de Barras", "🥧 Gráfico de Torta"])

    with tab1:
        fig_bar, ax = plt.subplots()
        ax.bar(classes, counts, color='purple')
        ax.set_xlabel("Clases")
        ax.set_ylabel("Cantidad")
        ax.set_title("Distribución por clase")
        ax.set_xticklabels(classes, rotation=45)
        st.pyplot(fig_bar)

    with tab2:
        fig_pie, ax = plt.subplots()
        ax.pie(counts, labels=classes, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig_pie)

descripcion1 = """
## Leucocitos maduros

### 1. Monocito
Fagocita microorganismos y se convierte en macrófago en tejidos.  
Núcleo en forma de riñón.

### 2. Basófilo
Participa en reacciones alérgicas y liberación de histamina.  
Núcleo lobulado y gránulos oscuros.

### 3. Neutrófilo
Principal célula en infecciones bacterianas.  
Fagocita patógenos; núcleo multilobulado.

### 4. Eosinófilo
Combate parásitos y participa en alergias.  
Tiene gránulos rosados y núcleo bilobulado.

### 5. Linfocito
Responsable de la respuesta inmune específica (linfocitos B y T).  
Núcleo grande y redondo, citoplasma escaso.
"""

descripcion2 = """
## Leucocitos inmaduros

### 1. Mielocito
Célula más madura que el promielocito.  
Comienza a mostrar gránulos específicos según su tipo final.

### 2. Metamielocito
Precursor directo de la célula en banda.  
Núcleo en forma de riñón, no segmentado todavía.

### 3. Neutrófilo inmaduro
Célula en banda con núcleo en forma de banda o bastón.  
Indica una respuesta activa del sistema inmune (infección aguda).

### 4. Blasto
Célula madre inmadura precursora de leucocitos.  
En exceso puede indicar leucemia.

### 5. Promielocito
Etapa inmadura en la línea de los granulocitos.  
Gran cantidad de gránulos primarios; precursor del mielocito.
"""