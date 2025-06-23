from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

# utils.py
class_names = [
    "Neutrófilo inmaduro", "Basófilo", "Blasto", "Eosinófilo",
    "Linfocito", "Monocito", "Neutrófilo", "Promielocito",
    "Mielocito", "Metamielocito"
]

# Definición directa de colores en BGR (no RGBA)
red_colors = [
    (255, 0, 0),   # Rojo puro (más claro) - Ahora es (B=255, G=0, R=0) para un color rojo
    (220, 0, 0),
    (180, 0, 0),
    (140, 0, 0),
    (100, 0, 0)
]

green_colors = [
    (0, 255, 0),   # Verde puro (más claro)
    (0, 220, 0),   #
    (0, 180, 0),   #
    (0, 140, 0),   #
    (0, 100, 0)    # Verde oscuro
]

# Rearmar lista final según el orden de class_names
colors_ordered = [
    red_colors[0],    # Neutrófilo inmaduro
    green_colors[0],  # Basófilo
    red_colors[1],    # Blasto
    green_colors[1],  # Eosinófilo
    green_colors[2],  # Linfocito
    green_colors[3],  # Monocito
    green_colors[4],  # Neutrófilo
    red_colors[2],    # Promielocito
    red_colors[3],    # Mielocito
    red_colors[4],    # Metamielocito
]


def load_model(model_path):
    """Carga un modelo YOLO entrenado desde un archivo .pt/.pth"""
    model = YOLO(model_path)
    return model

def predict_image(model, image_path, conf_threshold=0.35, iou_threshold=0.4):
    """
    Ejecuta la predicción sobre una imagen.
    Retorna la imagen original, resultados de predicción y el tensor de detecciones.
    """
    results = model(image_path, conf=conf_threshold, iou=iou_threshold)
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
        color = colors_ordered[cls_id % len(colors_ordered)]

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
            (255, 255, 255),  # Fondo blanco
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

    fig_pie, ax = plt.subplots()

    # 1. Quitar el color de fondo del gráfico
    fig_pie.patch.set_alpha(0.0) # Hace el fondo de la figura transparente
    ax.patch.set_alpha(0.0)    # Hace el fondo del área de los ejes transparente

    wedges, texts = ax.pie(counts,
                           startangle=90,
                           radius=0.8,
                           wedgeprops={"edgecolor": "white", 'linewidth': 0.5, 'antialiased': True},
                           shadow=True)

    ax.axis("equal")

    # 3. Mostrar porcentajes en la leyenda
    total = sum(counts)
    labels_with_percentages = []
    for i, (cls_name, count_val) in enumerate(zip(classes, counts)):
        percentage = (count_val / total) * 100
        labels_with_percentages.append(f'{cls_name}: {percentage:.1f}%') # Formato a un decimal

    # Creamos la leyenda con los nuevos labels
    ax.legend(wedges, labels_with_percentages,
              title="Clases",
              loc="lower center",
              bbox_to_anchor=(1, 0, 0.5, 1))

    st.pyplot(fig_pie)

def plot_selected_class(image: Image.Image, result, class_names, selected_class: str, colors):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]

    selected_class_id = class_names.index(selected_class)

    for box, cls, conf in zip(result.boxes.xywh, result.boxes.cls, result.boxes.conf):
        class_id = int(cls.item())
        if class_id != selected_class_id:
            continue

        x_center, y_center, box_w, box_h = box.tolist()

        x1 = int((x_center - box_w / 2))
        y1 = int((y_center - box_h / 2))
        x2 = int((x_center + box_w / 2))
        y2 = int((y_center + box_h / 2))

        label = f"{class_names[class_id]} {conf.item():.2f}"
        color = colors[class_id]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=7)
        cv2.putText(
            img, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 3, lineType=cv2.LINE_AA
        )

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

descripciones = {
    "Neutrófilo inmaduro": "Los **neutrófilos** inmaduros, como las bandas o células en banda, suelen aparecer en la sangre periférica durante infecciones bacterianas agudas. Su presencia puede indicar una respuesta inflamatoria activa o una 'desviación a la izquierda' en la médula ósea, lo que sugiere que el cuerpo está produciendo neutrófilos rápidamente para combatir una infección o proceso inflamatorio severo.",

    "Basófilo": "Los **basófilos** son los glóbulos blancos menos abundantes. Participan en reacciones alérgicas y en la liberación de histamina, lo que provoca síntomas como picazón e inflamación. Un aumento en basófilos puede observarse en enfermedades alérgicas, mieloproliferativas crónicas o ciertas infecciones virales. Su función aún no se comprende completamente, pero se relacionan con procesos inmunitarios complejos.",

    "Blasto": "Los **blastos** son células precursoras inmaduras que normalmente no deben encontrarse en la sangre periférica. Su presencia puede ser signo de leucemia aguda u otros síndromes mielodisplásicos. Identificar blastos es crítico para el diagnóstico temprano de enfermedades hematológicas graves. Su detección debe conducir a estudios adicionales como aspirado medular o inmunofenotipificación.",

    "Eosinófilo": "Los **eosinófilos** intervienen en respuestas alérgicas, parasitarias y procesos inflamatorios. Contienen gránulos citotóxicos que se liberan en presencia de alérgenos o parásitos. Su aumento (eosinofilia) puede observarse en asma, rinitis, infecciones parasitarias, y enfermedades autoinmunes. También pueden formar parte de la respuesta inflamatoria en algunos cánceres. Su observación ayuda al diagnóstico diferencial clínico.",

    "Linfocito": "Los **linfocitos** forman parte clave del sistema inmune adaptativo. Existen linfocitos B, T y NK, que cumplen funciones en la producción de anticuerpos, la destrucción de células infectadas o tumorales y la regulación inmune. Su aumento puede indicar infecciones virales, leucemias linfoides o enfermedades autoinmunes. Su recuento y morfología son fundamentales para el diagnóstico hematológico.",

    "Monocito": "Los **monocitos** son los mayores leucocitos y se transforman en macrófagos cuando migran a tejidos. Participan en la fagocitosis de patógenos, presentación de antígenos y modulación de la respuesta inmune. Su aumento puede observarse en infecciones crónicas, inflamaciones persistentes o trastornos hematológicos. También se asocian con recuperación de infecciones agudas y actividad inmunitaria basal.",

    "Neutrófilo": "Los **neutrófilos** son los glóbulos blancos más abundantes. Actúan como primera línea de defensa ante infecciones bacterianas mediante fagocitosis y liberación de enzimas antimicrobianas. Un aumento sugiere infección o inflamación aguda, mientras que una disminución puede asociarse a inmunodeficiencia o daño medular. Su conteo es clave en el hemograma y análisis clínicos rutinarios.",

    "Promielocito": "Los **promielocitos** son precursores de los granulocitos. Su presencia en sangre periférica es anormal y puede indicar leucemia promielocítica aguda (LPA), una emergencia hematológica que requiere tratamiento urgente. Contienen abundantes gránulos y pueden participar en coagulopatías si proliferan en exceso. Detectarlos precozmente permite intervenir antes de complicaciones graves como la coagulación intravascular diseminada.",

    "Mielocito": "Los **mielocitos** son etapas intermedias en la maduración de granulocitos. Normalmente residen en la médula ósea. Su presencia en sangre puede indicar una regeneración acelerada de neutrófilos, como en infecciones severas o recuperación posquimioterapia. También se observan en síndromes mieloproliferativos. Son indicadores útiles del estado funcional de la médula ósea.",

    "Metamielocito": "Los **metamielocitos** preceden a los neutrófilos maduros. Su hallazgo en sangre sugiere una liberación acelerada de granulocitos desde la médula, generalmente en respuesta a infecciones agudas, estrés inflamatorio o daño medular. Aunque menos inmaduros que los mielocitos, su presencia aún indica actividad hematopoyética anormal. Su proporción ayuda a interpretar la respuesta inmune del paciente."
}

