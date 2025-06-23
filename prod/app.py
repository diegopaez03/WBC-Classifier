from utils import load_model, predict_image, plot_predictions, class_names, show_class_distribution, build_class_summary_table, plot_selected_class, colors_ordered, descripciones
import streamlit as st
from PIL import Image
import os

def main():

    model = load_model(os.path.join("prod", "model.pt"))

    st.set_page_config(page_title="WBC Classifier", layout="wide")

    st.title("Detector y Clasificador de Glóbulos Blancos")

    # Sección de introducción general
    with st.container():
        st.subheader("🔍 ¿Qué hace esta herramienta?")
        st.markdown("""
        Esta aplicación permite **detectar, clasificar y contar automáticamente glóbulos blancos** en imágenes de frotis de sangre.  
        Utiliza **visión computacional y deep learning**, orientada a asistir a laboratorios que **no cuentan con equipamiento automático**, como contadores hematológicos.

        Está pensada para **bioquímicos, técnicos de laboratorio, estudiantes** y situaciones donde el tiempo o los recursos son limitados.
        """)

    # Tres columnas informativas
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("🔬 **Detección Inteligente**\n\nDetecta 10 tipos de leucocitos, tanto maduros como inmaduros, directamente desde una imagen.")

    with col2:
        st.success("📊 **Estadísticas Claras**\n\nVisualizá los resultados con gráficos y tablas interactivas que resumen el conteo y porcentaje por clase.")

    with col3:
        st.warning("🧠 **Modelo Avanzado**\n\nEntrenado con YOLOv8n, una red neuronal eficiente y precisa, adaptada al dominio hematológico.")

    # Sección de uso
    st.subheader("⚙️ ¿Cómo usar la aplicación?")
    with st.expander("➡️ Ver instrucciones"):
        st.markdown("""
        1. **Subí una imagen** de un frotis de sangre (formato JPG o PNG).
        2. El sistema detectará automáticamente los glóbulos blancos presentes.
        3. Se mostrarán:
            - Una imagen con cajas delimitadoras y etiquetas.
            - Gráficos de barras y tortas con el conteo por clase.
            - Un resumen numérico con porcentajes.
            - Detalles adicionales al seleccionar una clase.
        """)

    # Problemática y solución
    st.subheader("❗ ¿Qué problemática resuelve?")
    with st.expander("📌 Ver más"):
        st.markdown("""
        El conteo manual de leucocitos es una tarea **lenta, repetitiva y propensa a errores humanos**, especialmente en laboratorios sin automatización.

        Esta herramienta busca:
        - Reducir el tiempo dedicado al conteo.
        - Minimizar errores por fatiga o distracción.
        - Proveer un **apoyo confiable** en entornos con recursos limitados.
        - Ser utilizada también con fines **educativos** y de formación.
        """)

    # Modelo técnico
    st.subheader("🧪 ¿Cómo funciona internamente?")
    with st.expander("🔧 Ver detalles técnicos"):
        st.markdown("""
        - Dataset: **RV-PBS**, con 10 clases de leucocitos y más de 600 imágenes reales.
        - Arquitectura: **YOLOv8n**, liviana pero efectiva, entrenada con *fine-tuning* sobre un modelo preentrenado.
        - Entrenamiento: Realizado en **Google Colab**, con métricas destacadas:
            - **Precisión media** (mAP@0.5) por clase entre 0.90 y 0.99.
            - **Recall promedio** superior al 90% en 8 de las 10 clases.
        """)

    st.markdown("---")  # Línea divisoria para separar visualmente
    st.header("📌 ¡Probá el clasificador ahora mismo!")
    st.text("Para resultados óptimos, asegurate de que la imagen esté bien enfocada y con un nivel de zoom adecuado.")

    uploaded_file = st.file_uploader("Subí una imagen de un frotis de sangre", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image.save(os.path.join("prod", "input.jpg"))  # Temporal

        orig_img, result, boxes = predict_image(model, os.path.join("prod", "input.jpg"))

        img_with_boxes = plot_predictions(orig_img, boxes)
        
        st.header("Resultados de la Predicción")

        # Mostrar los resultados en la app
        col1, col2 = st.columns([1, 1])  

        with col1:
            st.image(img_with_boxes, use_container_width=True)

        
        summary_df, total_count = build_class_summary_table(result, class_names)

        with col2:
            show_class_distribution(result, class_names)

            st.caption("Esta predicción puede no ser precisa. Para un diagnóstico médico, consultá a un profesional de la salud.")

        # Mostrar tabla resumen debajo
        st.subheader("Resumen general")
        st.dataframe(summary_df, use_container_width=True)
        st.markdown(f"**Total de células detectadas:** {total_count}")

        auxcol1, auxcol2 = st.columns([1.1, 1])  
        
        with auxcol1:
            auxcol11, auxcol12 = st.columns([0.7, 0.3])

            with auxcol11:
            
                st.subheader("Detalles por Clase")

                # Selección de clase única
                selected_class = st.selectbox(
                    "Seleccioná una clase para ver detalles:",
                    summary_df["Clase"].tolist()
                )

        if selected_class:            
            # Filtrar resultados de detección para esa clase
            class_ids = [int(cls.item()) for cls in result.boxes.cls]
            selected_class_id = class_names.index(selected_class)
            selected_indices = [i for i, cls_id in enumerate(class_ids) if cls_id == selected_class_id]
            
            with auxcol1:
                
                with auxcol12:
                    st.image(os.path.join("prod", "class_images", f"{selected_class}.jpg"), caption = selected_class, use_container_width=True)

                with auxcol11:
                    st.write(f"Total de células detectadas para esta clase: {len(selected_indices)}")
                st.markdown(f"{descripciones[selected_class]}")

            # Mostrar imagen con solo las detecciones de esta clase
            highlighted_img = plot_selected_class(orig_img, result, class_names, selected_class, colors_ordered)
            
            with auxcol2:
                st.image(highlighted_img, caption=f"Detecciones de la clase {selected_class}", use_container_width=True)

    
    # Advertencia
    st.subheader("⚠️ A tener en cuenta")
    st.warning("""
    🧪 Esta herramienta es un apoyo complementario. No reemplaza el diagnóstico clínico ni la interpretación de un profesional.

    📤 Está pensada para ser usada **en línea**, fácilmente integrable a entornos formativos o como prototipo para sistemas de análisis asistido.
    """)

    st.markdown("💬 ¿Sos profesional bioquímico? [Contactanos](mailto:diego.paezj2000@gmail.com) para participar en la validación del sistema.")


    st.html("<hr>")
    st.html("<p style='text-align: center;'>Hecho por Diego Paez</p>")

if __name__ == "__main__":
    main()

