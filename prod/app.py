from utils import load_model, predict_image, plot_predictions, count_detected_classes, class_names, show_class_distribution, build_class_summary_table, descripcion1, descripcion2
import streamlit as st
from PIL import Image

def main():

    model = load_model("model.pt")

    st.set_page_config(page_title="WBC Classifier", layout="wide")

    st.title("Detector y Clasificador de Glóbulos Blancos")

    st.text("Podes subir una imágen de un frotis de sangre y te diremos cuántos glóbulos blancos hay y qué tipo son.")
    st.text("Para resultados óptimos, asegurate de que la imagen esté bien enfocada y con un nivel de zoom adecuado.")

    uploaded_file = st.file_uploader("Subí una imagen de un frotis de sangre", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image.save("input.jpg")  # Temporal

        orig_img, result, boxes = predict_image(model, "input.jpg")

        img_with_boxes = plot_predictions(orig_img, boxes)
        
        # Mostrar los resultados en la app
        col1, col2 = st.columns([1.1, 1])  # Ajustá el ancho relativo si querés

        with col1:
            st.header("Resultados de la Predicción")
            st.image(img_with_boxes, use_container_width=True)

        with col2:
            show_class_distribution(result, class_names)

        # Mostrar tabla resumen debajo
        st.subheader("Resumen")
        summary_df, total_count = build_class_summary_table(result, class_names)
        st.dataframe(summary_df, use_container_width=True)
        st.markdown(f"**Total de objetos detectados:** {total_count}")
    
    st.caption("Esta predicción puede no ser precisa. Para un diagnóstico médico, consultá a un profesional de la salud.")

    st.markdown("### Referencias")
    icol1, icol2, icol3 = st.columns([0.5, 3, 0.5])
    with icol2:
        st.image("Clases.svg", caption="Clases de glóbulos blancos detectadas por el modelo", use_container_width=True)

    jcol1, jcol2 = st.columns([0.5, 0.5])
    with jcol1:
        st.markdown(descripcion1)
    with jcol2:
        st.markdown(descripcion2)

    st.html("<hr>")
    st.html("<p style='text-align: center;'>Hecho por Diego Paez</p>")

if __name__ == "__main__":
    main()

