import streamlit as st
from PIL import Image

from image_analysor import ImageAnalysor


def main():
    st.title("Object Detection with YOLOv5")

    # Sidebar for model selection
    st.sidebar.title("Settings")
    model_name = st.sidebar.selectbox(
        "Select YOLOv5 Model", ["yolov5s", "yolov5m", "yolov5l", "yolov5x"]
    )

    # Image upload
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Run object detection
        analysor = ImageAnalysor(model_name)
        analysor.analyse_image(uploaded_file)

        if analysor.errors:
            st.error(analysor.errors)
        else:
            st.success("Image analysis completed.")
            st.text(analysor.get_result())

            # Display the result image with detections
            st.image(
                analysor.results.render()[0],
                caption="Detected Objects",
                use_column_width=True,
            )


if __name__ == "__main__":
    main()
