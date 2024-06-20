from image_analysor import ImageAnalysor


def main():
    image_path = r"C:\Users\MBR\Desktop\download.jpg"
    model = ImageAnalysor("yolov5s")
    model.analyse_image(image_path)
    print(model.get_result())


if __name__ == "__main__":
    main()
