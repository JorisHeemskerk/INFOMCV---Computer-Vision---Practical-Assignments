from yolov1_base import YOLOv1Base


def main()-> None:
    model = YOLOv1Base()
    print(
        f"\033[30mModel:\n{model}\nTotal number of parameters: "
        f"\033[1;30m{sum(p.numel() for p in model.parameters()):,}\033[37m"
    )

if __name__ == "__main__":
    main()
