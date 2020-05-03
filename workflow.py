from utils import (
    Dataset,
    get_config,
)

config = get_config()


def main():
    worker = Dataset(image_path=config['image_base_path'])
    worker.collect_images()
    worker.detect_faces()
    worker.classify_faces_age_gender()


if __name__ == "__main__":
    main()
