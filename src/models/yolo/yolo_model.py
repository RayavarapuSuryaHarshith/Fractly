from ultralytics import YOLO

class YOLOModel:
    def __init__(self, model_path='yolov5s.pt'):
        self.model = YOLO(model_path)

    def predict(self, image):
        results = self.model(image)
        return results

    def load_weights(self, weights_path):
        self.model.load(weights_path)

    def save_weights(self, save_path):
        self.model.save(save_path)