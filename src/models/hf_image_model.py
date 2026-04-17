"""Image model adapter used by the image pipeline."""


class HFImageModel:
    def __init__(self, model=None, id2label_map=None):
        self.model = model
        self._id2label_map = id2label_map or {0: "Fake", 1: "Real"}

    def predict(self, image):
        if self.model is None:
            raise NotImplementedError(
                "Attach a backing image model before calling predict()."
            )

        if hasattr(self.model, "predict"):
            return self.model.predict(image)

        return self.model(image)

    def id2label(self):
        return self._id2label_map
