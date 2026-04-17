"""Audio model adapter used by the audio pipeline."""


class HFAudioModel:
    def __init__(self, model=None, id2label_map=None):
        self.model = model
        self._id2label_map = id2label_map or {0: "Fake", 1: "Real"}

    def predict(self, waveform, sample_rate):
        if self.model is None:
            raise NotImplementedError(
                "Attach a backing audio model before calling predict()."
            )

        if hasattr(self.model, "predict"):
            return self.model.predict(waveform, sample_rate)

        return self.model(waveform, sample_rate)

    def id2label(self):
        return self._id2label_map
