import abc

class BaseOCRModel(abc.ABC):
    """Small provider contract consumed by :class:`BenchmarkRunner`.

    Adapters may use any SDK internally, but they must return the normalized
    dictionary documented below. Keeping this boundary stable lets the UI,
    evaluator and reports stay independent from individual model libraries.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abc.abstractmethod
    def perform_ocr(self, image_path: str) -> dict:
        """
        Performs OCR on the given image.
        
        Returns a dictionary with:
            - "text": Extracted text (clean transcription)
            - "raw_response": Original raw response from model
            - "latency": Execution time in seconds
        """
        raise NotImplementedError

    def close(self) -> None:
        """Release provider resources before the next model is loaded."""
        return None
