import abc

class BaseOCRModel(abc.ABC):
    """
    Base class for all OCR models in the benchmark.
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
        pass

    def close(self) -> None:
        """Release provider resources before the next model is loaded."""
        return None
