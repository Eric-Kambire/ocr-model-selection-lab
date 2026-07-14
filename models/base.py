import abc

class BaseOCRModel(abc.ABC):
    """
    Base class for all OCR models in the benchmark.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abc.abstractmethod
    def perform_ocr(self, image_path: str, *, prompt: str | None = None) -> dict:
        """
        Performs OCR on the given image.

        ``prompt`` is optional so a structured workflow can reuse one loaded
        vision model for different pages of the same document. Adapters that
        do not support prompts simply ignore it.
        
        Returns a dictionary with:
            - "text": Extracted text (clean transcription)
            - "raw_response": Original raw response from model
            - "latency": Execution time in seconds
        """
        pass
