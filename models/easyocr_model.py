import os
import time
from models.base import BaseOCRModel

try:
    import easyocr
    import torch
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

class EasyOCRModel(BaseOCRModel):
    def __init__(self):
        super().__init__("EasyOCR-Local")
        if not EASYOCR_AVAILABLE:
            raise ImportError("The 'easyocr' library is not installed. Please run 'pip install easyocr'.")
            
        # Detect CPU/GPU
        self.device_name = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing EasyOCR on device: {self.device_name}...")
        
        # Load reader (French and English support)
        # gpu=True/False depending on CUDA availability
        self.reader = easyocr.Reader(['fr', 'en'], gpu=torch.cuda.is_available())

    def perform_ocr(self, image_path: str, *, prompt: str | None = None) -> dict:
        """Read an image; EasyOCR intentionally ignores generative prompts."""
        start_time = time.time()
        try:
            # Run EasyOCR
            # detail=0 returns only text strings
            results = self.reader.readtext(image_path, detail=0)
            
            # Combine paragraphs
            extracted_text = "\n".join(results)
            latency = time.time() - start_time
            return {
                "text": extracted_text,
                "latency": latency,
                "error": None,
                "status": "success",
                "raw_response": repr(results),
                "device": self.device_name,
                "input_tokens": None,
                "output_tokens": None,
                "tokens_per_second": None,
            }
        except Exception as e:
            latency = time.time() - start_time
            return {
                "text": "",
                "latency": latency,
                "error": str(e),
                "status": "failed",
                "raw_response": None,
                "device": self.device_name,
                "input_tokens": None,
                "output_tokens": None,
                "tokens_per_second": None,
            }
