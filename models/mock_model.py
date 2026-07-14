import os
import json
import random
import time
from models.base import BaseOCRModel

class MockOCRModel(BaseOCRModel):
    """
    A simulated OCR model that returns the ground truth text with added noise
    (character substitutions, missing words, casing changes) and latency.
    """
    def __init__(self, model_name: str = "MockOCR-V1", dataset_json_path: str = "dataset/dataset.json", error_rate: float = 0.05):
        super().__init__(model_name)
        self.dataset_json_path = dataset_json_path
        self.error_rate = error_rate
        self.ground_truth_cache = {}
        self._load_dataset()

    def _load_dataset(self):
        if os.path.exists(self.dataset_json_path):
            try:
                with open(self.dataset_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Cache normalized paths to handle slash mismatches
                    for item in data:
                        normalized_path = self._normalize_path(item["image_path"])
                        self.ground_truth_cache[normalized_path] = item["ground_truth"]
            except Exception as e:
                print(f"Error loading dataset in mock model: {e}")

    def _corrupt_text(self, text: str) -> str:
        if not text:
            return ""
        
        # 1. Word drops
        words = text.split()
        corrupted_words = []
        for word in words:
            if random.random() > (self.error_rate / 2.0):
                corrupted_words.append(word)
        
        corrupted_text = " ".join(corrupted_words)
        
        # 2. Character substitutions
        chars = list(corrupted_text)
        substitutions = {
            'e': 'c', 'o': '0', 'l': '1', 'i': 'l', 's': '5', 't': '7', 
            'a': '@', 'O': '0', 'I': '1', 'S': '5', 'G': '6'
        }
        
        for i, char in enumerate(chars):
            if char in substitutions and random.random() < self.error_rate:
                chars[i] = substitutions[char]
            elif random.random() < (self.error_rate / 4.0):
                # Random swap case
                chars[i] = char.swapcase()
                
        return "".join(chars)

    def perform_ocr(self, image_path: str, *, prompt: str | None = None) -> dict:
        """Return the mock response; ``prompt`` is accepted for adapter parity."""
        start_time = time.time()
        
        # Simulate CPU processing delay
        latency = random.uniform(0.3, 1.2)
        time.sleep(latency)
        
        normalized_img_path = self._normalize_path(image_path)
        
        # If cache is empty (dataset hasn't been loaded or created), reload it
        if not self.ground_truth_cache:
            self._load_dataset()
            
        ground_truth = self.ground_truth_cache.get(normalized_img_path)
        
        if not ground_truth:
            # Fallback if image not found in dataset
            basename = os.path.basename(image_path)
            text = f"Mock transcription for unknown image: {basename}"
        else:
            text = self._corrupt_text(ground_truth)
            
        end_time = time.time()
        actual_latency = end_time - start_time
        
        return {
            "text": text,
            "raw_response": f"MOCK ENGINE RESPONSE:\n---\n{text}\n---\nLatency: {actual_latency:.4f}s",
            "latency": actual_latency,
            "status": "success",
            "error": None,
            "device": "cpu",
            "input_tokens": None,
            "output_tokens": None,
            "tokens_per_second": None,
        }

    @staticmethod
    def _normalize_path(path: str) -> str:
        return os.path.normpath(path.replace("\\", os.sep).replace("/", os.sep))
