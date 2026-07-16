import time
import os
from models.base import BaseOCRModel

DEFAULT_OCR_PROMPT = """You are a professional layout-preserving OCR engine.
Your task is to transcribe all the text, tables, and handwriting in this image.
Rules:
1. Output ONLY the transcription. Do NOT add greetings, preamble, explanations, notes, or code blocks.
2. Preserve the document layout using Markdown where appropriate (e.g., use '|' for table columns).
3. Format mathematical formulas using LaTeX syntax ($...$ or $$...$$).
4. Transcribe handwriting exactly as written."""


class OllamaOCRModel(BaseOCRModel):
    """
    An OCR model wrapper that uses a local Ollama vision model (e.g., gemma3:1b, llama3.2-vision).
    """
    def __init__(self, model_name: str, prompt: str | None = None):
        super().__init__(model_name)
        self.prompt = prompt.strip() if prompt and prompt.strip() else DEFAULT_OCR_PROMPT
        # Import ollama here to avoid dependency issues if not installed
        try:
            import ollama
            self.client = ollama
        except ImportError:
            self.client = None
            print("Warning: 'ollama' Python library not installed. Please install it using pip.")

    def perform_ocr(self, image_path: str, *, prompt: str | None = None, system_prompt: str | None = None) -> dict:
        """Run OCR and allow a structured workflow to override one prompt."""
        if not self.client:
            return {
                "text": "",
                "raw_response": "Error: Ollama library not installed.",
                "latency": 0.0,
                "status": "failed",
                "error": "Ollama library not installed.",
                "device": "ollama",
            }

        if not os.path.exists(image_path):
            return {
                "text": "",
                "raw_response": f"Error: Image path not found: {image_path}",
                "latency": 0.0,
                "status": "failed",
                "error": f"Image path not found: {image_path}",
                "device": "ollama",
            }

        effective_prompt = prompt.strip() if prompt and prompt.strip() else self.prompt
        effective_system = system_prompt.strip() if system_prompt and system_prompt.strip() else None
        start_time = time.time()
        
        try:
            # Call Ollama chat API with images
            messages = []
            if effective_system:
                messages.append({"role": "system", "content": effective_system})
            messages.append({"role": "user", "content": effective_prompt, "images": [image_path]})
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": 0.0  # Keep transcription deterministic
                }
            )
            
            if isinstance(response, dict):
                message = response.get("message", {})
                extracted_text = message.get("content", "").strip()
                reasoning = message.get("thinking") or message.get("reasoning")
                input_tokens = response.get("prompt_eval_count")
                output_tokens = response.get("eval_count")
                eval_duration = response.get("eval_duration")
            else:
                message = getattr(response, "message", None)
                extracted_text = str(getattr(message, "content", "")).strip()
                reasoning = (
                    getattr(message, "thinking", None)
                    or getattr(message, "reasoning", None)
                )
                input_tokens = getattr(response, "prompt_eval_count", None)
                output_tokens = getattr(response, "eval_count", None)
                eval_duration = getattr(response, "eval_duration", None)
            
            # Clean up potential markdown formatting code blocks wrapped by LLM (e.g. ```markdown ... ```)
            if extracted_text.startswith("```"):
                lines = extracted_text.split("\n")
                if len(lines) >= 2 and lines[-1].startswith("```"):
                    # Remove first and last lines
                    first_line = lines[0]
                    if "markdown" in first_line or "html" in first_line or "text" in first_line or first_line == "```":
                        extracted_text = "\n".join(lines[1:-1]).strip()

            latency = time.time() - start_time
            tokens_per_second = None
            if output_tokens is not None and eval_duration:
                # Ollama durations are expressed in nanoseconds.
                tokens_per_second = float(output_tokens) / (float(eval_duration) / 1_000_000_000)
            
            return {
                "text": extracted_text,
                "raw_response": str(response),
                "reasoning": str(reasoning) if reasoning else None,
                "latency": latency,
                "status": "success",
                "error": None,
                "device": "ollama",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "tokens_per_second": tokens_per_second,
            }
            
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Error during Ollama OCR inference: {str(e)}"
            return {
                "text": "",
                "raw_response": error_msg,
                "latency": latency,
                "status": "failed",
                "error": error_msg,
                "device": "ollama",
            }
