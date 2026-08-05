import time
import os
import logging
from models.base import BaseOCRModel
from models.ollama_capabilities import (
    OllamaVisionCapability,
    inspect_ollama_vision_capability,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_OLLAMA_REQUEST_TIMEOUT_SECONDS = 300.0

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
    def __init__(
        self,
        model_name: str,
        prompt: str | None = None,
        *,
        cpu_threads: int | None = None,
        unload_after_task: bool = True,
        request_timeout: float | None = None,
    ):
        super().__init__(model_name)
        self.prompt = prompt.strip() if prompt and prompt.strip() else DEFAULT_OCR_PROMPT
        self.cpu_threads = max(1, int(cpu_threads)) if cpu_threads else None
        self.unload_after_task = bool(unload_after_task)
        self.host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        self.request_timeout = (
            float(request_timeout)
            if request_timeout is not None and float(request_timeout) > 0
            else DEFAULT_OLLAMA_REQUEST_TIMEOUT_SECONDS
        )
        # Le résultat de ``ollama.show`` reste valable pendant la vie de
        # l'adaptateur. Ce cache évite de répéter l'appel pour recto et verso.
        self._vision_capability: OllamaVisionCapability | None = None
        # Import ollama here to avoid dependency issues if not installed
        try:
            import ollama
            # Ne pas utiliser le client global du SDK : son timeout dépend de
            # la version/install du poste. Le client dédié reçoit exactement
            # la valeur affichée dans les paramètres de l'application.
            self.client = ollama.Client(
                host=self.host,
                timeout=self.request_timeout,
            )
            LOGGER.info(
                "Ollama client configured | model=%s | host=%s | request_timeout=%.1fs",
                self.model_name,
                self.host,
                self.request_timeout,
            )
        except ImportError:
            self.client = None
            print("Warning: 'ollama' Python library not installed. Please install it using pip.")

    def perform_ocr(
        self,
        image_path: str,
        *,
        prompt: str | None = None,
        system_prompt: str | None = None,
        output_format: str = "prompt",
        output_schema: dict | None = None,
        image_only: bool = False,
    ) -> dict:
        """Exécute Ollama avec une contrainte JSON facultative.

        La réponse brute reste toujours retournée. Le mode ``schema`` utilise
        le JSON Schema natif d'Ollama ; ``json`` demande seulement un objet JSON
        valide ; ``prompt`` conserve le comportement historique.
        """
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

        capability = self._get_vision_capability()
        if not capability.supported:
            error = (
                f"Le modèle Ollama '{self.model_name}' n'est pas utilisable "
                "pour ce benchmark d'images : "
                f"{capability.reason} Aucune image n'a été analysée."
            )
            LOGGER.error(
                "Ollama vision requirement rejected | model=%s | capabilities=%s | reason=%s",
                self.model_name,
                list(capability.capabilities),
                capability.reason,
            )
            return {
                "text": "",
                "raw_response": error,
                "latency": 0.0,
                "status": "incompatible_model",
                "error": error,
                "device": "ollama",
                "vision_capabilities": list(capability.capabilities),
                "image_submitted": False,
            }

        # En mode image seule, aucun prompt SYSTEM/USER fourni par
        # l'application ne doit masquer le SYSTEM embarqué dans le Modelfile.
        # Le message utilisateur reste présent avec un contenu vide car il
        # transporte l'image dans l'API chat d'Ollama.
        if image_only:
            effective_prompt = ""
            effective_system = None
        else:
            effective_prompt = (
                prompt.strip() if prompt and prompt.strip() else self.prompt
            )
            effective_system = (
                system_prompt.strip()
                if system_prompt and system_prompt.strip()
                else None
            )
        start_time = time.time()
        LOGGER.info(
            "Ollama request started | model=%s | image=%s | request_timeout=%.1fs | image_only=%s",
            self.model_name,
            image_path,
            self.request_timeout,
            bool(image_only),
        )
        
        try:
            # Call Ollama chat API with images
            messages = []
            if effective_system:
                messages.append({"role": "system", "content": effective_system})
            messages.append({"role": "user", "content": effective_prompt, "images": [image_path]})
            options = {"temperature": 0.0}
            if self.cpu_threads:
                options["num_thread"] = self.cpu_threads
            request = {
                "model": self.model_name,
                "messages": messages,
                "options": options,
            }
            if output_format == "schema" and output_schema:
                request["format"] = output_schema
            elif output_format == "json":
                request["format"] = "json"
            response = self.client.chat(
                **request
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
                "vision_capabilities": list(capability.capabilities),
                "image_submitted": True,
                "configured_timeout_seconds": self.request_timeout,
                "prompt_delivery_mode": (
                    "image_only" if image_only else "application_prompt"
                ),
            }
            
        except Exception as e:
            latency = time.time() - start_time
            error_msg = (
                "Error during Ollama OCR inference "
                f"after {latency:.1f}s "
                f"(configured HTTP timeout: {self.request_timeout:.1f}s, "
                f"exception: {type(e).__name__}): {e}"
            )
            LOGGER.exception(
                "Ollama request failed | model=%s | elapsed=%.1fs | "
                "configured_timeout=%.1fs | exception=%s",
                self.model_name,
                latency,
                self.request_timeout,
                type(e).__name__,
            )
            return {
                "text": "",
                "raw_response": error_msg,
                "latency": latency,
                "status": "failed",
                "error": error_msg,
                "device": "ollama",
                "configured_timeout_seconds": self.request_timeout,
            }

    def _get_vision_capability(self) -> OllamaVisionCapability:
        """Inspecte une seule fois le modèle avant tout envoi d'image."""

        if self._vision_capability is None:
            self._vision_capability = inspect_ollama_vision_capability(
                self.client,
                self.model_name,
            )
            LOGGER.info(
                "Ollama capabilities checked | model=%s | vision=%s | capabilities=%s | reason=%s",
                self.model_name,
                self._vision_capability.supported,
                list(self._vision_capability.capabilities),
                self._vision_capability.reason,
            )
        return self._vision_capability

    def close(self) -> None:
        """Demande explicitement au serveur Ollama de libérer le modèle."""
        if not self.client or not self.unload_after_task:
            return
        try:
            self.client.generate(model=self.model_name, prompt="", keep_alive=0)
        except Exception:
            # La libération est une optimisation mémoire : son échec ne doit
            # jamais remplacer un résultat OCR déjà obtenu.
            return
