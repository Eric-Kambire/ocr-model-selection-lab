import abc

class BaseOCRModel(abc.ABC):
    """
    Base class for all OCR models in the benchmark.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abc.abstractmethod
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
        """
        Performs OCR on the given image.

        ``output_format`` vaut ``prompt``, ``json`` ou ``schema``. Les
        adaptateurs qui ne supportent pas la contrainte structurée peuvent
        l'ignorer ; le parser applicatif validera toujours la réponse.

        ``image_only`` signifie qu'aucun texte applicatif ne doit remplacer le
        prompt système éventuellement embarqué dans le modèle fournisseur.
        
        Returns a dictionary with:
            - "text": Extracted text (clean transcription)
            - "raw_response": Original raw response from model
            - "latency": Execution time in seconds
        """
        pass
