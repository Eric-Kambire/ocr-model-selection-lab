# Attribution

This dataset is derived from public invoice data released under CC BY 4.0.

## Original Dataset

Kozlowski, Marek; Weichbroth, Pawel (2021), "Samples of electronic invoices", Mendeley Data, V2.

- DOI: `10.17632/tnj49gpmtz.2`
- URL: <https://data.mendeley.com/datasets/tnj49gpmtz/2>
- License: CC BY 4.0

## Derived Work

This release adds:

- clearOCR OCR outputs,
- OCR-to-Markdown reconstruction,
- Markdown-to-JSON invoice extraction produced by a local fine-tuned extraction LLM,
- image-based judge model evaluations,
- Qwen/Gemini visual verification artifacts.

clearOCR:

- Website: <https://clearocr.com>
- clearOCR is an OCR API for PDF, JPG and PNG documents, focused on Polish and English business documents.

## Required Attribution Text

When using or redistributing this derived dataset, include attribution to:

1. Kozlowski and Weichbroth's Mendeley invoice dataset, DOI `10.17632/tnj49gpmtz.2`.
2. clearOCR for the OCR outputs and derived evaluation artifacts.

The structured JSON extraction artifacts are model-generated outputs from a local fine-tuned extraction LLM. They are not manually verified accounting labels.
