# Dataset Schema

The dataset is stored as a readable split layout:

- `<split>/images/*.jpg`
- `<split>/ocr/*.ocr`
- `<split>/markdown/*.mdoc`
- `<split>/json/*.invoice.payload`
- `<split>/judges/*.judges.payload`
- `<split>/metadata.jsonl`

Each split-level `metadata.jsonl` row describes one invoice image and points to companion artifacts with paths relative to that split directory.

## Metadata Fields

- `file_name`: relative path to image file.
- `source_id`: stable source identifier, for example `train:123`.
- `clearocr_text_path`: path to clearOCR OCR text.
- `markdown_path`: path to Markdown reconstruction.
- `invoice_json_path`: path to extracted invoice JSON.
- `judges_path`: path to full visual judge payload.

## clearOCR

The OCR text is stored in `clearocr_text_path`.

## Markdown

The Markdown content is stored in `markdown_path`.

## Invoice JSON

The extracted invoice JSON is stored in `invoice_json_path`.

JSON artifacts use neutral `.payload` suffixes so Hugging Face Dataset Viewer does not try to parse them as standalone dataset files. The file contents are still valid JSON.

Expected invoice JSON shape:

```json
{
  "header": {
    "invoice_no": "...",
    "invoice_date": "...",
    "seller": "...",
    "client": "...",
    "seller_tax_id": "...",
    "client_tax_id": "...",
    "iban": "..."
  },
  "items": [
    {
      "item_desc": "...",
      "item_qty": "...",
      "item_net_price": "...",
      "item_net_worth": "...",
      "item_vat": "...",
      "item_gross_worth": "..."
    }
  ],
  "summary": {
    "total_net_worth": "...",
    "total_vat": "...",
    "total_gross_worth": "..."
  }
}
```

## Judges

The full visual verification payload is stored in `judges_path`.

Judge verdict values:

- `pass`
- `minor_issues`
- `fail`
- `uncertain`
- `error`
- `missing`
