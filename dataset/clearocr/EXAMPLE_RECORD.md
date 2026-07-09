# Example Record

This example shows how one invoice is represented in the dataset. Large artifacts are stored as files and referenced from `metadata.jsonl`.

Example source id:

```text
test:0
```

## Metadata Row

```json
{
  "file_name": "images/00000.jpg",
  "source_id": "test:0",
  "clearocr_text_path": "ocr/00000.ocr",
  "markdown_path": "markdown/00000.mdoc",
  "invoice_json_path": "json/00000.invoice.payload",
  "judges_path": "judges/00000.judges.payload"
}
```

## clearOCR Text

Path: `test/ocr/00000.ocr`

```text
Invoice no: 97159829
Date of issue: 09/18/2015

Seller:
Bradley-Andrade
9879 Elizabeth Common
Lake Jonathan, RI 12335
Tax Id: 985-73-8194
IBAN: GB81LZWO32519172531418

Client:
Castro PLC
Unit 9678 Box 9664
DPO AP 69387
Tax Id: 994-72-1270
```

## Markdown Reconstruction

Path: `test/markdown/00000.mdoc`

```html
**Invoice no: 97159829**
Date of issue: 09/18/2015

**Seller:**
Bradley-Andrade
9879 Elizabeth Common
Lake Jonathan, RI 12335
Tax Id: 985-73-8194
IBAN: GB81LZWO32519172531418

**Client:**
Castro PLC
Unit 9678 Box 9664
DPO AP 69387
Tax Id: 994-72-1270
```

## Extracted JSON

Path: `test/json/00000.invoice.payload`

```json
{
  "header": {
    "client": "Castro PLC",
    "client_tax_id": "994-72-1270",
    "iban": "GB81LZWO32519172531418",
    "invoice_date": "09/18/2015",
    "invoice_no": "97159829",
    "seller": "Bradley-Andrade",
    "seller_tax_id": "985-73-8194"
  },
  "items": [
    {
      "item_desc": "12\" Marble Lapis Inlay Chess Table Top With 2\" Pieces & 15\" Wooden Stand W537",
      "item_gross_worth": "978,12",
      "item_net_price": "444,60",
      "item_net_worth": "889,20",
      "item_qty": "2,00",
      "item_vat": "10%"
    }
  ],
  "summary": {
    "total_gross_worth": "$ 978,12",
    "total_net_worth": "$ 889,20",
    "total_vat": "$ 88,92"
  }
}
```

## Verification

The public release includes only records where both verification models returned `pass` for:

- image vs Markdown,
- image vs extracted JSON.
