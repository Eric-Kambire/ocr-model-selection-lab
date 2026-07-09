import re

def levenshtein_distance(seq1, seq2):
    """
    Computes the Levenshtein distance between two sequences.
    Works for both strings (character level) and lists of strings (word level).
    """
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    
    # Optimize space: we only need the current and previous rows
    previous_row = list(range(size_y))
    current_row = [0] * size_y

    for x in range(1, size_x):
        current_row[0] = x
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                current_row[y] = previous_row[y - 1]
            else:
                current_row[y] = min(
                    previous_row[y] + 1,     # Deletion
                    current_row[y - 1] + 1,  # Insertion
                    previous_row[y - 1] + 1  # Substitution
                )
        previous_row = list(current_row)
        
    return previous_row[size_y - 1]

def normalize_text(text: str) -> str:
    """
    Normalizes text for evaluation:
    - Lowercase
    - Replace multiple spaces and newlines with a single space
    - Remove punctuation (optional, but standard for robust word matching)
    """
    if not text:
        return ""
    text = text.lower()
    # Replace carriage returns and newlines with space
    text = re.sub(r'[\r\n]+', ' ', text)
    # Remove punctuation except spaces
    text = re.sub(r'[^\w\s]', '', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def calculate_cer(reference: str, hypothesis: str, normalize: bool = True) -> float:
    """
    Calculates the Character Error Rate (CER).
    """
    if normalize:
        ref = normalize_text(reference)
        hyp = normalize_text(hypothesis)
    else:
        ref = reference or ""
        hyp = hypothesis or ""
        
    if not ref and not hyp:
        return 0.0
    if not ref:
        return 1.0  # Hyp has text, ref is empty: 100% error
        
    dist = levenshtein_distance(ref, hyp)
    # CER may legitimately exceed 100% when the model hallucinates extra text.
    return dist / len(ref)

def calculate_wer(reference: str, hypothesis: str, normalize: bool = True) -> float:
    """
    Calculates the Word Error Rate (WER).
    """
    if normalize:
        ref = normalize_text(reference)
        hyp = normalize_text(hypothesis)
    else:
        ref = reference or ""
        hyp = hypothesis or ""
        
    ref_words = ref.split()
    hyp_words = hyp.split()
    
    if not ref_words and not hyp_words:
        return 0.0
    if not ref_words:
        return 1.0
        
    dist = levenshtein_distance(ref_words, hyp_words)
    # WER may legitimately exceed 100% when insertions dominate.
    return dist / len(ref_words)

def assess_structure_preservation(reference: str, hypothesis: str) -> dict:
    """
    Analyzes how well structural elements like markdown tables and LaTeX mathematical
    formulas are preserved in the hypothesis text.
    """
    ref = reference or ""
    hyp = hypothesis or ""
    
    # 1. Table check (Markdown tables use '|' and '---')
    ref_has_table = "|" in ref and "---" in ref
    hyp_has_table = "|" in hyp and "---" in hyp
    
    # Absence of a structure is not evidence that the model preserved it.
    table_score = None if not ref_has_table else (1.0 if hyp_has_table else 0.0)
    if ref_has_table and not hyp_has_table:
        table_status = "Table detected in Ground Truth, but missing in OCR output."
    elif not ref_has_table and hyp_has_table:
        table_status = "Table detected in OCR output, but not in Ground Truth (potential hallucination)."
    elif ref_has_table and hyp_has_table:
        table_status = "Table structure preserved."
    else:
        table_status = "No tables present in document."

    # 2. LaTeX/Math check (Math formulas wrapped in '$' or '$$')
    ref_has_math = "$" in ref
    hyp_has_math = "$" in hyp
    
    math_score = None if not ref_has_math else (1.0 if hyp_has_math else 0.0)
    if ref_has_math and not hyp_has_math:
        math_status = "Math notation detected in Ground Truth, but missing in OCR output."
    elif not ref_has_math and hyp_has_math:
        math_status = "Math notation detected in OCR output, but not in Ground Truth."
    elif ref_has_math and hyp_has_math:
        math_status = "Math notation preserved."
    else:
        math_status = "No mathematical notation present."

    return {
        "ref_has_table": ref_has_table,
        "hyp_has_table": hyp_has_table,
        "table_preservation_score": table_score,
        "table_status": table_status,
        "ref_has_math": ref_has_math,
        "hyp_has_math": hyp_has_math,
        "math_preservation_score": math_score,
        "math_status": math_status
    }

def evaluate_ocr(reference: str, hypothesis: str) -> dict:
    """
    Runs complete evaluation suite comparing hypothesis text to reference text.
    """
    cer_normalized = calculate_cer(reference, hypothesis, normalize=True)
    cer_raw = calculate_cer(reference, hypothesis, normalize=False)
    wer_normalized = calculate_wer(reference, hypothesis, normalize=True)
    wer_raw = calculate_wer(reference, hypothesis, normalize=False)
    structure = assess_structure_preservation(reference, hypothesis)
    
    # Calculate an overall accuracy score (average of 1 - normalized CER and structure metrics)
    accuracy_score = max(0.0, 1.0 - cer_normalized)
    
    return {
        "accuracy_score": accuracy_score,
        "cer_normalized": cer_normalized,
        "cer_raw": cer_raw,
        "wer_normalized": wer_normalized,
        "wer_raw": wer_raw,
        "structure": structure
    }

def is_valid_iban(iban_str: str) -> bool:
    clean = iban_str.replace(" ", "").upper()
    if len(clean) < 15 or not clean[:2].isalpha() or not clean[2:4].isdigit():
        return False
    rearranged = clean[4:] + clean[:4]
    converted = ""
    for char in rearranged:
        if char.isalpha():
            converted += str(ord(char) - 55)
        else:
            converted += char
    try:
        return int(converted) % 97 == 1
    except ValueError:
        return False

def extract_ibans(text: str) -> list:
    pattern = r'\b[A-Z]{2}[0-9]{2}(?:\s?[A-Z0-9]{4}){2,7}(?:\s?[A-Z0-9]{1,4})?\b'
    raw_matches = re.findall(pattern, text.upper())
    cleaned_matches = []
    for match in raw_matches:
        clean = re.sub(r'\s+', '', match)
        if len(clean) >= 15:
            cleaned_matches.append(clean)
    return cleaned_matches

def extract_amounts(text: str) -> list:
    pattern = r'(?:\b\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?\s?(?:EUR|USD|GBP|CHF|\$|€|£)\b)|(?:\b(?:EUR|USD|GBP|CHF|\$|€|£)\s?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?\b)'
    raw_matches = re.findall(pattern, text, re.IGNORECASE)
    plain_decimals = re.findall(r'\b\d{1,9}[.,]\d{2}\b', text)
    normalized = []
    for m in raw_matches + plain_decimals:
        clean = re.sub(r'\s+', '', m).lower()
        if clean not in normalized:
            normalized.append(clean)
    return normalized

def calculate_numeric_cer(reference: str, hypothesis: str) -> float:
    ref_digits = "".join(c for c in reference if c.isdigit())
    hyp_digits = "".join(c for c in hypothesis if c.isdigit())
    if not ref_digits and not hyp_digits:
        return 0.0
    if not ref_digits:
        return 1.0
    dist = levenshtein_distance(ref_digits, hyp_digits)
    return dist / len(ref_digits)

def evaluate_bankmark(reference: str, hypothesis: str) -> dict:
    """
    Runs banking-specific evaluation (Bankmark mode):
    - Numeric CER (accuracy of numbers)
    - IBAN Exact Match Rate (EMR)
    - IBAN Mathematical Validity Rate of output
    - Amount Exact Match Rate (EMR)
    - Combined Bankmark Score
    """
    ref_ibans = extract_ibans(reference)
    hyp_ibans = extract_ibans(hypothesis)
    
    ref_amounts = extract_amounts(reference)
    hyp_amounts = extract_amounts(hypothesis)
    
    # 1. IBAN Exact Match
    if ref_ibans:
        matched_ibans = sum(1 for iban in ref_ibans if iban in hyp_ibans)
        iban_emr = matched_ibans / len(ref_ibans)
    else:
        iban_emr = None
        
    # 2. IBAN Validity Rate
    if hyp_ibans:
        valid_ibans = sum(1 for iban in hyp_ibans if is_valid_iban(iban))
        iban_valid_rate = valid_ibans / len(hyp_ibans)
    else:
        iban_valid_rate = None
        
    # 3. Amount Exact Match
    if ref_amounts:
        matched_amounts = sum(1 for amt in ref_amounts if amt in hyp_amounts)
        amount_emr = matched_amounts / len(ref_amounts)
    else:
        amount_emr = None
        
    # 4. Numeric CER
    num_cer = calculate_numeric_cer(reference, hypothesis)
    num_accuracy = max(0.0, 1.0 - num_cer)
    
    # 5. General CER (normalized)
    cer_normalized = calculate_cer(reference, hypothesis, normalize=True)
    general_accuracy = max(0.0, 1.0 - cer_normalized)
    general_wer = calculate_wer(reference, hypothesis, normalize=True)
    
    # 6. Combined Bankmark Score
    # Highly penalize errors in IBANs and amounts:
    # 35% IBAN match, 25% Amount match, 20% Numeric Accuracy, 20% General Text Accuracy
    weighted_metrics = [
        (0.35, iban_emr),
        (0.25, amount_emr),
        (0.20, num_accuracy),
        (0.20, general_accuracy),
    ]
    applicable = [(weight, value) for weight, value in weighted_metrics if value is not None]
    total_weight = sum(weight for weight, _ in applicable)
    bankmark_score = sum(weight * value for weight, value in applicable) / total_weight
    
    iban_status = f"IBANs matched: {sum(1 for iban in ref_ibans if iban in hyp_ibans)}/{len(ref_ibans)} ({iban_emr*100:.0f}%)" if ref_ibans else "Not applicable: no IBAN in the reference."
    if hyp_ibans:
        iban_status += f" | Output validity: {valid_ibans}/{len(hyp_ibans)} ({iban_valid_rate*100:.0f}%) valid"
        
    amount_status = f"Amounts matched: {sum(1 for amt in ref_amounts if amt in hyp_amounts)}/{len(ref_amounts)} ({amount_emr*100:.0f}%)" if ref_amounts else "Not applicable: no amount in the reference."
    
    return {
        "bankmark_score": bankmark_score,
        "numeric_cer": num_cer,
        "numeric_accuracy": num_accuracy,
        "iban_emr": iban_emr,
        "iban_valid_rate": iban_valid_rate,
        "iban_status": iban_status,
        "amount_emr": amount_emr,
        "amount_status": amount_status,
        "general_accuracy": general_accuracy,
        "general_wer": general_wer,
    }
