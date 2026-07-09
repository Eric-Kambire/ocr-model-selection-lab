import os
import json
import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# Define directories
DATASET_DIR = "dataset"
TABLES_DIR = os.path.join(DATASET_DIR, "tables")
HANDWRITTEN_DIR = os.path.join(DATASET_DIR, "handwritten")
LAYOUT_DIR = os.path.join(DATASET_DIR, "complex_layout")

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(HANDWRITTEN_DIR, exist_ok=True)
os.makedirs(LAYOUT_DIR, exist_ok=True)

HANDWRITING_FONT_URL = "https://github.com/google/fonts/raw/main/ofl/architectsdaughter/ArchitectsDaughter-Regular.ttf"
HANDWRITING_FONT_PATH = os.path.join(DATASET_DIR, "handwriting_font.ttf")

def download_handwriting_font():
    if not os.path.exists(HANDWRITING_FONT_PATH):
        print("Downloading handwriting font Caveat...")
        download_file(HANDWRITING_FONT_URL, HANDWRITING_FONT_PATH)

def get_handwriting_font(size=18):
    download_handwriting_font()
    if os.path.exists(HANDWRITING_FONT_PATH):
        try:
            return ImageFont.truetype(HANDWRITING_FONT_PATH, size)
        except Exception:
            pass
    return get_font(size - 4)

# Try to find a standard Windows font, fallback to default if not found
def get_font(size=14):
    font_paths = [
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\calibri.ttf",
        "C:\\Windows\\Fonts\\segoeui.ttf",
        "arial.ttf"
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except IOError:
                continue
    return ImageFont.load_default()

def get_wikimedia_image_url(page_url):
    print(f"Scraping Wikimedia page {page_url} for raw image URL...")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        res = requests.get(page_url, headers=headers, timeout=15)
        res.raise_for_status()
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(res.text, "html.parser")
        div = soup.find("div", class_="fullMedia")
        if div:
            a = div.find("a")
            if a and 'href' in a.attrs:
                url = a['href']
                if url.startswith("//"):
                    url = "https:" + url
                print(f"Found Wikimedia raw URL: {url}")
                return url
    except Exception as e:
        print(f"Scraping Wikimedia failed: {e}")
    return None

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(response.content)
        print("Download successful!")
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

def generate_synthetic_table():
    path = os.path.join(TABLES_DIR, "synthetic_table_1.png")
    print(f"Generating synthetic table at {path}...")
    
    img = Image.new("RGB", (600, 300), color="white")
    draw = ImageDraw.Draw(img)
    font = get_font(16)
    
    # Define columns and rows
    headers = ["ID", "Product", "Qty", "Price"]
    rows = [
        ["01", "Laptop", "1", "$999"],
        ["02", "Mouse", "2", "$25"],
        ["03", "Keyboard", "1", "$75"],
        ["04", "Monitor", "1", "$150"]
    ]
    
    col_widths = [80, 200, 100, 120]
    row_height = 40
    start_x = 50
    start_y = 50
    
    # Draw header text and background
    draw.rectangle([start_x, start_y, start_x + sum(col_widths), start_y + row_height], fill="#e0e0e0")
    
    # Draw headers
    current_x = start_x
    for i, h in enumerate(headers):
        draw.text((current_x + 10, start_y + 10), h, fill="black", font=font)
        current_x += col_widths[i]
        
    # Draw rows
    for r_idx, row in enumerate(rows):
        current_y = start_y + (r_idx + 1) * row_height
        current_x = start_x
        for c_idx, val in enumerate(row):
            draw.text((current_x + 10, current_y + 10), val, fill="black", font=font)
            current_x += col_widths[c_idx]
            
    # Draw grid lines
    total_width = sum(col_widths)
    total_height = (len(rows) + 1) * row_height
    
    for i in range(len(rows) + 2):
        y = start_y + i * row_height
        draw.line([start_x, y, start_x + total_width, y], fill="black", width=1)
        
    current_x = start_x
    for i in range(len(col_widths) + 1):
        draw.line([current_x, start_y, current_x, start_y + total_height], fill="black", width=1)
        if i < len(col_widths):
            current_x += col_widths[i]
            
    img.save(path)
    print("Synthetic table generated!")
    
    # Define Markdown representation for ground truth
    markdown_gt = (
        "| ID | Product | Qty | Price |\n"
        "| --- | --- | --- | --- |\n"
        "| 01 | Laptop | 1 | $999 |\n"
        "| 02 | Mouse | 2 | $25 |\n"
        "| 03 | Keyboard | 1 | $75 |\n"
        "| 04 | Monitor | 1 | $150 |"
    )
    return path, markdown_gt

def generate_synthetic_layout():
    path = os.path.join(LAYOUT_DIR, "synthetic_layout_1.png")
    print(f"Generating synthetic layout at {path}...")
    
    img = Image.new("RGB", (600, 500), color="white")
    draw = ImageDraw.Draw(img)
    
    title_font = get_font(20)
    body_font = get_font(13)
    
    # Title
    draw.text((50, 40), "COMPARATIVE ANALYSIS OF OCR SYSTEMS", fill="black", font=title_font)
    draw.line([50, 75, 550, 75], fill="gray", width=2)
    
    # Left column
    left_col_text = [
        "SECTION 1: INTRODUCTION",
        "Optical Character Recognition (OCR) is",
        "a technology that enables the conversion",
        "of different types of documents, such as",
        "scanned paper documents, PDF files or",
        "images captured by a digital camera into",
        "editable and searchable data. The main",
        "objective is to automate data entry",
        "tasks and improve document access."
    ]
    
    current_y = 100
    for line in left_col_text:
        draw.text((50, current_y), line, fill="black", font=body_font)
        current_y += 22
        
    # Right column
    right_col_text = [
        "SECTION 2: METHODOLOGY",
        "Our benchmark methodology evaluates",
        "OCR engines on accuracy and speed.",
        "We calculate Word Error Rate (WER)",
        "and Character Error Rate (CER).",
        "We also assess how well models",
        "preserve document structure like",
        "tables and mathematical equations",
        "in the output format."
    ]
    
    current_y = 100
    for line in right_col_text:
        draw.text((320, current_y), line, fill="black", font=body_font)
        current_y += 22
        
    # Draw a line between columns
    draw.line([300, 95, 300, 300], fill="lightgray", width=1)
    
    # Callout Box at bottom
    draw.rectangle([50, 330, 550, 420], outline="black", width=1, fill="#f9f9f9")
    draw.text((65, 340), "IMPORTANT NOTICE:", fill="black", font=body_font)
    draw.text((65, 365), "Running models on CPU will take longer.", fill="black", font=body_font)
    draw.text((65, 390), "Ensure that Ollama is active on localhost.", fill="black", font=body_font)
    
    # Footer
    draw.line([50, 450, 550, 450], fill="gray", width=1)
    draw.text((50, 460), "Page 1 of 1 - Benchmark Suite", fill="gray", font=body_font)
    
    img.save(path)
    print("Synthetic layout generated!")
    
    # Ground truth representation
    gt = (
        "COMPARATIVE ANALYSIS OF OCR SYSTEMS\n\n"
        "SECTION 1: INTRODUCTION\n"
        "Optical Character Recognition (OCR) is a technology that enables the conversion "
        "of different types of documents, such as scanned paper documents, PDF files or "
        "images captured by a digital camera into editable and searchable data. The main "
        "objective is to automate data entry tasks and improve document access.\n\n"
        "SECTION 2: METHODOLOGY\n"
        "Our benchmark methodology evaluates OCR engines on accuracy and speed. "
        "We calculate Word Error Rate (WER) and Character Error Rate (CER). "
        "We also assess how well models preserve document structure like "
        "tables and mathematical equations in the output format.\n\n"
        "IMPORTANT NOTICE:\n"
        "Running models on CPU will take longer.\n"
        "Ensure that Ollama is active on localhost.\n\n"
        "Page 1 of 1 - Benchmark Suite"
    )
    return path, gt

def generate_valid_iban(country_code, bank_code, branch_code, account_num, national_key):
    bban = f"{bank_code}{branch_code}{account_num}{national_key}"
    temp_iban = f"{bban}{country_code}00"
    converted = ""
    for char in temp_iban:
        if char.isalpha():
            converted += str(ord(char.upper()) - 55)
        else:
            converted += char
    val = int(converted)
    check_digits = 98 - (val % 97)
    raw_iban = f"{country_code}{check_digits:02d}{bban}"
    formatted = " ".join(raw_iban[i:i+4] for i in range(0, len(raw_iban), 4))
    return formatted

def generate_synthetic_bank_receipt():
    path = os.path.join(LAYOUT_DIR, "bank_receipt_1.png")
    print(f"Generating synthetic bank receipt at {path}...")
    
    img = Image.new("RGB", (600, 500), color="white")
    draw = ImageDraw.Draw(img)
    
    title_font = get_font(18)
    section_font = get_font(14)
    body_font = get_font(12)
    
    # Header
    draw.text((50, 40), "GLOBAL BANKING CORP", fill="#1e3d59", font=title_font)
    draw.text((50, 65), "OFFICIAL TRANSFER RECEIPT", fill="gray", font=body_font)
    draw.line([50, 85, 550, 85], fill="#1e3d59", width=2)
    
    # Metadata
    draw.text((50, 100), "Date: 2026-07-08", fill="black", font=body_font)
    draw.text((50, 120), "Status: COMPLETED", fill="green", font=body_font)
    draw.text((50, 140), "Transaction Reference: GBC-TXN-20260708-A9", fill="black", font=body_font)
    
    sender_iban = generate_valid_iban("FR", "30006", "10120", "01234567890", "89")
    beneficiary_iban = generate_valid_iban("FR", "10007", "20450", "98765432101", "42")
    
    # Sender details
    draw.text((50, 175), "SENDER DETAILS", fill="#1e3d59", font=section_font)
    draw.line([50, 195, 270, 195], fill="#1e3d59", width=1)
    draw.text((50, 205), "Account Holder: JOHN DOE", fill="black", font=body_font)
    draw.text((50, 225), f"Debit Account (IBAN):\n{sender_iban}", fill="black", font=body_font)
    
    # Beneficiary details
    draw.text((320, 175), "BENEFICIARY DETAILS", fill="#1e3d59", font=section_font)
    draw.line([320, 195, 550, 195], fill="#1e3d59", width=1)
    draw.text((320, 205), "Beneficiary Name: ACME SOLUTIONS", fill="black", font=body_font)
    draw.text((320, 225), f"Credit Account (IBAN):\n{beneficiary_iban}", fill="black", font=body_font)
    
    # Draw boxes around details
    draw.rectangle([45, 170, 280, 270], outline="lightgray", width=1)
    draw.rectangle([315, 170, 555, 270], outline="lightgray", width=1)
    
    # Transfer details
    draw.text((50, 295), "TRANSFER DETAILS", fill="#1e3d59", font=section_font)
    draw.line([50, 315, 550, 315], fill="#1e3d59", width=1)
    
    draw.text((50, 325), "Amount: 1250.00 EUR", fill="black", font=body_font)
    draw.text((50, 345), "Reason: Invoice #87654-A", fill="black", font=body_font)
    
    # Security Stamp box at bottom
    draw.rectangle([50, 380, 550, 440], outline="red", width=1, fill="#fff5f5")
    draw.text((65, 390), "SECURITY VALIDATION STAMP - CONFIRMED", fill="red", font=section_font)
    draw.text((65, 415), "Verified Beneficiary. No suspicious behavior patterns detected.", fill="red", font=body_font)
    
    # Footer
    draw.text((50, 460), "Document generated electronically. Page 1/1.", fill="gray", font=body_font)
    
    img.save(path)
    print("Synthetic bank receipt generated!")
    
    gt = (
        "GLOBAL BANKING CORP\n"
        "OFFICIAL TRANSFER RECEIPT\n\n"
        "Date: 2026-07-08\n"
        "Status: COMPLETED\n"
        "Transaction Reference: GBC-TXN-20260708-A9\n\n"
        "SENDER DETAILS\n"
        "Account Holder: JOHN DOE\n"
        f"Debit Account (IBAN): {sender_iban}\n\n"
        "BENEFICIARY DETAILS\n"
        "Beneficiary Name: ACME SOLUTIONS\n"
        f"Credit Account (IBAN): {beneficiary_iban}\n\n"
        "TRANSFER DETAILS\n"
        "Amount: 1250.00 EUR\n"
        "Reason: Invoice #87654-A\n\n"
        "SECURITY VALIDATION STAMP - CONFIRMED\n"
        "Verified Beneficiary. No suspicious behavior patterns detected.\n"
        "Document generated electronically. Page 1/1."
    )
    return path, gt, sender_iban, beneficiary_iban

def apply_image_noise(image_path, noise_type):
    """
    Applies blur or rotation to an image and saves the resulting file.
    Returns the path of the new image.
    """
    try:
        img = Image.open(image_path)
        base, ext = os.path.splitext(image_path)
        new_path = f"{base}_{noise_type}{ext}"
        
        if noise_type == "blurry":
            noisy_img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
            noisy_img.save(new_path)
        elif noise_type == "skewed":
            noisy_img = img.rotate(angle=2.0, resample=Image.BICUBIC, expand=True, fillcolor="white")
            noisy_img.save(new_path)
        elif noise_type == "skewed_neg":
            noisy_img = img.rotate(angle=-2.0, resample=Image.BICUBIC, expand=True, fillcolor="white")
            noisy_img.save(new_path)
        else:
            return image_path
        print(f"Applied noise '{noise_type}' -> saved to {new_path}")
        return new_path
    except Exception as e:
        print(f"Error applying noise: {e}")
        return image_path

def generate_handwritten_bank_receipt():
    path = os.path.join(LAYOUT_DIR, "bank_receipt_handwritten.png")
    print(f"Generating handwritten bank receipt at {path}...")
    
    img = Image.new("RGB", (600, 500), color="white")
    draw = ImageDraw.Draw(img)
    
    title_font = get_font(18)
    section_font = get_font(14)
    body_font = get_font(12)
    hand_font = get_handwriting_font(20)
    
    # Printed Template (Headers, Labels)
    draw.text((50, 40), "GLOBAL BANKING CORP", fill="#1e3d59", font=title_font)
    draw.text((50, 65), "OFFICIAL TRANSFER REQUEST FORM", fill="gray", font=body_font)
    draw.line([50, 85, 550, 85], fill="#1e3d59", width=2)
    
    draw.text((50, 100), "Date:", fill="gray", font=body_font)
    draw.text((50, 120), "Status:", fill="gray", font=body_font)
    draw.text((50, 140), "Request Reference:", fill="gray", font=body_font)
    
    sender_iban = generate_valid_iban("FR", "30006", "10120", "98765432109", "12")
    beneficiary_iban = generate_valid_iban("FR", "10007", "20450", "01234567890", "55")
    
    # Hand-filled metadata
    draw.text((90, 94), "2026-07-10", fill="#104c91", font=hand_font)
    draw.text((100, 114), "APPROVED", fill="#104c91", font=hand_font)
    draw.text((180, 134), "GBC-REQ-20260710-X5", fill="#104c91", font=hand_font)
    
    # Sender details template
    draw.text((50, 175), "SENDER DETAILS", fill="#1e3d59", font=section_font)
    draw.line([50, 195, 270, 195], fill="#1e3d59", width=1)
    draw.text((50, 205), "Account Holder:", fill="gray", font=body_font)
    draw.text((50, 225), "Debit Account (IBAN):", fill="gray", font=body_font)
    
    # Hand-filled sender
    draw.text((150, 199), "MARIE DUBOIS", fill="#104c91", font=hand_font)
    draw.text((50, 245), sender_iban, fill="#104c91", font=hand_font)
    
    # Beneficiary details template
    draw.text((320, 175), "BENEFICIARY DETAILS", fill="#1e3d59", font=section_font)
    draw.line([320, 195, 550, 195], fill="#1e3d59", width=1)
    draw.text((320, 205), "Beneficiary Name:", fill="gray", font=body_font)
    draw.text((320, 225), "Credit Account (IBAN):", fill="gray", font=body_font)
    
    # Hand-filled beneficiary
    draw.text((430, 199), "PIERRE MARTIN", fill="#104c91", font=hand_font)
    draw.text((320, 245), beneficiary_iban, fill="#104c91", font=hand_font)
    
    # Draw boxes
    draw.rectangle([45, 170, 280, 280], outline="lightgray", width=1)
    draw.rectangle([315, 170, 555, 280], outline="lightgray", width=1)
    
    # Transfer details template
    draw.text((50, 305), "TRANSFER DETAILS", fill="#1e3d59", font=section_font)
    draw.line([50, 325, 550, 325], fill="#1e3d59", width=1)
    
    draw.text((50, 335), "Amount:", fill="gray", font=body_font)
    draw.text((50, 355), "Reason:", fill="gray", font=body_font)
    
    # Hand-filled transfer details
    draw.text((110, 329), "950.00 EUR", fill="#104c91", font=hand_font)
    draw.text((110, 349), "Rent July 2026", fill="#104c91", font=hand_font)
    
    # Signatures
    draw.text((350, 380), "Authorized Signature:", fill="gray", font=body_font)
    draw.text((350, 400), "Marie Dubois", fill="#104c91", font=hand_font)
    
    # Footer
    draw.text((50, 460), "Document scanned via mobile banking application. Page 1/1.", fill="gray", font=body_font)
    
    img.save(path)
    
    gt = (
        "GLOBAL BANKING CORP\n"
        "OFFICIAL TRANSFER REQUEST FORM\n\n"
        "Date: 2026-07-10\n"
        "Status: APPROVED\n"
        "Request Reference: GBC-REQ-20260710-X5\n\n"
        "SENDER DETAILS\n"
        "Account Holder: MARIE DUBOIS\n"
        f"Debit Account (IBAN): {sender_iban}\n\n"
        "BENEFICIARY DETAILS\n"
        "Beneficiary Name: PIERRE MARTIN\n"
        f"Credit Account (IBAN): {beneficiary_iban}\n\n"
        "TRANSFER DETAILS\n"
        "Amount: 950.00 EUR\n"
        "Reason: Rent July 2026\n\n"
        "Authorized Signature:\n"
        "Marie Dubois\n"
        "Document scanned via mobile banking application. Page 1/1."
    )
    return path, gt, sender_iban, beneficiary_iban

def generate_handwritten_financial_table():
    path = os.path.join(TABLES_DIR, "financial_table_handwritten.png")
    print(f"Generating handwritten table at {path}...")
    
    img = Image.new("RGB", (650, 300), color="white")
    draw = ImageDraw.Draw(img)
    font = get_font(15)
    hand_font = get_handwriting_font(22)
    
    headers = ["Quarter", "Revenue", "Operating Expenses", "Net Profit"]
    rows = [
        ["Q1 2026", "$1,250,000", "$950,000", "$300,000"],
        ["Q2 2026", "$1,420,000", "$980,000", "$440,000"],
        ["Q3 2026", "$1,380,000", "$960,000", "$420,000"],
        ["Q4 2026", "$1,650,000", "$1,020,000", "$630,000"]
    ]
    
    col_widths = [120, 160, 200, 120]
    row_height = 40
    start_x = 25
    start_y = 50
    
    # Header Draw
    draw.rectangle([start_x, start_y, start_x + sum(col_widths), start_y + row_height], fill="#f5f5f5")
    
    # Draw headers
    current_x = start_x
    for i, h in enumerate(headers):
        draw.text((current_x + 10, start_y + 12), h, fill="black", font=font)
        current_x += col_widths[i]
        
    # Draw rows
    for r_idx, row in enumerate(rows):
        current_y = start_y + (r_idx + 1) * row_height
        current_x = start_x
        for c_idx, val in enumerate(row):
            draw.text((current_x + 10, current_y + 6), val, fill="#1c3b57", font=hand_font)
            current_x += col_widths[c_idx]
            
    # Grid lines
    total_width = sum(col_widths)
    total_height = (len(rows) + 1) * row_height
    
    for i in range(len(rows) + 2):
        y = start_y + i * row_height
        draw.line([start_x, y, start_x + total_width, y], fill="gray", width=1)
        
    current_x = start_x
    for i in range(len(col_widths) + 1):
        draw.line([current_x, start_y, current_x, start_y + total_height], fill="gray", width=1)
        if i < len(col_widths):
            current_x += col_widths[i]
            
    img.save(path)
    
    markdown_gt = (
        "| Quarter | Revenue | Operating Expenses | Net Profit |\n"
        "| --- | --- | --- | --- |\n"
        "| Q1 2026 | $1,250,000 | $950,000 | $300,000 |\n"
        "| Q2 2026 | $1,420,000 | $980,000 | $440,000 |\n"
        "| Q3 2026 | $1,380,000 | $960,000 | $420,000 |\n"
        "| Q4 2026 | $1,650,000 | $1,020,000 | $630,000 |"
    )
    return path, markdown_gt

def _draw_handwritten_text(filename, lines, ground_truth, bg_color="#faf8f0"):
    """Helper: render lines of text using handwriting font on a paper-like background."""
    path = os.path.join(HANDWRITTEN_DIR, filename)
    hand_font = get_handwriting_font(22)
    height = max(300, 60 + len(lines) * 30)
    img = Image.new("RGB", (600, height), color=bg_color)
    draw = ImageDraw.Draw(img)
    y = 30
    for line in lines:
        draw.text((40, y), line, fill="#1a1a2e", font=hand_font)
        y += 28
    img.save(path)
    return path, ground_truth


def _draw_handwritten_table(filename, title, headers, rows, ground_truth, bg_color="#faf8f0"):
    """Helper: render a table with handwriting font on a paper-like background."""
    path = os.path.join(HANDWRITTEN_DIR, filename)
    hand_font = get_handwriting_font(20)
    title_font = get_handwriting_font(24)
    n_cols = len(headers)
    col_w = 600 // n_cols
    row_h = 36
    width = max(620, n_cols * col_w + 40)
    height = 80 + (len(rows) + 1) * row_h + 20
    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)
    draw.text((20, 15), title, fill="#1a1a2e", font=title_font)
    sx, sy = 20, 60
    tw = n_cols * col_w
    th = (len(rows) + 1) * row_h
    cx = sx
    for h in headers:
        draw.text((cx + 6, sy + 6), h, fill="#333", font=hand_font)
        cx += col_w
    for ri, row in enumerate(rows):
        cy = sy + (ri + 1) * row_h
        cx = sx
        for val in row:
            draw.text((cx + 6, cy + 4), str(val), fill="#1a1a2e", font=hand_font)
            cx += col_w
    for i in range(len(rows) + 2):
        y = sy + i * row_h
        draw.line([sx, y, sx + tw, y], fill="#888", width=1)
    for i in range(n_cols + 1):
        x = sx + i * col_w
        draw.line([x, sy, x, sy + th], fill="#888", width=1)
    img.save(path)
    return path, ground_truth


def generate_handwritten_texts():
    """Generate 10 varied handwritten text images."""
    print("Generating 10 handwritten text images...")
    records = []
    samples = [
        {"file": "hw_note_1.png", "lines": ["Dear Mum,", "I arrived safely in Paris.", "The weather is lovely today.", "I will call you tonight.", "Love, Sophie"], "gt": "Dear Mum,\nI arrived safely in Paris.\nThe weather is lovely today.\nI will call you tonight.\nLove, Sophie", "desc": "Handwritten personal letter note."},
        {"file": "hw_note_2.png", "lines": ["Meeting Notes - July 9, 2026", "Attendees: Alice, Bob, Clara", "Topic: Q3 Budget Review", "Action: Submit reports by Friday", "Next meeting: July 16, 2026"], "gt": "Meeting Notes - July 9, 2026\nAttendees: Alice, Bob, Clara\nTopic: Q3 Budget Review\nAction: Submit reports by Friday\nNext meeting: July 16, 2026", "desc": "Handwritten meeting notes."},
        {"file": "hw_note_3.png", "lines": ["Shopping List", "- Milk (2 liters)", "- Bread (whole wheat)", "- Eggs (12 pack)", "- Butter", "- Apples (1 kg)", "- Rice (basmati)", "- Chicken breast (500g)"], "gt": "Shopping List\n- Milk (2 liters)\n- Bread (whole wheat)\n- Eggs (12 pack)\n- Butter\n- Apples (1 kg)\n- Rice (basmati)\n- Chicken breast (500g)", "desc": "Handwritten shopping list."},
        {"file": "hw_note_4.png", "lines": ["Dr. Martin Leclerc", "42 Rue de la Paix", "75002 Paris, France", "Tel: +33 1 42 56 78 90", "Email: m.leclerc@cabinet.fr"], "gt": "Dr. Martin Leclerc\n42 Rue de la Paix\n75002 Paris, France\nTel: +33 1 42 56 78 90\nEmail: m.leclerc@cabinet.fr", "desc": "Handwritten contact/address card."},
        {"file": "hw_note_5.png", "lines": ["To Do - Monday", "1. Send invoice #4521", "2. Call Mr. Dupont at 14h", "3. Review contract draft", "4. Order office supplies", "5. Prepare presentation slides"], "gt": "To Do - Monday\n1. Send invoice #4521\n2. Call Mr. Dupont at 14h\n3. Review contract draft\n4. Order office supplies\n5. Prepare presentation slides", "desc": "Handwritten to-do list."},
        {"file": "hw_note_6.png", "lines": ["MEMO", "From: Director of Operations", "To: All Department Heads", "Date: July 9, 2026", "Subject: Office Closure", "", "The office will be closed on", "July 14 for Bastille Day.", "Please plan accordingly."], "gt": "MEMO\nFrom: Director of Operations\nTo: All Department Heads\nDate: July 9, 2026\nSubject: Office Closure\n\nThe office will be closed on\nJuly 14 for Bastille Day.\nPlease plan accordingly.", "desc": "Handwritten office memo."},
        {"file": "hw_note_7.png", "lines": ["Recipe: Crepes", "Ingredients:", "250g flour", "4 eggs", "500ml milk", "50g melted butter", "1 pinch of salt", "Mix until smooth.", "Cook 2 min each side."], "gt": "Recipe: Crepes\nIngredients:\n250g flour\n4 eggs\n500ml milk\n50g melted butter\n1 pinch of salt\nMix until smooth.\nCook 2 min each side.", "desc": "Handwritten recipe."},
        {"file": "hw_note_8.png", "lines": ["Patient: Jean Dupuis", "DOB: 15/03/1985", "Blood Pressure: 130/85", "Heart Rate: 72 bpm", "Weight: 78 kg", "Prescription: Amlodipine 5mg", "Follow-up: 2 weeks"], "gt": "Patient: Jean Dupuis\nDOB: 15/03/1985\nBlood Pressure: 130/85\nHeart Rate: 72 bpm\nWeight: 78 kg\nPrescription: Amlodipine 5mg\nFollow-up: 2 weeks", "desc": "Handwritten medical notes."},
        {"file": "hw_note_9.png", "lines": ["Delivery Note #7842", "From: Warehouse Lyon", "To: Store Paris 12e", "Items: 24 boxes (fragile)", "Weight: 156 kg total", "Carrier: DHL Express", "Tracking: FR789456123"], "gt": "Delivery Note #7842\nFrom: Warehouse Lyon\nTo: Store Paris 12e\nItems: 24 boxes (fragile)\nWeight: 156 kg total\nCarrier: DHL Express\nTracking: FR789456123", "desc": "Handwritten delivery note."},
        {"file": "hw_note_10.png", "lines": ["Bank Deposit Slip", "Account: 4456 7890 1234", "Name: Claire Moreau", "Cash: 350.00 EUR", "Cheque: 1,200.00 EUR", "Total: 1,550.00 EUR", "Date: 09/07/2026", "Branch: Agence Centrale"], "gt": "Bank Deposit Slip\nAccount: 4456 7890 1234\nName: Claire Moreau\nCash: 350.00 EUR\nCheque: 1,200.00 EUR\nTotal: 1,550.00 EUR\nDate: 09/07/2026\nBranch: Agence Centrale", "desc": "Handwritten bank deposit slip."},
    ]
    for s in samples:
        path, gt = _draw_handwritten_text(s["file"], s["lines"], s["gt"])
        records.append({"image_path": path, "ground_truth": gt, "category": "handwritten", "description": s["desc"]})
        print(f"  Created {s['file']}")
    return records


def generate_handwritten_tables():
    """Generate 10 varied handwritten table images."""
    print("Generating 10 handwritten table images...")
    records = []
    tables = [
        {"file": "hw_table_attendance.png", "title": "Attendance - Week 28", "headers": ["Name", "Mon", "Tue", "Wed", "Thu", "Fri"], "rows": [["Alice B.", "P", "P", "A", "P", "P"], ["Bob C.", "P", "P", "P", "P", "A"], ["Clara D.", "A", "P", "P", "P", "P"], ["David E.", "P", "A", "P", "P", "P"]], "desc": "Handwritten weekly attendance sheet."},
        {"file": "hw_table_grades.png", "title": "Exam Results - Math 101", "headers": ["Student", "Midterm", "Final", "Grade"], "rows": [["Martin L.", "15/20", "17/20", "A"], ["Sophie R.", "12/20", "14/20", "B"], ["Hugo T.", "18/20", "16/20", "A"], ["Emma V.", "10/20", "11/20", "C"], ["Lucas M.", "14/20", "15/20", "B"]], "desc": "Handwritten student exam grades table."},
        {"file": "hw_table_budget.png", "title": "Monthly Budget - July 2026", "headers": ["Category", "Planned", "Actual"], "rows": [["Rent", "850 EUR", "850 EUR"], ["Groceries", "400 EUR", "435 EUR"], ["Transport", "75 EUR", "62 EUR"], ["Utilities", "120 EUR", "118 EUR"], ["Savings", "300 EUR", "250 EUR"]], "desc": "Handwritten personal monthly budget."},
        {"file": "hw_table_inventory.png", "title": "Inventory Count", "headers": ["Item", "Stock", "Min", "Reorder"], "rows": [["Paper A4", "24", "10", "No"], ["Ink Black", "3", "5", "Yes"], ["Pens Blue", "48", "20", "No"], ["Folders", "7", "15", "Yes"], ["Staples", "12", "5", "No"]], "desc": "Handwritten office inventory count."},
        {"file": "hw_table_schedule.png", "title": "Weekly Schedule", "headers": ["Time", "Monday", "Tuesday", "Wednesday"], "rows": [["09:00", "Math", "French", "History"], ["10:00", "Physics", "Math", "English"], ["11:00", "Break", "Break", "Break"], ["11:30", "English", "Art", "Math"], ["14:00", "Sport", "Music", "Science"]], "desc": "Handwritten school weekly schedule."},
        {"file": "hw_table_expenses.png", "title": "Travel Expenses - Paris", "headers": ["Date", "Description", "Amount"], "rows": [["07/01", "Train ticket", "89.00 EUR"], ["07/01", "Hotel 2 nights", "240.00 EUR"], ["07/02", "Restaurant", "45.50 EUR"], ["07/02", "Metro pass", "16.90 EUR"], ["07/03", "Taxi airport", "52.00 EUR"]], "desc": "Handwritten travel expense report."},
        {"file": "hw_table_contacts.png", "title": "Emergency Contacts", "headers": ["Name", "Relation", "Phone"], "rows": [["Marie D.", "Mother", "06 12 34 56 78"], ["Paul D.", "Father", "06 98 76 54 32"], ["Dr. Simon", "Doctor", "01 42 56 78 90"], ["Pompiers", "Fire", "18"]], "desc": "Handwritten emergency contacts list."},
        {"file": "hw_table_bank_txn.png", "title": "Bank Transactions - June", "headers": ["Date", "Label", "Debit", "Credit"], "rows": [["01/06", "Salary", "", "2850.00"], ["03/06", "Rent", "850.00", ""], ["05/06", "Groceries", "67.30", ""], ["12/06", "EDF", "95.00", ""], ["15/06", "Transfer", "", "200.00"], ["28/06", "Insurance", "145.00", ""]], "desc": "Handwritten bank transaction ledger."},
        {"file": "hw_table_workout.png", "title": "Workout Log", "headers": ["Exercise", "Sets", "Reps", "Weight"], "rows": [["Squat", "4", "10", "80 kg"], ["Bench Press", "3", "8", "60 kg"], ["Deadlift", "3", "6", "100 kg"], ["Pull-ups", "3", "12", "BW"], ["Plank", "3", "60s", "-"]], "desc": "Handwritten gym workout log."},
        {"file": "hw_table_recipe.png", "title": "Ingredients - Ratatouille", "headers": ["Ingredient", "Qty", "Unit"], "rows": [["Eggplant", "2", "pieces"], ["Zucchini", "3", "pieces"], ["Tomatoes", "4", "pieces"], ["Bell pepper", "2", "pieces"], ["Onion", "1", "large"], ["Garlic", "3", "cloves"], ["Olive oil", "4", "tbsp"]], "desc": "Handwritten recipe ingredients table."},
    ]
    for t in tables:
        gt = "| " + " | ".join(t["headers"]) + " |\n"
        gt += "| " + " | ".join(["---"] * len(t["headers"])) + " |\n"
        for row in t["rows"]:
            gt += "| " + " | ".join(row) + " |\n"
        gt = gt.rstrip("\n")
        path, _ = _draw_handwritten_table(t["file"], t["title"], t["headers"], t["rows"], gt)
        records.append({"image_path": path, "ground_truth": gt, "category": "handwritten", "description": t["desc"]})
        print(f"  Created {t['file']}")
    return records


def copy_kaggle_checks():
    checks_src_dir = os.path.join(DATASET_DIR, "kaggle_checks")
    records = []
    
    # Check 1
    chk1_src = os.path.join(checks_src_dir, "Bofa_handwritten_check.png")
    chk1_dest = os.path.join(HANDWRITTEN_DIR, "kaggle_check_1.png")
    if os.path.exists(chk1_src):
        with open(chk1_src, "rb") as f_in:
            with open(chk1_dest, "wb") as f_out:
                f_out.write(f_in.read())
                
        chk1_gt = (
            "BANK OF AMERICA\n"
            "MICHAEL J. DAVIS\n"
            "123 OAK ST\n"
            "ANYTOWN, FL 12345\n\n"
            "Check No: 1001\n"
            "Date: Jan 15, 2024\n"
            "Pay to the Order of: City Utilities\n"
            "Amount: $174.50\n"
            "Amount in words: One hundred seventy-four and 50/100\n"
            "Memo: Electricity Bill - Dec\n"
            "Signature: Michael J. Davis\n"
            "MICR: 123456789 1234567890 1001"
        )
        records.append({
            "image_path": chk1_dest,
            "ground_truth": chk1_gt,
            "category": "bank",
            "description": "Kaggle real handwritten Bank of America check."
        })
        
    # Check 2
    chk2_src = os.path.join(checks_src_dir, "citibank_handwritten_check.png")
    chk2_dest = os.path.join(HANDWRITTEN_DIR, "kaggle_check_2.png")
    if os.path.exists(chk2_src):
        with open(chk2_src, "rb") as f_in:
            with open(chk2_dest, "wb") as f_out:
                f_out.write(f_in.read())
                
        chk2_gt = (
            "CITIBANK, N.A.\n"
            "ROBERT K. SINGH\n\n"
            "Check No: 179\n"
            "Date: 11/29/21\n"
            "Pay to the Order of: Elijah Davis\n"
            "Amount: $4,850.00\n"
            "Amount in words: Four thousand eight hundred fifty and 00/100\n"
            "Memo: Freelance Services\n"
            "Signature: Olivia Moreno\n"
            "MICR: 021272655 9344408999 0179"
        )
        records.append({
            "image_path": chk2_dest,
            "ground_truth": chk2_gt,
            "category": "bank",
            "description": "Kaggle real handwritten Citibank check."
        })
        
    print(f"Copied and registered {len(records)} Kaggle checks.")
    return records

def load_clearocr_dataset():
    clearocr_dir = os.path.join(DATASET_DIR, "clearocr")
    test_dir = os.path.join(clearocr_dir, "test")
    metadata_path = os.path.join(clearocr_dir, "test_metadata.jsonl")
    
    records = []
    if not os.path.exists(metadata_path):
        print(f"Kaggle clearOCR dataset not found at {metadata_path}. Skipping integration.")
        return records
        
    print(f"Integrating Kaggle clearOCR dataset from {metadata_path}...")
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                
                img_rel = item["file_name"]
                img_path = os.path.join(test_dir, img_rel)
                
                md_rel = item["markdown_path"]
                md_path = os.path.join(test_dir, md_rel)
                
                if os.path.exists(img_path) and os.path.exists(md_path):
                    with open(md_path, "r", encoding="utf-8") as md_f:
                        gt_text = md_f.read().strip()
                        
                    records.append({
                        "image_path": img_path,
                        "ground_truth": gt_text,
                        "category": "bank",
                        "description": f"Kaggle clearOCR Real Invoice (Source: {item['source_id']})."
                    })
        print(f"Successfully integrated {len(records)} real invoices from Kaggle clearOCR dataset.")
    except Exception as e:
        print(f"Error parsing clearOCR dataset: {e}")
        
    return records


def load_kaggle_handwriting_ocr():
    """
    Load real handwritten word images from the Kaggle 'Handwriting Recognition OCR' dataset
    (ssarkar445/handwriting-recognitionocr). Reads labels from the CSV and matches with
    downloaded test images.
    """
    import csv
    
    kaggle_hw_dir = os.path.join(DATASET_DIR, "kaggle_handwriting")
    csv_path = os.path.join(kaggle_hw_dir, "written_name_test.csv")
    images_dir = os.path.join(kaggle_hw_dir, "images")
    
    records = []
    
    if not os.path.exists(csv_path):
        print(f"Kaggle handwriting OCR CSV not found at {csv_path}. Skipping.")
        return records
    
    if not os.path.exists(images_dir):
        print(f"Kaggle handwriting OCR images directory not found at {images_dir}. Skipping.")
        return records
    
    # Read CSV into a dict: filename -> label
    labels = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["FILENAME"]] = row["IDENTITY"]
    
    # Copy images into the handwritten dir and build records
    available_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".jpg")])
    count = 0
    for img_file in available_files:
        if count >= 20:
            break
        
        label = labels.get(img_file)
        if not label or label.strip() == "" or label == "UNREADABLE":
            continue
        
        src = os.path.join(images_dir, img_file)
        dest = os.path.join(HANDWRITTEN_DIR, f"kaggle_hw_{img_file}")
        
        # Copy to handwritten dir
        try:
            with open(src, "rb") as f_in:
                with open(dest, "wb") as f_out:
                    f_out.write(f_in.read())
        except Exception as e:
            print(f"  Error copying {img_file}: {e}")
            continue
        
        records.append({
            "image_path": dest,
            "ground_truth": label,
            "category": "handwritten",
            "description": f"Kaggle real handwritten word: '{label}' (source: ssarkar445/handwriting-recognitionocr)."
        })
        count += 1
        print(f"  Integrated {img_file} -> '{label}'")
    
    print(f"Integrated {len(records)} real handwritten images from Kaggle OCR dataset.")
    return records


def generate_synthetic_bank_receipt_2():
    path = os.path.join(LAYOUT_DIR, "bank_receipt_2.png")
    print(f"Generating synthetic bank receipt 2 at {path}...")
    
    img = Image.new("RGB", (600, 500), color="white")
    draw = ImageDraw.Draw(img)
    
    title_font = get_font(18)
    section_font = get_font(14)
    body_font = get_font(12)
    
    # Header
    draw.text((50, 40), "GLOBAL BANKING CORP", fill="#1e3d59", font=title_font)
    draw.text((50, 65), "OFFICIAL TRANSFER RECEIPT", fill="gray", font=body_font)
    draw.line([50, 85, 550, 85], fill="#1e3d59", width=2)
    
    # Metadata
    draw.text((50, 100), "Date: 2026-07-09", fill="black", font=body_font)
    draw.text((50, 120), "Status: COMPLETED", fill="green", font=body_font)
    draw.text((50, 140), "Transaction Reference: GBC-TXN-20260709-B2", fill="black", font=body_font)
    
    sender_iban = generate_valid_iban("FR", "30006", "10120", "01234567890", "89")
    beneficiary_iban = generate_valid_iban("FR", "20041", "01005", "11122233344", "07")
    
    # Sender details
    draw.text((50, 175), "SENDER DETAILS", fill="#1e3d59", font=section_font)
    draw.line([50, 195, 270, 195], fill="#1e3d59", width=1)
    draw.text((50, 205), "Account Holder: JOHN DOE", fill="black", font=body_font)
    draw.text((50, 225), f"Debit Account (IBAN):\n{sender_iban}", fill="black", font=body_font)
    
    # Beneficiary details
    draw.text((320, 175), "BENEFICIARY DETAILS", fill="#1e3d59", font=section_font)
    draw.line([320, 195, 550, 195], fill="#1e3d59", width=1)
    draw.text((320, 205), "Beneficiary Name: SARL TRANS-RAPIDE", fill="black", font=body_font)
    draw.text((320, 225), f"Credit Account (IBAN):\n{beneficiary_iban}", fill="black", font=body_font)
    
    # Draw boxes
    draw.rectangle([45, 170, 280, 270], outline="lightgray", width=1)
    draw.rectangle([315, 170, 555, 270], outline="lightgray", width=1)
    
    # Transfer details
    draw.text((50, 295), "TRANSFER DETAILS", fill="#1e3d59", font=section_font)
    draw.line([50, 315, 550, 315], fill="#1e3d59", width=1)
    
    draw.text((50, 325), "Amount: 450.00 EUR", fill="black", font=body_font)
    draw.text((50, 345), "Reason: Deliveries June 2026", fill="black", font=body_font)
    
    # Security Stamp box
    draw.rectangle([50, 380, 550, 440], outline="red", width=1, fill="#fff5f5")
    draw.text((65, 390), "SECURITY VALIDATION STAMP - CONFIRMED", fill="red", font=section_font)
    draw.text((65, 415), "Verified Beneficiary. No suspicious behavior patterns detected.", fill="red", font=body_font)
    
    # Footer
    draw.text((50, 460), "Document generated electronically. Page 1/1.", fill="gray", font=body_font)
    
    img.save(path)
    
    gt = (
        "GLOBAL BANKING CORP\n"
        "OFFICIAL TRANSFER RECEIPT\n\n"
        "Date: 2026-07-09\n"
        "Status: COMPLETED\n"
        "Transaction Reference: GBC-TXN-20260709-B2\n\n"
        "SENDER DETAILS\n"
        "Account Holder: JOHN DOE\n"
        f"Debit Account (IBAN): {sender_iban}\n\n"
        "BENEFICIARY DETAILS\n"
        "Beneficiary Name: SARL TRANS-RAPIDE\n"
        f"Credit Account (IBAN): {beneficiary_iban}\n\n"
        "TRANSFER DETAILS\n"
        "Amount: 450.00 EUR\n"
        "Reason: Deliveries June 2026\n\n"
        "SECURITY VALIDATION STAMP - CONFIRMED\n"
        "Verified Beneficiary. No suspicious behavior patterns detected.\n"
        "Document generated electronically. Page 1/1."
    )
    return path, gt, sender_iban, beneficiary_iban

def generate_synthetic_financial_table():
    path = os.path.join(TABLES_DIR, "synthetic_table_2.png")
    print(f"Generating synthetic table 2 at {path}...")
    
    img = Image.new("RGB", (650, 300), color="white")
    draw = ImageDraw.Draw(img)
    font = get_font(15)
    
    headers = ["Quarter", "Revenue", "Operating Expenses", "Net Profit"]
    rows = [
        ["Q1 2026", "$1,250,000", "$950,000", "$300,000"],
        ["Q2 2026", "$1,420,000", "$980,000", "$440,000"],
        ["Q3 2026", "$1,380,000", "$960,000", "$420,000"],
        ["Q4 2026", "$1,650,000", "$1,020,000", "$630,000"]
    ]
    
    col_widths = [120, 160, 200, 120]
    row_height = 40
    start_x = 25
    start_y = 50
    
    # Header Draw
    draw.rectangle([start_x, start_y, start_x + sum(col_widths), start_y + row_height], fill="#e0e0e0")
    
    # Draw headers
    current_x = start_x
    for i, h in enumerate(headers):
        draw.text((current_x + 10, start_y + 12), h, fill="black", font=font)
        current_x += col_widths[i]
        
    # Draw rows
    for r_idx, row in enumerate(rows):
        current_y = start_y + (r_idx + 1) * row_height
        current_x = start_x
        for c_idx, val in enumerate(row):
            draw.text((current_x + 10, current_y + 12), val, fill="black", font=font)
            current_x += col_widths[c_idx]
            
    # Grid lines
    total_width = sum(col_widths)
    total_height = (len(rows) + 1) * row_height
    
    for i in range(len(rows) + 2):
        y = start_y + i * row_height
        draw.line([start_x, y, start_x + total_width, y], fill="black", width=1)
        
    current_x = start_x
    for i in range(len(col_widths) + 1):
        draw.line([current_x, start_y, current_x, start_y + total_height], fill="black", width=1)
        if i < len(col_widths):
            current_x += col_widths[i]
            
    img.save(path)
    
    markdown_gt = (
        "| Quarter | Revenue | Operating Expenses | Net Profit |\n"
        "| --- | --- | --- | --- |\n"
        "| Q1 2026 | $1,250,000 | $950,000 | $300,000 |\n"
        "| Q2 2026 | $1,420,000 | $980,000 | $440,000 |\n"
        "| Q3 2026 | $1,380,000 | $960,000 | $420,000 |\n"
        "| Q4 2026 | $1,650,000 | $1,020,000 | $630,000 |"
    )
    return path, markdown_gt

def main():
    dataset = []
    
    # 1. Download real table image
    real_table_url = "https://raw.githubusercontent.com/eihli/image-table-ocr/master/resources/test_data/simple.png"
    real_table_path = os.path.join(TABLES_DIR, "real_table_1.png")
    if download_file(real_table_url, real_table_path):
        # Ground truth for simple.png from image-table-ocr
        real_table_gt = (
            "| Date | Description | Amount |\n"
            "| --- | --- | --- |\n"
            "| 2014-04-01 | Invoice #1001 | 120.00 |\n"
            "| 2014-04-02 | Invoice #1002 | 240.00 |\n"
            "| 2014-04-03 | Invoice #1003 | 360.00 |"
        )
        dataset.append({
            "image_path": real_table_path,
            "ground_truth": real_table_gt,
            "category": "tables",
            "description": "Real-world printed table containing dates, invoice descriptions, and amounts."
        })
        
    # 2. Download real handwriting image (line level)
    real_handwriting_url = "https://raw.githubusercontent.com/githubharald/SimpleHTR/master/data/line.png"
    real_handwriting_path = os.path.join(HANDWRITTEN_DIR, "real_handwriting_1.png")
    if download_file(real_handwriting_url, real_handwriting_path):
        dataset.append({
            "image_path": real_handwriting_path,
            "ground_truth": "or work on line level",
            "category": "handwritten",
            "description": "Real handwritten text line from the SimpleHTR / IAM database."
        })
        
    # 3. Download real historical handwriting (Gettysburg Address page 1)
    gettysburg_url = get_wikimedia_image_url("https://commons.wikimedia.org/wiki/File:Haycopy-1.jpg")
    if not gettysburg_url:
        gettysburg_url = "https://upload.wikimedia.org/wikipedia/commons/2/22/Haycopy-1.jpg"
    gettysburg_path = os.path.join(HANDWRITTEN_DIR, "real_handwriting_2.jpg")
    if download_file(gettysburg_url, gettysburg_path):
        gettysburg_gt = (
            "Four score and seven years ago our fathers brought forth, upon this continent, "
            "a new nation, conceived in Liberty, and dedicated to the proposition that "
            "all men are created equal.\n\n"
            "Now we are engaged in a great civil war, testing whether that nation, or any "
            "nation so conceived, and so dedicated, can long endure. We are met on a "
            "great battle-field of that war. We have come to dedicate a portion of it, "
            "as a final resting place for those who died here, that the nation might live. "
            "This we may, in all propriety do."
        )
        dataset.append({
            "image_path": gettysburg_path,
            "ground_truth": gettysburg_gt,
            "category": "handwritten",
            "description": "Real historical handwriting: Abraham Lincoln's Gettysburg Address (Hay Copy page 1)."
        })
        
    # 4. Generate synthetic table
    syn_table_path, syn_table_gt = generate_synthetic_table()
    dataset.append({
        "image_path": syn_table_path,
        "ground_truth": syn_table_gt,
        "category": "tables",
        "description": "Programmatically generated clean printed table."
    })
    
    # 5. Generate synthetic layout
    syn_layout_path, syn_layout_gt = generate_synthetic_layout()
    dataset.append({
        "image_path": syn_layout_path,
        "ground_truth": syn_layout_gt,
        "category": "complex_layout",
        "description": "Programmatically generated 2-column printed layout with title and callout box."
    })
    
    # 6. Generate synthetic banking receipt
    bank_path, bank_gt, sender_ib, beneficiary_ib = generate_synthetic_bank_receipt()
    dataset.append({
        "image_path": bank_path,
        "ground_truth": bank_gt,
        "category": "bank",
        "description": f"Synthetic official bank receipt with sender IBAN ({sender_ib}) and beneficiary IBAN ({beneficiary_ib})."
    })
    
    # 7. Generate synthetic bank receipt 2
    bank2_path, bank2_gt, sender2_ib, beneficiary2_ib = generate_synthetic_bank_receipt_2()
    dataset.append({
        "image_path": bank2_path,
        "ground_truth": bank2_gt,
        "category": "bank",
        "description": f"Second synthetic bank receipt with sender IBAN ({sender2_ib}) and beneficiary IBAN ({beneficiary2_ib})."
    })
    
    # 8. Generate synthetic financial table
    syn_table2_path, syn_table2_gt = generate_synthetic_financial_table()
    dataset.append({
        "image_path": syn_table2_path,
        "ground_truth": syn_table2_gt,
        "category": "tables",
        "description": "Programmatically generated clean printed corporate financial table."
    })
    
    # 9. Generate blurry bank receipt variation
    bank_blurry_path = apply_image_noise(bank_path, "blurry")
    dataset.append({
        "image_path": bank_blurry_path,
        "ground_truth": bank_gt,
        "category": "bank",
        "description": "Noisy variation: Blurry bank receipt to test OCR resilience to camera defocus."
    })
    
    # 10. Generate skewed bank receipt variation
    bank_skewed_path = apply_image_noise(bank_path, "skewed")
    dataset.append({
        "image_path": bank_skewed_path,
        "ground_truth": bank_gt,
        "category": "bank",
        "description": "Noisy variation: Rotated bank receipt (2 degrees skew) to test deskew capability."
    })
    
    # 11. Generate blurry table variation
    table_blurry_path = apply_image_noise(syn_table_path, "blurry")
    dataset.append({
        "image_path": table_blurry_path,
        "ground_truth": syn_table_gt,
        "category": "tables",
        "description": "Noisy variation: Blurry synthetic table."
    })
    
    # 12. Generate skewed table variation
    table_skewed_path = apply_image_noise(syn_table_path, "skewed_neg")
    dataset.append({
        "image_path": table_skewed_path,
        "ground_truth": syn_table_gt,
        "category": "tables",
        "description": "Noisy variation: Rotated synthetic table (-2 degrees skew)."
    })
    
    # 13. Generate handwritten bank receipt (using Caveat font)
    hw_bank_path, hw_bank_gt, hw_sender_ib, hw_beneficiary_ib = generate_handwritten_bank_receipt()
    dataset.append({
        "image_path": hw_bank_path,
        "ground_truth": hw_bank_gt,
        "category": "bank",
        "description": f"Synthetic handwritten bank receipt with sender IBAN ({hw_sender_ib}) and beneficiary IBAN ({hw_beneficiary_ib}) written in Caveat script."
    })
    
    # 14. Generate handwritten financial table (using Caveat font)
    hw_table_path, hw_table_gt = generate_handwritten_financial_table()
    dataset.append({
        "image_path": hw_table_path,
        "ground_truth": hw_table_gt,
        "category": "tables",
        "description": "Synthetic handwritten corporate financial table written in Caveat script."
    })
    
    # 15. Generate 10 handwritten text images
    hw_text_records = generate_handwritten_texts()
    dataset.extend(hw_text_records)
    
    # 16. Generate 10 handwritten table images
    hw_table_records = generate_handwritten_tables()
    dataset.extend(hw_table_records)
    
    # 17. Copy and register Kaggle real handwritten bank checks
    kaggle_check_records = copy_kaggle_checks()
    dataset.extend(kaggle_check_records)

    # Write dataset catalog JSON
    clearocr_records = load_clearocr_dataset()
    dataset.extend(clearocr_records)
    
    # Kaggle handwritten words (20 real images)
    kaggle_hw_ocr = load_kaggle_handwriting_ocr()
    dataset.extend(kaggle_hw_ocr)
    
    catalog_path = os.path.join(DATASET_DIR, "dataset.json")
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
        
    print(f"Dataset initialization complete! Catalog written to {catalog_path} with {len(dataset)} entries.")

if __name__ == "__main__":
    main()
