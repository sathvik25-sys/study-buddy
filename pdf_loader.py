import fitz  # PyMuPDF

def extract_text_from_pdf(filepath):
    doc = fitz.open(filepath)
    all_text = ""
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        all_text += f"\n--- Page {page_num + 1} ---\n{text}"
    
    return all_text

# Test it
if __name__ == "__main__":
    text = extract_text_from_pdf("sample.pdf")
    print(text[:500])  # print first 500 characters