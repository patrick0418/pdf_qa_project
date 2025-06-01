
import os
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter

def load_pdf_text(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def split_text_to_chunks(text, chunk_size=500, chunk_overlap=50):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

if __name__ == "__main__":
    pdf_path = "./data/DELTA_IA_AM_OM_TC_20240415.pdf"  # 放入你準備的 PDF 檔
    if not os.path.exists(pdf_path):
        print("找不到 PDF：", pdf_path)
        exit()

    text = load_pdf_text(pdf_path)
    print("PDF 讀取完成，共字數：", len(text))

    chunks = split_text_to_chunks(text)
    print("分段數量：", len(chunks))
    print("第一段內容：\n", chunks[0])
