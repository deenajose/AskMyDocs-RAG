from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Read the PDF
pdf_path = "data/my_document.pdf"
reader = PdfReader(pdf_path)

# 2. Extract text from all pages
text = ""

for page in reader.pages:
    text += page.extract_text()

# 3. Create the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# 4. Split the document into chunks
chunks = text_splitter.split_text(text)

# 5. Display the results
print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks[:5]):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk)