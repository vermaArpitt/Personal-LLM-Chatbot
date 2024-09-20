from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from embedding_function import get_embedding_function
from langchain.vectorstores.chroma import Chroma

DATA_PATH = "data"

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 80,
        length_function = len,
        is_separator_regex = False,
    )
    return text_splitter.split_documents(documents)

def main():
    documents = load_documents()
    #print(documents[1].metadata)

    chunks = split_documents(documents)
    #print(chunks[0])

if __name__ == "__main__":
    main()