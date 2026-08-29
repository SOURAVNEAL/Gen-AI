from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path="notebooks",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)
documents = loader.load()
print(len(documents))
print(documents[175].page_content)

