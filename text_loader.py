from langchain_community.document_loaders import TextLoader

loader = TextLoader("DocumentLoader/sample.txt", encoding="utf-8")
documents = loader.load()
print(documents)
print(len(documents))
print(documents[0].page_content)