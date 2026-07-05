import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain_voyageai import VoyageAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone


load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")
    print(os.getenv("INDEX_NAME"))

    loader = TextLoader ("mediumblog1.txt", encoding="utf-8")
    document = loader.load()

    print("splitting....")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)

    print("splitting complete...")
    print(f"{len(texts)} documents")

    # embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    embeddings = VoyageAIEmbeddings(api_key=os.getenv("VOYAGEAI_API_KEY"), model="voyage-3")

    print("ingesting...")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("INDEX_NAME"))
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.getenv("INDEX_NAME")
    )
