import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
from langchain_voyageai import VoyageAIEmbeddings

load_dotenv()

print("Initializing components...")

embeddings=VoyageAIEmbeddings(model="voyage-3.5")
llm=ChatAnthropic(model_name="claude-haiku-4-5")

vectore_store=PineconeVectorStore(
    index_name=os.getenv("INDEX_NAME"),
    embedding=embeddings
)

vectore_store_retriever=vectore_store.as_retriever(search_kwargs={"k":3})

prompt_template=ChatPromptTemplate.from_template(
    """
    Answer the question only based on the following context:
    
    {context}
    
    Question: {question}
    
    Provide a detailed answer: 
    """
)

def retrieval_without_langchain_lcel(query: str):
    """
    Simple Retrieval without LCEL
    Manually retrieves the documents, formats them and generates the response

    Limitations:
    - Manual step by step execution
    - No Streaming support
    - No Async support
    - Harder to compose with other chains
    - More verbose and error-prone
    """

    # Step1: Retrieve documents
    documents=vectore_store_retriever.invoke(query)

    # Step2: Format retrieved documents into context string
    context=format_docs(documents)

    # Step3: Format the prompt_template with context and user query
    messages=prompt_template.format_messages(context=context, question=query)

    # Step4: Invoke LLM
    llm_response=llm.invoke(messages)

    # Step5: Print LLM result
    return llm_response.content

def retrieval_with_langchain_lcel():
    """
    Simple Retrieval with LCEL
    Returns a chain that can be invoked with user query.
    """
    retrieval_chain=(
        RunnablePassthrough.assign(
            context= itemgetter("question") | vectore_store_retriever| format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain


def format_docs(docs):
    """ Fromat revtried documents in to single string """
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":
    print("retrieving....")

    #Query
    #query="What is Pinecone in machine learning"
    query="Tell about Sai Krishna Reddy Chityala"

    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    print("=" * 70)
    raw_result=llm.invoke([HumanMessage(content=query)])
    print("\nAnswer:")
    print(raw_result.content)

    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Without LCEL (RAG)")
    print("=" * 70)
    result_without_lcel = retrieval_without_langchain_lcel(query)
    print("\nAnswer:")
    print(result_without_lcel)

    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: With LCEL (RAG)")
    print("=" * 70)
    chain_with_lcel = retrieval_with_langchain_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(result_without_lcel)