import os, sys
from os.path import dirname as up

sys.path.append(os.path.abspath(os.path.join(up(__file__), os.pardir)))

from utils.common_libraries import *


def get_vectorstore_from_url(url):
    try:
        logger.info("Initializing WebBaseLoader with URL: %s", url)
        loader = WebBaseLoader(url)
        documents = loader.load()
        logger.info("Loaded documents from URL.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        logger.info("Split documents into texts.")

        embeddings = HuggingFaceEmbeddings(
            model_name="thenlper/gte-large",
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Initialized HuggingFace embeddings.")

        vector_store = chroma.Chroma.from_documents(texts, embeddings)
        logger.info("Vector store created successfully.")

        return vector_store
    except Exception as e:
        logger.error("Failed to get vector store from URL: %s", e)
        raise


def get_context_retriever_chain(vector_store):
    try:
        logger.info("Initializing ChatOpenAI for context retriever chain.")
        llm = ChatOpenAI()

        retriever = vector_store.as_retriever()
        logger.info("Vector store converted to retriever.")

        prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                (
                    "user",
                    "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
                ),
            ]
        )

        retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
        logger.info("Context retriever chain created successfully.")

        return retriever_chain
    except Exception as e:
        logger.error("Failed to get context retriever chain: %s", e)
        raise


def get_conversational_rag_chain(retriever_chain):
    try:
        logger.info("Initializing ChatOpenAI for conversational RAG chain.")
        llm = ChatOpenAI()

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Answer the user's questions based on the below context:\n\n{context}",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
            ]
        )

        stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
        logger.info("Stuff documents chain created.")

        rag_chain = create_retrieval_chain(retriever_chain, stuff_documents_chain)
        logger.info("Conversational RAG chain created successfully.")

        return rag_chain
    except Exception as e:
        logger.error("Failed to get conversational RAG chain: %s", e)
        raise


def get_response(user_query):
    try:
        logger.info("Fetching context retriever chain.")
        retriever_chain = get_context_retriever_chain(st.session_state.vector_store)

        logger.info("Fetching conversational RAG chain.")
        conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

        logger.info("Invoking conversational RAG chain with user query.")
        return conversation_rag_chain.stream(
            {"chat_history": st.session_state.chat_history, "input": user_query}
        )
    except Exception as e:
        logger.error("Failed to get response: %s", e)
        return "An error occurred while processing your request."
