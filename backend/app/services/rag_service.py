from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector

from app.core.config import settings


class RAGService:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL
        )

        self.vector_store = PGVector(
            collection_name="docs",
            connection_string=settings.SYNC_DATABASE_URI,
            embedding_function=self.embeddings,
        )

        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
        )

    def get_retriever_chain(self):
        retriever = self.vector_store.as_retriever()

        prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                (
                    "user",
                    "Given the conversation above, "
                    "generate a search query to retrieve "
                    "information relevant to the user's question.",
                ),
            ]
        )

        return create_history_aware_retriever(
            self.llm,
            retriever,
            prompt,
        )

    def get_rag_chain(self):
        retriever_chain = self.get_retriever_chain()

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Answer the user's question using only "
                    "the provided context.\n\n"
                    "Context:\n{context}",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
            ]
        )

        document_chain = create_stuff_documents_chain(
            self.llm,
            prompt,
        )

        return create_retrieval_chain(
            retriever_chain,
            document_chain,
        )

    def ask(
        self,
        message: str,
        chat_history: list[BaseMessage],
    ) -> str:
        chain = self.get_rag_chain()

        response = chain.invoke(
            {
                "input": message,
                "chat_history": chat_history,
            }
        )

        return response["answer"]