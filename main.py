import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.vectorstores import Chroma, FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class RAGConfig:
    documents_dir: str
    doc_type: str  # one of: txt, pdf, docx
    vector_backend: str  # one of: faiss, chroma
    persist_dir: str = "vector_store"
    faiss_index_path: str = "vector_store/faiss_index"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 4
    chat_model: str = "openai/gpt-oss-120b"
    temperature: float = 0.0
    max_history_turns: int = 6
    show_retrieved_docs: bool = False


class RAGPipeline:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model_name
        )
        self.vector_store = None
        self.retriever = None
        self.history_aware_retriever = None
        self.chat_history: List[HumanMessage | AIMessage] = []

        self.llm = ChatGroq(
            model_name=self.config.chat_model,
            temperature=self.config.temperature,
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant. Use retrieved context and conversation history to answer. "
                    "If the context does not contain the answer, say you do not know.",
                ),
                MessagesPlaceholder(variable_name="history"),
                (
                    "human",
                    "Retrieved context:\n{context}\n\nCurrent question: {question}",
                ),
            ]
        )
        self.retriever_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Rewrite the current user question into one retrieval query. "
                    "Give highest priority to the current question. "
                    "Use chat history only if needed to resolve references (for example pronouns, omitted entities, or time context). "
                    "Do not overuse history, do not add new facts, and do not expand scope unnecessarily. "
                    "If the current question is already standalone, return it unchanged. "
                    "Return only the rewritten query text.",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )

    def load_documents(self) -> List[Document]:
        doc_type = self.config.doc_type.lower().strip()
        pattern_map = {
            "txt": "*.txt",
            "pdf": "*.pdf",
            "docx": "*.docx",
        }
        if doc_type not in pattern_map:
            raise ValueError("doc_type must be one of: txt, pdf, docx")

        docs: List[Document] = []
        directory = Path(self.config.documents_dir)
        files = sorted(directory.glob(pattern_map[doc_type]))
        if not files:
            raise FileNotFoundError(
                f"No '{doc_type}' files found in directory: {self.config.documents_dir}"
            )

        for file_path in files:
            loader = self._loader_for_file(file_path)
            docs.extend(loader.load())

        return docs

    def _loader_for_file(self, file_path: Path):
        suffix = file_path.suffix.lower()
        if suffix == ".txt":
            return TextLoader(str(file_path), encoding="utf-8", autodetect_encoding=True)
        if suffix == ".pdf":
            return PyPDFLoader(str(file_path))
        if suffix == ".docx":
            return Docx2txtLoader(str(file_path))
        raise ValueError(f"Unsupported file type: {suffix}")

    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        return splitter.split_documents(docs)

    def build_or_load_vector_store(self, chunks: Optional[List[Document]] = None):
        backend = self.config.vector_backend.lower().strip()
        os.makedirs(self.config.persist_dir, exist_ok=True)

        if backend == "faiss":
            return self._build_or_load_faiss(chunks)
        if backend == "chroma":
            return self._build_or_load_chroma(chunks)
        raise ValueError("vector_backend must be one of: faiss, chroma")

    def _build_or_load_faiss(self, chunks: Optional[List[Document]]):
        index_dir = self.config.faiss_index_path
        if os.path.exists(index_dir):
            self.vector_store = FAISS.load_local(
                folder_path=index_dir,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            if not chunks:
                raise ValueError("Chunks are required to create a new FAISS index.")
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            os.makedirs(index_dir, exist_ok=True)
            self.vector_store.save_local(index_dir)

        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.config.top_k}
        )
        return self.vector_store

    def _build_or_load_chroma(self, chunks: Optional[List[Document]]):
        self.vector_store = Chroma(
            collection_name=f"rag_{self.config.doc_type}",
            embedding_function=self.embeddings,
            persist_directory=self.config.persist_dir,
        )

        existing_count = self.vector_store._collection.count()
        if existing_count == 0:
            if not chunks:
                raise ValueError("Chunks are required to create a new Chroma collection.")
            self.vector_store.add_documents(chunks)

        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.config.top_k}
        )
        return self.vector_store

    def ingest(self):
        docs = self.load_documents()
        chunks = self.chunk_documents(docs)
        self.build_or_load_vector_store(chunks=chunks)
        return len(docs), len(chunks)

    def create_context_aware_retriever(self):
        if not self.retriever:
            self.build_or_load_vector_store(chunks=None)

        if self.history_aware_retriever is None:
            self.history_aware_retriever = create_history_aware_retriever(
                llm=self.llm,
                retriever=self.retriever,
                prompt=self.retriever_prompt,
            )
        return self.history_aware_retriever

    def _retrieve_docs(self, query: str, use_context_aware_retriever: bool):
        if not use_context_aware_retriever:
            return self.retriever.invoke(query)

        history_aware_retriever = self.create_context_aware_retriever()
        return history_aware_retriever.invoke(
            {
                "input": query,
                "chat_history": self._get_history_window(),
            }
        )

    def answer_query(self, query: str, use_context_aware_retriever: bool = False) -> str:
        if not self.retriever:
            self.build_or_load_vector_store(chunks=None)

        retrieved_docs = self._retrieve_docs(
            query=query,
            use_context_aware_retriever=use_context_aware_retriever,
        )

        if self.config.show_retrieved_docs:
            for i, doc in enumerate(retrieved_docs, start=1):
                source = doc.metadata.get("source", "unknown")
                page = doc.metadata.get("page", "n/a")
                print(f"\n--- Retrieved doc {i} | source={source} | page={page} ---")
                print(doc.page_content)

        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        if not context.strip():
            context = "No relevant context retrieved."

        history = self._get_history_window()
        chain = self.prompt | self.llm
        response = chain.invoke(
            {
                "history": history,
                "context": context,
                "question": query,
            }
        )
        answer = str(response.content)

        self.chat_history.append(HumanMessage(content=query))
        self.chat_history.append(AIMessage(content=answer))
        return answer

    def _get_history_window(self):
        if self.config.max_history_turns <= 0:
            return []
        max_messages = self.config.max_history_turns * 2
        return self.chat_history[-max_messages:]

    def chat_cli(self, use_context_aware_retriever: bool = False):
        print("\nInteractive RAG chat started. Type 'exit' to quit.")
        while True:
            try:
                query = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting chat.")
                break

            if not query:
                continue
            if query.lower() in {"exit", "quit", "q"}:
                print("Exiting chat.")
                break

            answer = self.answer_query(
                query,
                use_context_aware_retriever=use_context_aware_retriever,
            )
            print(f"\nAssistant: {answer}")


def parse_args():
    parser = argparse.ArgumentParser(description="LangChain RAG pipeline")
    parser.add_argument("--documents_dir", required=True, help="Directory with files")
    parser.add_argument(
        "--doc_type",
        required=True,
        choices=["txt", "pdf", "docx"],
        help="Document type to load",
    )
    parser.add_argument(
        "--vector_backend",
        default="chroma",
        choices=["faiss", "chroma"],
        help="Vector store backend",
    )
    parser.add_argument(
        "--persist_dir",
        default="vector_store",
        help="Persistence directory (used by Chroma and FAISS path base)",
    )
    parser.add_argument(
        "--faiss_index_path",
        default="vector_store/faiss_index",
        help="Directory for FAISS index files",
    )
    parser.add_argument(
        "--query",
        help="Single question mode. If omitted, interactive chat mode is used.",
    )
    parser.add_argument(
        "--max_history_turns",
        type=int,
        default=6,
        help="Number of previous user-assistant turns to keep in memory.",
    )
    parser.add_argument(
        "--show_retrieved_docs",
        action="store_true",
        help="Print retrieved chunks each turn for debugging.",
    )
    parser.add_argument(
        "--use_context_aware_retriever",
        action="store_true",
        help="Use history-aware query rewriting for retrieval.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = RAGConfig(
        documents_dir=args.documents_dir,
        doc_type=args.doc_type,
        vector_backend=args.vector_backend,
        persist_dir=args.persist_dir,
        faiss_index_path=args.faiss_index_path,
        max_history_turns=args.max_history_turns,
        show_retrieved_docs=args.show_retrieved_docs,
    )

    pipeline = RAGPipeline(config)
    docs_count, chunks_count = pipeline.ingest()
    print(f"Ingested {docs_count} documents into {chunks_count} chunks.")

    if args.query:
        answer = pipeline.answer_query(
            args.query,
            use_context_aware_retriever=args.use_context_aware_retriever,
        )
        print("\nAnswer:\n")
        print(answer)
    else:
        pipeline.chat_cli(
            use_context_aware_retriever=args.use_context_aware_retriever
        )


if __name__ == "__main__":
    main()
