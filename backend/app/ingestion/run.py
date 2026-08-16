import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from pathlib import Path
import yaml
import json
from typing import List

from dotenv import load_dotenv
from fastapi.encoders import jsonable_encoder

from langchain_core.documents import Document
from langchain_community.vectorstores import PGVector
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import TokenTextSplitter

from unstructured.cleaners.core import clean_extra_whitespace

from app.core.config import logger
from app.schemas.ingestion_schema import LOADER_DICT
from app.utils.general_helpers import find_project_root
from utils.embedding_models import get_embedding_model


load_dotenv()


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

current_script_path = Path(__file__).resolve()
project_root = find_project_root(current_script_path)

ingestion_config_path = (
    project_root / "app" / "config" / "ingestion.yml"
)

with open(ingestion_config_path, "r") as file:
    ingestion_config = yaml.load(
        file,
        Loader=yaml.FullLoader,
    )


path_input_folder = project_root.parent / ingestion_config.get(
    "PATH_RAW_PDF",
    "data/raw",
)

path_extraction_folder = project_root.parent / ingestion_config.get(
    "PATH_EXTRACTION",
    "data/extraction",
)

collection_name = ingestion_config.get(
    "COLLECTION_NAME",
    "docs",
)

pdf_parser = ingestion_config.get(
    "PDF_PARSER",
    "Unstructured",
)

chunk_size = ingestion_config.get(
    "TOKENIZER_CHUNK_SIZE",
    2000,
)

chunk_overlap = ingestion_config.get(
    "TOKENIZER_CHUNK_OVERLAP",
    200,
)


# ---------------------------------------------------------
# Database configuration
# ---------------------------------------------------------

db_name = os.getenv("DB_NAME")
database_host = os.getenv("DB_HOST")
database_port = os.getenv("DB_PORT")
database_user = os.getenv("DB_USER")
database_password = os.getenv("DB_PASS")


# ---------------------------------------------------------
# PDF Extraction Pipeline
# ---------------------------------------------------------

class PDFExtractionPipeline:
    """Extract PDFs, split them into chunks, create embeddings,
    and store them in PostgreSQL/PGVector.
    """

    def __init__(self):
        logger.info("Initializing PDFExtractionPipeline")

        # Keep the project's loader configuration available.
        self.pdf_loader = LOADER_DICT.get(pdf_parser)

        # Local Hugging Face embedding model.
        self.embedding_model = get_embedding_model()

        # PostgreSQL connection string.
        self.connection_str = (
            PGVector.connection_string_from_db_params(
                driver="psycopg2",
                host=database_host,
                port=database_port,
                database=db_name,
                user=database_user,
                password=database_password,
            )
        )

        logger.debug(
            f"Connection string: {self.connection_str}"
        )

    def run(self, collection_name: str):
        logger.info(
            f"Running extraction pipeline for collection: "
            f"{collection_name}"
        )

        self._load_documents(
            folder_path=path_input_folder,
            collection_name=collection_name,
        )

    def _load_documents(
        self,
        folder_path: Path,
        collection_name: str,
    ):
        """Load, split, embed, and store documents."""

        text_documents = self._load_docs(folder_path)

        logger.info(
            f"Loaded {len(text_documents)} documents"
        )

        if not text_documents:
            logger.warning(
                f"No PDF documents found in: {folder_path}"
            )
            return None

        # Use the values from ingestion.yml.
        text_splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        texts = text_splitter.split_documents(
            text_documents
        )

        logger.info(
            f"Created {len(texts)} document chunks"
        )

        # Add metadata.
        for text in texts:
            text.metadata["type"] = "Text"

        # Make sure extraction directory exists.
        path_extraction_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Store embeddings in PGVector.
        logger.info(
            f"Creating PGVector collection: "
            f"{collection_name}"
        )

        vector_store = PGVector.from_documents(
            embedding=self.embedding_model,
            collection_name=collection_name,
            documents=texts,
            connection_string=self.connection_str,
            pre_delete_collection=True,
        )

        logger.info(
            "Documents successfully stored in PGVector"
        )

        return vector_store

    def _load_docs(
        self,
        dir_path: Path,
    ) -> List[Document]:
        """Load PDF files and extract their text."""

        documents = []

        if not dir_path.exists():
            logger.error(
                f"Input directory does not exist: {dir_path}"
            )
            return documents

        logger.info(
            f"Looking for PDFs in: {dir_path}"
        )

        for file_path in dir_path.iterdir():

            if file_path.suffix.lower() != ".pdf":
                continue

            logger.info(
                f"Loading PDF: {file_path.name}"
            )

            try:

                loader = UnstructuredFileLoader(
                    file_path=str(file_path),
                    strategy="hi_res",
                    post_processors=[
                        clean_extra_whitespace
                    ],
                )

                file_docs = loader.load()

                documents.extend(file_docs)

                # Save extracted text as JSON.
                json_path = (
                    path_extraction_folder
                    / f"{file_path.stem}.json"
                )

                path_extraction_folder.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                with open(
                    json_path,
                    "w",
                    encoding="utf-8",
                ) as json_file:

                    json.dump(
                        jsonable_encoder(file_docs),
                        json_file,
                        indent=4,
                        ensure_ascii=False,
                    )

                logger.info(
                    f"{file_path.name} loaded successfully"
                )

                logger.info(
                    f"Extracted text saved to: {json_path}"
                )

            except Exception as e:

                logger.error(
                    f"Could not extract text from "
                    f"{file_path.name}: {repr(e)}"
                )

        return documents


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    logger.info(
        "Starting PDF extraction pipeline"
    )

    pipeline = PDFExtractionPipeline()

    pipeline.run(collection_name)