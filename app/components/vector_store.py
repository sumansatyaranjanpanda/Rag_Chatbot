from pathlib import Path
import os
from langchain_community.vectorstores import FAISS

from app.components.embeddings import get_embedding_model
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from app.config.config import DB_FAISS_PATH

logger = get_logger(__name__)

DB_FAISS_PATH = Path(DB_FAISS_PATH)  # ensure pathlib
INDEX_FILENAME = "index.faiss"
INDEX_PATH = DB_FAISS_PATH / INDEX_FILENAME

def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def load_vector_store():
    """
    Returns a FAISS vectorstore instance if index exists and loads correctly,
    otherwise returns None (caller should rebuild).
    """
    try:
        embedding_model = get_embedding_model()

        logger.info(f"Checking vectorstore at: {DB_FAISS_PATH.resolve()}")
        # Check BOTH dir and actual faiss index file
        if DB_FAISS_PATH.exists() and INDEX_PATH.exists():
            logger.info("Loading existing vectorstore...")
            db = FAISS.load_local(
                str(DB_FAISS_PATH),
                embedding_model,
                allow_dangerous_deserialization=True
            )
            logger.info("Vectorstore loaded successfully.")
            return db
        else:
            logger.warning(
                f"No vectorstore index found. Expected index at: {INDEX_PATH.resolve()}"
            )
            # Helpful debug logs
            if DB_FAISS_PATH.exists():
                logger.info(f"Directory exists but index missing. Contents: {list(DB_FAISS_PATH.iterdir())}")
            else:
                logger.info("Vectorstore directory does not exist.")
            return None

    except Exception as e:
        error_message = CustomException("Failed to load vectorstore", e)
        logger.error(str(error_message))
        return None

# Creating new vectorstore function
def save_vector_store(text_chunks):
    """
    Build and save a FAISS vectorstore given text_chunks (list of Documents).
    Returns the saved db or raises CustomException on failure.
    """
    try:
        if not text_chunks:
            raise CustomException("No chunks were provided to save_vector_store()")

        logger.info("Generating your new vectorstore")
        embedding_model = get_embedding_model()

        db = FAISS.from_documents(text_chunks, embedding_model)

        # Ensure dir exists before saving
        _ensure_dir(DB_FAISS_PATH)

        logger.info(f"Saving vectorstore to {DB_FAISS_PATH.resolve()}")
        db.save_local(str(DB_FAISS_PATH))

        # Verify index file exists after save (langchain faiss should write index.faiss)
        if not INDEX_PATH.exists():
            # extra check: list files to help debug
            contents = list(DB_FAISS_PATH.iterdir())
            logger.error(f"Vectorstore saved but expected {INDEX_FILENAME} not found. Dir contents: {contents}")
            raise CustomException(f"FAISS index file not found after save: {INDEX_PATH}")

        logger.info("Vectorstore saved successfully.")
        return db

    except CustomException:
        # re-raise custom exceptions as-is
        raise
    except Exception as e:
        error_message = CustomException("Failed to create/save new vectorstore", e)
        logger.error(str(error_message))
        return None
