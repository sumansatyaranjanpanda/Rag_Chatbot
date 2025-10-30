from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate

from app.components.llm import load_llm
from app.components.vector_store import load_vector_store

from app.config.config import HUGGINGFACE_REPO_ID, HF_TOKEN
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

# Main QA Prompt
QA_PROMPT_TEMPLATE = """Answer the following medical question in 2-3 lines maximum using only the information provided in the context.

Context:
{context}

Question: {question}

Answer:
"""

# Enhanced Prompt to rephrase question using history (for better retrieval)
CONDENSE_QUESTION_PROMPT_TEMPLATE = """You are a helpful assistant that rephrases follow-up questions for better search results in a medical Q&A.

Chat History:
{chat_history}

Follow-up Question: {question}

First, identify the main topic from the chat history (e.g., the core subject of the previous question and answer, like 'cancer as a genetic disease').
Then, rephrase the follow-up into a standalone, self-contained question that explicitly incorporates that main topic for precise retrieval. Keep it concise, under 1 sentence.

Standalone Question:"""

def set_qa_prompt():
    return PromptTemplate(
        template=QA_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

def set_condense_prompt():
    return PromptTemplate(
        template=CONDENSE_QUESTION_PROMPT_TEMPLATE,
        input_variables=["chat_history", "question"]
    )

def create_qa_chain():
    try:
        logger.info("Loading vector store for context")
        db = load_vector_store()

        if db is None:
            raise CustomException("Vector store not present or empty")

        llm = load_llm()

        if llm is None:
            raise CustomException("LLM not loaded")

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=db.as_retriever(search_kwargs={'k': 3}),  # Increased for richer context
            condense_question_prompt=set_condense_prompt(),
            combine_docs_chain_kwargs={
                'prompt': set_qa_prompt(),
            },
            return_source_documents=False,
            verbose=True  # Keep for debugging; set False in prod
        )

        logger.info("Successfully created the conversational QA chain")
        return qa_chain

    except Exception as e:
        error_message = CustomException("Failed to make a QA chain", e)
        logger.error(str(error_message))
        return None

if __name__ == "__main__":
    qa_chain = create_qa_chain()
    
    if qa_chain is None:
        logger.error("Failed to create QA chain. Exiting.")
    else:
        # CORRECT FORMAT: (question_string, answer_string)
        mock_history = [
            ("what is cancer", "group of diseases where some of the body's cells divide uncontrollably, grow beyond their usual boundaries, and spread to other parts of the body")
        ]
        
        # Use invoke instead of __call__
        res = qa_chain.invoke({
            "question": "explain more on that", 
            "chat_history": mock_history
        })
        
        print("Response:", res["answer"])