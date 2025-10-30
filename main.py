from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, func
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from dotenv import load_dotenv
from starlette.middleware.sessions import SessionMiddleware
import os
from contextlib import asynccontextmanager
import uuid
from datetime import datetime
from typing import List, Dict
from app.components.retriever import create_qa_chain

load_dotenv()

# Database setup (moved up for engine use in lifespan)
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Global QA chain
qa_chain = None

# Database Model
class Message(Base):
    __tablename__ = 'message'
    
    id = Column(Integer, primary_key=True, index=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=func.current_timestamp())
    conversation_id = Column(String(36), nullable=False)

    def __repr__(self):
        return f'<Message {self.role}: {self.content[:50]}...>'

# Create tables
def create_tables():
    Base.metadata.create_all(bind=engine)
    print("PostgreSQL Database table created!")

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def initialize_qa_chain():
    """Initialize QA chain at startup"""
    global qa_chain
    try:
        print("Initializing QA chain...")
        qa_chain = create_qa_chain()
        print("QA chain initialized successfully!")
    except Exception as e:
        print(f"ERROR: Failed to initialize QA chain: {str(e)}")
        print("The application will start but chat functionality will not work.")
        qa_chain = None

# Lifespan event (defined BEFORE app init)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    create_tables()
    initialize_qa_chain()
    yield
    # Optional: Cleanup (e.g., engine.dispose())

# FastAPI app (now references defined lifespan)
app = FastAPI(lifespan=lifespan)

# Add session middleware (use env for prod consistency)
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SESSION_SECRET", os.urandom(24).hex()))

# Templates
from pathlib import Path  # Import here to avoid circular issues

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "app" / "templates"

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

# Add nl2br filter to Jinja2
def nl2br(value):
    """Convert newlines to HTML breaks"""
    return value.replace("\n", "<br>\n")

templates.env.filters['nl2br'] = nl2br

# Optional: Mount static files if you have them
# app.mount("/static", StaticFiles(directory=BASE_DIR / "app" / "static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def index_get(request: Request, db: Session = Depends(get_db)):
    """Display chat interface"""
    # Get or create conversation ID
    if "conversation_id" not in request.session:
        request.session["conversation_id"] = str(uuid.uuid4())
    
    conv_id = request.session["conversation_id"]
    
    # Get messages
    messages = db.query(Message).filter(
        Message.conversation_id == conv_id
    ).order_by(Message.timestamp).all()
    
    msg_lst = [{"role": msg.role, "content": msg.content} for msg in messages]
    
    return templates.TemplateResponse(
        "index1.html", 
        {"request": request, "messages": msg_lst}
    )

@app.post("/")
async def index_post(
    request: Request,
    prompt: str = Form(...),
    db: Session = Depends(get_db)
):
    """Handle chat message submission"""
    global qa_chain
    
    # Get or create conversation ID
    if "conversation_id" not in request.session:
        request.session["conversation_id"] = str(uuid.uuid4())
    
    conv_id = request.session["conversation_id"]
    
    if prompt:
        # Save user message
        user_msg = Message(
            role="user",
            content=prompt,
            conversation_id=conv_id
        )
        db.add(user_msg)
        db.commit()
        
        # Get all previous messages
        all_messages = db.query(Message).filter(
            Message.conversation_id == conv_id
        ).order_by(Message.timestamp).all()
        
        # Get prior messages (exclude current)
        if len(all_messages) > 6:
            prior_messages = all_messages[-6:-1]
        else:
            prior_messages = all_messages[:-1]
        
        # Build chat history (last 5 Q&A pairs)
        chat_history = []
        i = len(prior_messages) - 1
        pairs_collected = 0
        
        while i >= 1 and pairs_collected < 5:
            ai_msg = prior_messages[i]
            user_msg_prev = prior_messages[i - 1]
            
            if ai_msg.role == "assistant" and user_msg_prev.role == "user":
                chat_history.insert(0, (user_msg_prev.content, ai_msg.content))
                pairs_collected += 1
                i -= 2
            else:
                i -= 1
        
        print(f"DEBUG: Chat history built: {chat_history}")
        
        try:
            # Check if QA chain is initialized
            if qa_chain is None:
                raise Exception(
                    "QA system not initialized. Please check server logs for configuration errors."
                )
            
            # Get response from QA chain
            response = qa_chain.invoke({
                "question": prompt,
                "chat_history": chat_history
            })
            result = response.get("answer", "No response generated")
            
            # Save assistant response
            ai_msg = Message(
                role="assistant",
                content=result,
                conversation_id=conv_id
            )
            db.add(ai_msg)
            db.commit()
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"Exception in chat: {error_msg}")
            db.rollback()
            
            # Save error message as assistant response
            ai_msg = Message(
                role="assistant",
                content=f"I apologize, but I encountered an error: {str(e)}",
                conversation_id=conv_id
            )
            db.add(ai_msg)
            db.commit()
    
    return RedirectResponse(url="/", status_code=303)

@app.get("/clear")
async def clear(request: Request, db: Session = Depends(get_db)):
    """Clear conversation history"""
    conv_id = request.session.get("conversation_id")
    if conv_id:
        db.query(Message).filter(Message.conversation_id == conv_id).delete()
        db.commit()
    
    request.session.pop("conversation_id", None)
    return RedirectResponse(url="/", status_code=303)

@app.get("/health")
async def health():
    """Health check endpoint"""
    status = {
        "database": "ok",
        "qa_chain": "ok" if qa_chain is not None else "error"
    }
    status_code = 200 if qa_chain is not None else 503
    return status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))