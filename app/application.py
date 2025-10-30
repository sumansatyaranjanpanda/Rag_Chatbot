from flask import Flask, render_template, request, session, redirect, url_for
from app.components.retriever import create_qa_chain
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
from markupsafe import Markup
import os
import uuid

load_dotenv()

db = SQLAlchemy()

app = Flask(__name__)
app.secret_key = os.urandom(24)
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chat_history.db'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql+psycopg2://chat_user:chat_pass123@localhost/chat_app1'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

# Global QA chain - initialized once
qa_chain = None

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    role = db.Column(db.String(20), nullable=False)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
    conversation_id = db.Column(db.String(36), nullable=False)

    def __repr__(self):
        return f'<Message {self.role}: {self.content[:50]}...>'

def create_tables():
    with app.app_context():
        db.create_all()
        print("PostgreSQL Database table created!")

def nl2br(value):
    """Convert newlines to HTML breaks"""
    return Markup(value.replace("\n", "<br>\n"))

app.jinja_env.filters['nl2br'] = nl2br

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

@app.route("/", methods=['GET', 'POST'])
def index():
    global qa_chain
    
    if "conversation_id" not in session:
        session["conversation_id"] = str(uuid.uuid4())

    conv_id = session['conversation_id']

    if request.method == "POST":
        user_input = request.form.get("prompt")

        if user_input:
            # Save user message
            user_msg = Message(role="user", content=user_input, conversation_id=conv_id)
            db.session.add(user_msg)
            db.session.commit()

            # Get all previous messages
            all_messages = Message.query.filter_by(
                conversation_id=conv_id
            ).order_by(Message.timestamp).all()

            # Exclude the current user message

            if len(all_messages)>6:
                prior_messages=all_messages[-6:-1]
            else:
                prior_messages = all_messages[:-1]




            # Build chat history (last 5 Q&A pairs) - FIXED FORMAT
            chat_history = []
            i = len(prior_messages) - 1
            pairs_collected = 0
            
            while i >= 1 and pairs_collected < 5:
                ai_msg = prior_messages[i]
                user_msg_prev = prior_messages[i - 1]
                
                if ai_msg.role == "assistant" and user_msg_prev.role == "user":
                    # Insert at beginning to maintain chronological order
                    # FORMAT: (user_question_string, ai_answer_string)
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

                # Get response from QA chain - FIXED: Use invoke instead of __call__
                response = qa_chain.invoke({
                    "question": user_input,
                    "chat_history": chat_history
                })
                result = response.get("answer", "No response generated")

                # Save assistant response
                ai_msg = Message(
                    role="assistant",
                    content=result,
                    conversation_id=conv_id
                )
                db.session.add(ai_msg)
                db.session.commit()

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                print(f"Exception in chat: {error_msg}")
                db.session.rollback()  # Undo user save on error
                
                # Save error message as assistant response
                ai_msg = Message(
                    role="assistant",
                    content=f"I apologize, but I encountered an error: {str(e)}",
                    conversation_id=conv_id
                )
                db.session.add(ai_msg)
                db.session.commit()

        return redirect(url_for("index"))

    # GET request - display messages
    messages = Message.query.filter_by(
        conversation_id=conv_id
    ).order_by(Message.timestamp).all()

    msg_lst = [{"role": msg.role, "content": msg.content} for msg in messages]

    return render_template("index.html", messages=msg_lst)


@app.route("/clear")
def clear():
    """Clear conversation history"""
    conv_id = session.get("conversation_id")
    if conv_id:
        Message.query.filter_by(conversation_id=conv_id).delete()
        db.session.commit()

    session.pop("conversation_id", None)
    return redirect(url_for('index'))


@app.route("/health")
def health():
    """Health check endpoint"""
    status = {
        "database": "ok",
        "qa_chain": "ok" if qa_chain is not None else "error"
    }
    return status, 200 if qa_chain is not None else 503


if __name__ == "__main__":
    # Create database tables
    create_tables()
    
    # Initialize QA chain before starting server
    initialize_qa_chain()
    
    # Start Flask app
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)