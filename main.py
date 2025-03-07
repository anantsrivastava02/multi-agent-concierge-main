import asyncio
import psycopg2
from PIL import Image
import pinecone
import redis
from datetime import datetime
from transformers import BartForConditionalGeneration, BartTokenizer
from anthropic import Anthropic
from typing import Optional, Tuple
import os
import json
from PIL import ExifTags
import logging
from colorama import Fore, Style
from dotenv import load_dotenv

from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import BaseTool
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI

from workflow import (
    AgentConfig,
    ConciergeAgent,
    ProgressEvent,
    ToolRequestEvent,
    ToolApprovedEvent,
    LLMManager,
    ErrorEvent,
    ToolCallEvent,
    ActiveSpeakerEvent,
    OrchestratorEvent,
)
from utils import FunctionToolWithContext

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Replace the synchronous Redis client with async Redis
from redis.asyncio import Redis as AsyncRedis

# Initialize Redis client (async)
redis_client = AsyncRedis(
    host=os.getenv("REDIS_HOST", "127.0.0.1"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    db=int(os.getenv("REDIS_DB", "0")),
    decode_responses=True
)

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

def summarize_conversation(conversation: str) -> str:
    """Summarize conversation using BART."""
    inputs = tokenizer([conversation], max_length=1024, truncation=True, return_tensors='pt')
    summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=40, length_penalty=2.0)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def get_initial_state() -> dict:
    return {
        "username": None,
        "session_token": None,
    }

def get_retriever_tools() -> list[BaseTool]:
    def retrieve_information(ctx: Context, query: str) -> str:
        """Retrieve information from a Pinecone vector database."""
        ctx.write_event_to_stream(
            ProgressEvent(msg=f"Retrieving information for query: {query}")
        )
        try:
            # Initialize Pinecone
            pinecone.init(
                api_key=os.getenv("PINECONE_API_KEY"),
                environment="embedding"
            )
            
            # Create embedding for query
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.embeddings.create(
                input=query,
                model="text-embedding-3-small"
            )
            query_embedding = response.data[0].embedding
            
            # Query the index
            index = pinecone.Index("embedding")
            results = index.query(
                vector=query_embedding,
                top_k=3,
                include_metadata=True
            )
            
            # Format results
            if not results.matches:
                return "I couldn't find any relevant information for your query."
                
            response = "Here's what I found:\n\n"
            for i, match in enumerate(results.matches, 1):
                content = match.metadata.get('content', 'No content available')
                score = match.score
                response += f"{i}. {content} (Relevance: {score:.2f})\n\n"
            
            return response
            
        except Exception as e:
            logger.error(f"Error in retrieve_information: {e}")
            return f"I encountered an error while retrieving information: {str(e)}"

    return [
        FunctionToolWithContext.from_defaults(fn=retrieve_information),
    ]


def get_authentication_tools() -> list[BaseTool]:
    async def is_authenticated(ctx: Context) -> bool:
        """Checks if the user has a session token."""
        ctx.write_event_to_stream(ProgressEvent(msg="Checking if authenticated"))
        user_state = await ctx.get("user_state")
        return user_state.get("session_token") is not None

    async def store_username(ctx: Context, username: str) -> str:
        """Adds the username to the user state."""
        ctx.write_event_to_stream(ProgressEvent(msg=f"Recording username: {username}"))
        user_state = await ctx.get("user_state", {})
        user_state["username"] = username
        await ctx.set("user_state", user_state)
        
        # Also store in conversation state
        conversation_state = await ctx.get("conversation_state", {})
        conversation_state["username"] = username
        await ctx.set("conversation_state", conversation_state)
        
        return f"Username '{username}' stored. Please provide your password to login."

    async def login(ctx: Context, password: str = None) -> str:
        """Given a password, logs in and stores a session token in the user state."""
        user_state = await ctx.get("user_state", {})
        username = user_state.get("username")
        
        # If password is not provided, try to get it from conversation state
        if not password:
            conversation_state = await ctx.get("conversation_state", {})
            password = conversation_state.get("password")
        
        if not username:
            return "Please provide your username first before attempting to login."
            
        ctx.write_event_to_stream(ProgressEvent(msg=f"Attempting login for user {username}"))
        
        # Check credentials in database
        conn = psycopg2.connect(os.getenv("POSTGRES_CONNECTION_STRING"))
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT EXISTS(SELECT 1 FROM users WHERE username = %s)",
                (username,)
            )
            user_exists = cur.fetchone()[0]
            
            if not user_exists:
                # For demo purposes, create a new user
                session_token = f"token_{username}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                user_state["session_token"] = session_token
                await ctx.set("user_state", user_state)
                
                # Store new user in database
                cur.execute(
                    """
                    INSERT INTO users (username, session_token, created_at)
                    VALUES (%s, %s, %s)
                    """,
                    (username, session_token, datetime.now())
                )
                conn.commit()
                
                return f"Welcome {username}! You've been registered and logged in successfully."
            else:
                # User exists, generate new session token
                session_token = f"token_{username}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                user_state["session_token"] = session_token
                await ctx.set("user_state", user_state)
                
                # Update session token
                cur.execute(
                    """
                    UPDATE users 
                    SET session_token = %s
                    WHERE username = %s
                    """,
                    (session_token, username)
                )
                conn.commit()
                
                return f"Welcome back, {username}! You've been logged in successfully."
                
        except Exception as e:
            logger.error(f"Database error during login: {e}")
            return f"Login failed due to a system error: {str(e)}"
        finally:
            cur.close()
            conn.close()

    async def verify_session(ctx: Context, username: str, session_token: str) -> tuple[bool, Optional[str]]:
        """Verify if session token is valid and load user context."""
        ctx.write_event_to_stream(ProgressEvent(msg="Verifying session token"))
        
        # Check Redis cache first
        cache_key = f"session:{username}:{session_token}"
        redis_client = AsyncRedis(
            host=os.getenv("REDIS_HOST", "127.0.0.1"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
            decode_responses=True
        )
        
        try:
            cached_data = await redis_client.get(cache_key)
            if cached_data:
                return True, cached_data
        except Exception as redis_error:
            logger.warning(f"Redis error: {redis_error}")
        finally:
            await redis_client.aclose()

        # If not in cache, check database
        conn = psycopg2.connect(os.getenv("POSTGRES_CONNECTION_STRING"))
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT EXISTS(SELECT 1 FROM users WHERE username = %s AND session_token = %s)",
                (username, session_token)
            )
            exists = cur.fetchone()[0]
            
            if exists:
                # Get conversation summary
                cur.execute(
                    "SELECT conversation_text FROM conversation_summaries WHERE username = %s ORDER BY timestamp DESC LIMIT 1",
                    (username,)
                )
                summary_row = cur.fetchone()
                summary = summary_row[0] if summary_row else None
                
                # Cache the result
                if summary:
                    try:
                        redis_client = AsyncRedis(
                            host=os.getenv("REDIS_HOST", "127.0.0.1"),
                            port=int(os.getenv("REDIS_PORT", "6379")),
                            db=int(os.getenv("REDIS_DB", "0")),
                            decode_responses=True
                        )
                        await redis_client.setex(cache_key, 3600, summary)  # Cache for 1 hour
                        await redis_client.aclose()
                    except Exception as redis_error:
                        logger.warning(f"Redis caching error: {redis_error}")
                
                return True, summary
            return False, None
        finally:
            cur.close()
            conn.close()

    return [
        FunctionToolWithContext.from_defaults(async_fn=is_authenticated),
        FunctionToolWithContext.from_defaults(async_fn=store_username),
        FunctionToolWithContext.from_defaults(async_fn=login),
        FunctionToolWithContext.from_defaults(async_fn=verify_session),
    ]





def get_image_processing_tools() -> list[BaseTool]:
    async def analyze_image(ctx: Context, image_path: str, question: str = None) -> str:
        """
        Analyzes an image and answers questions about it using GPT-4 Vision.
        If no question is provided, returns a general description.
        """
        ctx.write_event_to_stream(ProgressEvent(msg=f"Processing image: {image_path}"))
        
        try:
            # Check if image exists and is valid
            if not os.path.exists(image_path):
                return f"Error: Image file not found at {image_path}"
            
            # Read and encode image
            with open(image_path, "rb") as image_file:
                import base64
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Initialize OpenAI client
            client = OpenAI()
            
            # Prepare the message content
            content = [
                {
                    "type": "text",
                    "text": question if question else "What's in this image? Provide a detailed description."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    }
                }
            ]

            # Get response from GPT-4 Vision
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                max_tokens=500
            )
            
            # Cache the response
            cache_key = f"image_analysis:{image_path}:{question}"
            await ctx.get("cache_manager").set_cached_response(
                cache_key, 
                response.choices[0].message.content
            )
            
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return f"Error processing image: {str(e)}"

    async def save_uploaded_image(ctx: Context, image_data: str, filename: str) -> str:
        """
        Saves an uploaded image (base64 encoded) to the server and returns the path.
        """
        ctx.write_event_to_stream(ProgressEvent(msg=f"Saving uploaded image: {filename}"))
        
        try:
            # Create uploads directory if it doesn't exist
            upload_dir = "uploads"
            os.makedirs(upload_dir, exist_ok=True)
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = "".join(c for c in filename if c.isalnum() or c in "._-")
            unique_filename = f"{timestamp}_{safe_filename}"
            file_path = os.path.join(upload_dir, unique_filename)
            
            # Decode and save image
            import base64
            image_data_decoded = base64.b64decode(image_data)
            with open(file_path, "wb") as f:
                f.write(image_data_decoded)
            
            return file_path

        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return f"Error saving image: {str(e)}"

    async def get_image_metadata(ctx: Context, image_path: str) -> str:
        """
        Returns metadata about the image (size, format, etc.).
        """
        ctx.write_event_to_stream(ProgressEvent(msg=f"Getting metadata for: {image_path}"))
        
        try:
            image = Image.open(image_path)
            metadata = {
                "format": image.format,
                "mode": image.mode,
                "size": image.size,
                "filename": os.path.basename(image_path),
                "created": datetime.fromtimestamp(os.path.getctime(image_path)).isoformat(),
                "modified": datetime.fromtimestamp(os.path.getmtime(image_path)).isoformat(),
            }
            
            # Try to get EXIF data if available
            try:
                exif = image._getexif()
                if exif:
                    metadata["exif"] = {
                        ExifTags.TAGS[k]: str(v)
                        for k, v in exif.items()
                        if k in ExifTags.TAGS
                    }
            except:
                pass
                
            return json.dumps(metadata, indent=2)

        except Exception as e:
            logger.error(f"Error getting image metadata: {e}")
            return f"Error getting image metadata: {str(e)}"

    return [
        FunctionToolWithContext.from_defaults(async_fn=analyze_image),
        FunctionToolWithContext.from_defaults(async_fn=save_uploaded_image),
        FunctionToolWithContext.from_defaults(async_fn=get_image_metadata),
    ]


def get_conversation_storage_tools() -> list[BaseTool]:
    async def store_conversation(ctx: Context, user_message: str, agent_response: str) -> None:
        """Stores conversation history and summary in PostgreSQL database."""
        ctx.write_event_to_stream(ProgressEvent(msg="Storing conversation in database"))
        
        user_state = await ctx.get("user_state")
        username = user_state.get("username", "anonymous")
        
        conn = psycopg2.connect("postgresql://postgres:Papa&Maa1@localhost:5432/mytestdb")
        try:
            cur = conn.cursor()
            
            # Store conversation
            cur.execute(
                """
                INSERT INTO conversations 
                (username, user_message, agent_response, timestamp)
                VALUES (%s, %s, %s, %s)
                """,
                (username, user_message, agent_response, datetime.now())
            )

            # Get recent conversations for summarization
            cur.execute(
                """
                SELECT user_message, agent_response 
                FROM conversations 
                WHERE username = %s 
                ORDER BY timestamp DESC 
                LIMIT 10
                """,
                (username,)
            )
            conversations = cur.fetchall()
            
            # Create conversation text for summarization
            conversation_text = "\n".join([
                f"User: {msg}\nAgent: {resp}" 
                for msg, resp in conversations
            ])
            
            # Generate summary
            summary = summarize_conversation(conversation_text)
            
            # Store summary
            cur.execute(
                """
                INSERT INTO conversation_summaries 
                (username, conversation_text, summary, timestamp)
                VALUES (%s, %s, %s, %s)
                """,
                (username, conversation_text, summary, datetime.now())
            )
            
            conn.commit()
        finally:
            cur.close()
            conn.close()

    return [
        FunctionToolWithContext.from_defaults(async_fn=store_conversation),
    ]


def get_agent_configs() -> dict:
    """Get agent configurations."""
    return {
        "general": AgentConfig(
            name="general",
            description="General purpose assistant for basic conversation and queries",
            system_prompt="You are a helpful assistant. Answer user questions clearly and concisely.",
            tools=[],  # No special tools needed for general conversation
        ),
        "retriever": AgentConfig(
            name="retriever",
            description="Information retrieval specialist for searching and retrieving information",
            system_prompt="You are an information retrieval specialist. Help users find information using the available tools.",
            tools=get_retriever_tools(),
        ),
        "auth": AgentConfig(
            name="auth",
            description="Authentication specialist for handling user login and registration",
            system_prompt="You are an authentication specialist. Help users with login and registration.",
            tools=get_authentication_tools(),
            tools_requiring_human_confirmation=["login", "store_user_info_in_db"],
        ),
        "image": AgentConfig(
            name="image",
            description="Image processing specialist for handling image-related tasks",
            system_prompt="You are an image processing specialist. Help users with image-related tasks.",
            tools=get_image_processing_tools(),
            tools_requiring_human_confirmation=["analyze_image"],
        ),
    }


def check_environment():
    """Check if all required environment variables are set."""
    required_vars = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
        "POSTGRES_CONNECTION_STRING": os.getenv("POSTGRES_CONNECTION_STRING"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
    }
    
    missing = [k for k, v in required_vars.items() if not v]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")
    
    logger.info("All required environment variables are set")


async def test_connections():
    """Test database and Redis connections."""
    # Test PostgreSQL connection
    try:
        conn = psycopg2.connect(os.getenv("POSTGRES_CONNECTION_STRING"))
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        conn.close()
        logger.info("PostgreSQL connection successful")
    except Exception as e:
        logger.error(f"PostgreSQL connection failed: {e}")
        raise

    # Test Redis connection
    try:
        is_connected = await redis_client.ping()
        if is_connected:
            logger.info("Redis connection successful")
        else:
            raise ConnectionError("Redis ping returned False")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        raise

def ensure_database_tables():
    """Ensure all required database tables exist."""
    conn = psycopg2.connect(os.getenv("POSTGRES_CONNECTION_STRING"))
    try:
        cur = conn.cursor()
        
        # Create users table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(255) NOT NULL,
                session_token VARCHAR(255) NOT NULL,
                created_at TIMESTAMP NOT NULL
            )
        """)
        
        # Create conversations table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id SERIAL PRIMARY KEY,
                username VARCHAR(255) NOT NULL,
                user_message TEXT NOT NULL,
                agent_response TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL
            )
        """)
        
        # Create conversation_summaries table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversation_summaries (
                id SERIAL PRIMARY KEY,
                username VARCHAR(255) NOT NULL,
                conversation_text TEXT NOT NULL,
                summary TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                embedded_at TIMESTAMP
            )
        """)
        
        conn.commit()
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()

# Add this at the beginning of main.py, after imports
class ConversationState:
    """Tracks the state of the conversation."""
    def __init__(self):
        self.current_flow = None  # Can be "login", "retrieval", etc.
        self.flow_step = 0  # Step within the current flow
        self.collected_data = {}  # Data collected during the flow

async def check_authentication(ctx, user_msg, initial_state):
    """Check if user is authenticated and handle login if needed."""
    if initial_state.get("session_token"):
        # User is already authenticated
        return True, None
    
    # Check if this is a login attempt
    login_keywords = ["login", "sign in", "authenticate", "username", "password"]
    if any(keyword in user_msg.lower() for keyword in login_keywords):
        # This is likely a login attempt, let the auth agent handle it
        return False, "auth"
    
    # User is not authenticated and not trying to login
    return False, "Please login first. You can say something like 'I want to login' or provide your username and password."

async def main():
    """Main function to run the workflow."""
    workflow = None
    user_msg = "Hello!"  # Initialize user_msg
    
    # Simple authentication state
    auth_state = {
        "is_authenticated": False,
        "username": None,
        "session_token": None,
        "waiting_for_username": False,
        "waiting_for_password": False
    }
    
    try:
        check_environment()
        ensure_database_tables()
        await test_connections()
        logger.info("Starting main workflow...")
        
        # Initialize LLM with fallback capability
        logger.info("Initializing LLM Manager...")
        llm = LLMManager()
        logger.info("LLM Manager initialized successfully")
        
        logger.info("Initializing chat memory...")
        memory = ChatMemoryBuffer.from_defaults(llm=llm.openai_llm)
        logger.info("Chat memory initialized successfully")
        
        logger.info("Getting initial state and agent configs...")
        initial_state = get_initial_state()
        agent_configs = get_agent_configs()
        logger.info(f"Loaded {len(agent_configs)} agent configs")
        
        # Initialize workflow with timeout
        logger.info("Initializing ConciergeAgent...")
        workflow = ConciergeAgent(timeout=None)
        logger.info("ConciergeAgent initialized successfully")
        
        while True:
            try:
                # Handle authentication flow directly
                if (not auth_state["is_authenticated"] and 
                    (auth_state["waiting_for_username"] or auth_state["waiting_for_password"] or 
                     "login" in user_msg.lower() or "sign" in user_msg.lower())):
                    
                    # Start login process if not already started
                    if not auth_state["waiting_for_username"] and not auth_state["waiting_for_password"]:
                        print(Fore.BLUE + "AGENT >> Please provide your username:" + Style.RESET_ALL)
                        auth_state["waiting_for_username"] = True
                        user_msg = input("USER >> ")
                        continue
                    
                    if auth_state["waiting_for_username"]:
                        # Extract username (remove any extra text)
                        username = user_msg.split(',')[0].strip()
                        auth_state["username"] = username
                        auth_state["waiting_for_username"] = False
                        auth_state["waiting_for_password"] = True
                        print(Fore.BLUE + f"AGENT >> Username '{username}' stored. Please provide your password:" + Style.RESET_ALL)
                        user_msg = input("USER >> ")
                        continue
                    
                    if auth_state["waiting_for_password"]:
                        # Extract password (remove any extra text)
                        password = user_msg.split(',')[0].strip()
                        
                        # Generate a session token
                        session_token = f"token_{auth_state['username']}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                        
                        try:
                            # Store in database
                            conn = psycopg2.connect(os.getenv("POSTGRES_CONNECTION_STRING"))
                            try:
                                cur = conn.cursor()
                                cur.execute(
                                    "SELECT EXISTS(SELECT 1 FROM users WHERE username = %s)",
                                    (auth_state["username"],)
                                )
                                user_exists = cur.fetchone()[0]
                                
                                if not user_exists:
                                    # Create new user
                                    cur.execute(
                                        """
                                        INSERT INTO users (username, session_token, created_at)
                                        VALUES (%s, %s, %s)
                                        """,
                                        (auth_state["username"], session_token, datetime.now())
                                    )
                                else:
                                    # Update existing user
                                    cur.execute(
                                        """
                                        UPDATE users 
                                        SET session_token = %s
                                        WHERE username = %s
                                        """,
                                        (session_token, auth_state["username"])
                                    )
                                conn.commit()
                            except Exception as db_error:
                                logger.error(f"Database error: {db_error}")
                            finally:
                                cur.close()
                                conn.close()
                        except Exception as conn_error:
                            logger.error(f"Connection error: {conn_error}")
                            # Continue even if database fails - for testing
                        
                        # Update auth state
                        auth_state["is_authenticated"] = True
                        auth_state["session_token"] = session_token
                        auth_state["waiting_for_password"] = False
                        
                        # Update initial state
                        initial_state["username"] = auth_state["username"]
                        initial_state["session_token"] = session_token
                        
                        print(Fore.BLUE + f"AGENT >> Welcome, {auth_state['username']}! You've been logged in successfully." + Style.RESET_ALL)
                        user_msg = input("USER >> ")
                        continue
                
                # Check if user is authenticated
                if not auth_state["is_authenticated"]:
                    print(Fore.BLUE + "AGENT >> Please login first. Type 'login' to begin." + Style.RESET_ALL)
                    user_msg = input("USER >> ")
                    continue
                
                # User is authenticated, proceed with normal workflow
                logger.info("Setting up initial handler...")
                handler = workflow.run()
                await handler.ctx.set("user_msg", user_msg)
                await handler.ctx.set("llm", llm.openai_llm)
                await handler.ctx.set("chat_history", memory.get())
                
                # Set authenticated user state
                initial_state["username"] = auth_state["username"]  # Ensure username is set
                initial_state["session_token"] = auth_state["session_token"]  # Ensure token is set
                await handler.ctx.set("initial_state", initial_state)
                
                # Set up agent configs - make sure to include all your agents
                await handler.ctx.set("agent_configs", {
                    "retriever": AgentConfig(
                        name="retriever",
                        description="Information retrieval agent that can search for information",
                        system_prompt="You are an information retrieval agent. Help users find information by searching through available data.",
                        tools=get_retriever_tools()
                    ),
                    # Add other agents as needed
                })
                
                logger.info("Initial handler setup complete")

                # Process events
                async for event in handler.stream_events():
                    try:
                        logger.info(f"Received event: {type(event)}")
                        if isinstance(event, ProgressEvent):
                            print(Fore.GREEN + f"SYSTEM >> {event.msg}" + Style.RESET_ALL)
                        elif isinstance(event, ErrorEvent):
                            print(Fore.RED + f"ERROR >> {event.error}" + Style.RESET_ALL)
                    except Exception as event_error:
                        logger.error(f"Error handling event: {event_error}")
                        continue

                # Get result
                result = await handler
                if result.get("error"):
                    print(Fore.RED + f"ERROR >> {result['response']}" + Style.RESET_ALL)
                    user_msg = input("USER >> ")
                    if user_msg.strip().lower() in ["exit", "quit", "bye"]:
                        break
                    continue

                print(Fore.BLUE + f"AGENT >> {result['response']}" + Style.RESET_ALL)

                # Cache the conversation
                await workflow.cache_conversation(
                    user_msg=user_msg,
                    response=result['response'],
                    username=initial_state.get('username', 'anonymous')
                )

                # Update memory
                for msg in result["chat_history"]:
                    if msg not in memory.get():
                        memory.put(msg)

                user_msg = input("USER >> ")
                if user_msg.strip().lower() in ["exit", "quit", "bye"]:
                    break

            except Exception as loop_error:
                logger.error(f"Error in conversation loop: {loop_error}")
                print(Fore.RED + f"System error occurred. Please try again." + Style.RESET_ALL)
                user_msg = input("USER >> ")
                if user_msg.strip().lower() in ["exit", "quit", "bye"]:
                    break
                continue

    except Exception as e:
        logger.error(f"Error in main workflow: {e}", exc_info=True)
        raise

    finally:
        try:
            if workflow is not None:
                await workflow.cleanup()
                logger.info("Workflow cleaned up successfully")
            await redis_client.aclose()
            logger.info("Redis client closed successfully")
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {cleanup_error}")

    # Create uploads directory if it doesn't exist
    if not os.path.exists("uploads"):
        os.makedirs("uploads")


if __name__ == "__main__":
    asyncio.run(main())

