from typing import Any, List, Optional, Dict
from pydantic import BaseModel, ConfigDict, Field
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from contextlib import contextmanager
import logging
from datetime import datetime
import os
import asyncio
from dotenv import load_dotenv
from llama_index.core.llms import ChatMessage, LLM
from llama_index.core.program.function_program import get_function_tool
from llama_index.core.tools import (
    BaseTool,
    ToolSelection,
)
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Context,
)
from llama_index.core.workflow.events import InputRequiredEvent, HumanResponseEvent
from llama_index.llms.openai import OpenAI
from anthropic import Anthropic
from utils import FunctionToolWithContext
import json
from redis.asyncio import Redis as AsyncRedis

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database Connection Pool
DB_POOL = SimpleConnectionPool(
    minconn=5,
    maxconn=20,
    dsn=os.getenv("POSTGRES_CONNECTION_STRING", "postgresql://postgres:@localhost:5432/mytestdb")
)

# Redis Connection with error handling
try:
    REDIS = AsyncRedis(
        host=os.getenv("REDIS_HOST", "127.0.0.1"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        db=int(os.getenv("REDIS_DB", "0")),
        decode_responses=True
    )
except Exception as e:
    logger.error(f"Failed to connect to Redis: {e}")
    raise

@contextmanager
def get_db_connection():
    """Context manager for database connections from the pool."""
    conn = DB_POOL.getconn()
    try:
        yield conn
    finally:
        DB_POOL.putconn(conn)

class CacheManager:
    """Manages caching operations."""
    def __init__(self, redis_client: AsyncRedis):
        self.redis = redis_client
        self.default_ttl = 3600  # 1 hour

    async def _safe_redis_operation(self, operation):
        """Safely execute Redis operations with retries."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return await operation()
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Redis operation failed after {max_retries} attempts: {e}")
                    raise
                await asyncio.sleep(0.1 * (2 ** attempt))

    async def get_cached_response(self, key: str) -> Optional[str]:
        return await self._safe_redis_operation(
            lambda: self.redis.get(key)
        )

    async def set_cached_response(self, key: str, value: str, ttl: int = None):
        await self._safe_redis_operation(
            lambda: self.redis.set(key, value, ex=ttl or self.default_ttl)
        )

class LLMManager:
    """Manages LLM interactions with fallback and retry logic."""
    def __init__(self):
        self.openai_llm = OpenAI(model="gpt-4", temperature=0.4)
        self.anthropic = Anthropic()
        self.claude_llm = None
        self.retry_delay = 1

    async def get_response(self, prompt: str, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                return await self.openai_llm.acomplete(prompt)
            except Exception as e:
                logger.warning(f"OpenAI API error (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    try:
                        if self.claude_llm is None:
                            self.claude_llm = self.anthropic.messages.create(
                                model="claude-3-opus-20240229",
                                max_tokens=1024,
                                temperature=0.4
                            )
                        return await self.claude_llm.acomplete(prompt)
                    except Exception as claude_error:
                        logger.error(f"Both LLM services failed: {claude_error}")
                        raise
                await asyncio.sleep(self.retry_delay * (2 ** attempt))

# ---- Pydantic models for config/llm prediction ----


class AgentConfig(BaseModel):
    """Used to configure an agent."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    system_prompt: str | None = None
    tools: list[BaseTool] | None = None
    tools_requiring_human_confirmation: list[str] = Field(default_factory=list)
    retry_attempts: int = 3
    timeout: int = 30


class TransferToAgent(BaseModel):
    """Used to transfer the user to a specific agent."""

    agent_name: str


class RequestTransfer(BaseModel):
    """Used to signal that either you don't have the tools to complete the task, or you've finished your task and want to transfer to another agent."""

    pass


# ---- Events used to orchestrate the workflow ----


class ActiveSpeakerEvent(Event):
    """Event for active speaker transitions."""
    pass


class OrchestratorEvent(Event):
    """Event for orchestrator decisions."""
    user_msg: str | None = None


class ToolCallEvent(Event):
    """Event for tool calls."""
    tool_call: ToolSelection
    tools: list[BaseTool]


class ToolCallResultEvent(Event):
    """Event for tool call results."""
    chat_message: ChatMessage


class ToolRequestEvent(InputRequiredEvent):
    """Event for tool requests."""
    tool_name: str
    tool_id: str
    tool_kwargs: dict


class ToolApprovedEvent(HumanResponseEvent):
    """Event for tool approval responses."""
    tool_name: str
    tool_id: str
    tool_kwargs: dict
    approved: bool
    response: str | None = None


class ProgressEvent(Event):
    """Event for progress updates."""
    msg: str


class ErrorEvent(Event):
    """Event for error handling."""
    error: Exception


# ---- Workflow ----

DEFAULT_ORCHESTRATOR_PROMPT = """
You are an orchestration agent.
Your job is to decide which agent to run based on the current state of the user and what they've asked to do.
You do not need to figure out dependencies between agents; the agents will handle that themselves.
Here are the agents you can choose from:
{agent_context_str}

Here is the current user state:
{user_state_str}

Please assist the user and transfer them as needed.
"""

DEFAULT_TOOL_REJECT_STR = "The tool call was not approved, likely due to a mistake or preconditions not being met."


class ConciergeAgent(Workflow):
    def __init__(
        self,
        orchestrator_prompt: str | None = None,
        default_tool_reject_str: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.orchestrator_prompt = orchestrator_prompt or DEFAULT_ORCHESTRATOR_PROMPT
        self.default_tool_reject_str = default_tool_reject_str or DEFAULT_TOOL_REJECT_STR
        self.llm_manager = LLMManager()
        self.cache_manager = CacheManager(REDIS)
        self.conversation_batch_size = int(os.getenv("CONVERSATION_BATCH_SIZE", "10"))
        self.conversation_buffer = []

    async def cache_conversation(self, user_msg: str, response: str, username: str):
        """Cache conversation and batch process to database."""
        self.conversation_buffer.append({
            'username': username,
            'user_msg': user_msg,
            'response': response,
            'timestamp': datetime.now()
        })

        if len(self.conversation_buffer) >= self.conversation_batch_size:
            await self._flush_conversation_buffer()

    async def _flush_conversation_buffer(self):
        """Flush conversation buffer to database."""
        if not self.conversation_buffer:
            return

        with get_db_connection() as conn:
            cur = conn.cursor()
            try:
                cur.executemany(
                    """
                    INSERT INTO conversations 
                    (username, user_message, agent_response, timestamp)
                    VALUES (%(username)s, %(user_msg)s, %(response)s, %(timestamp)s)
                    """,
                    self.conversation_buffer
                )
                conn.commit()
            except Exception as e:
                logger.error(f"Error flushing conversations: {e}")
                conn.rollback()
            finally:
                cur.close()

        self.conversation_buffer.clear()

    @step
    async def setup(self, ctx: Context, ev: StartEvent) -> OrchestratorEvent | StopEvent:
        """Sets up the workflow, validates inputs, and stores them in the context."""
        try:
            # Get parameters from context
            try:
                user_msg = "Hello"  # Default value
                try:
                    user_msg = await ctx.get("user_msg")
                except Exception:
                    logger.info("Using default user message")
                    
                chat_history = []
                try:
                    chat_history = await ctx.get("chat_history")
                except Exception:
                    logger.info("Using empty chat history")
                    
                llm = None
                try:
                    llm = await ctx.get("llm")
                except Exception:
                    logger.info("Using default LLM")
                    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.7)
                    
                initial_state = {}
                try:
                    initial_state = await ctx.get("initial_state")
                except Exception:
                    logger.info("Using empty initial state")
                    
                agent_configs = {}
                try:
                    agent_configs = await ctx.get("agent_configs")
                except Exception:
                    logger.info("No agent configs found")
                    
                # Check authentication status
                is_authenticated = initial_state.get("session_token") is not None
                
                # If not authenticated and not trying to login, redirect to auth agent
                if not is_authenticated:
                    login_keywords = ["login", "sign in", "authenticate", "username", "password"]
                    if any(keyword in user_msg.lower() for keyword in login_keywords):
                        # This is likely a login attempt, set active speaker to auth agent
                        await ctx.set("active_speaker", "auth")
                        
                        # Add user message to chat history
                        if user_msg and (not chat_history or chat_history[-1].role != "user" or chat_history[-1].content != user_msg):
                            chat_history.append(ChatMessage(role="user", content=user_msg))
                        
                        await ctx.set("chat_history", chat_history)
                        await ctx.set("llm", llm)
                        await ctx.set("user_state", initial_state)
                        await ctx.set("agent_configs", agent_configs)
                        
                        return ActiveSpeakerEvent()
                    else:
                        # Not trying to login, inform user to login first
                        auth_message = "Please login first. You can say something like 'I want to login' or provide your username and password."
                        return StopEvent(
                            result={
                                "response": auth_message,
                                "chat_history": chat_history + [ChatMessage(role="assistant", content=auth_message)],
                            }
                        )
                
                # User is authenticated or this is a login attempt
                # Add user message to chat history
                if user_msg and (not chat_history or chat_history[-1].role != "user" or chat_history[-1].content != user_msg):
                    chat_history.append(ChatMessage(role="user", content=user_msg))
                
                # Ensure chat history is not empty
                if not chat_history:
                    chat_history.append(ChatMessage(role="system", content="You are a helpful assistant."))
                
                # Store everything in context
                await ctx.set("chat_history", chat_history)
                await ctx.set("llm", llm)
                await ctx.set("user_state", initial_state)
                await ctx.set("agent_configs", agent_configs)
                
                # If active speaker is already set, use it
                active_speaker = None
                try:
                    active_speaker = await ctx.get("active_speaker")
                except Exception:
                    pass
                    
                if active_speaker:
                    return ActiveSpeakerEvent()
                
                # Otherwise, let orchestrator decide
                return OrchestratorEvent()
                
            except Exception as context_error:
                logger.error(f"Error getting context variables: {context_error}")
                raise
                
        except Exception as e:
            logger.error(f"Error in setup: {e}")
            return StopEvent(
                result={
                    "response": f"I encountered an error: {str(e)}. Please try again.",
                    "chat_history": [],
                    "error": True
                }
            )

    @step
    async def speak_with_sub_agent(
        self, ctx: Context, ev: ActiveSpeakerEvent
    ) -> ToolCallEvent | ToolRequestEvent | StopEvent:
        """Speaks with the active sub-agent and handles tool calls (if any)."""
        try:
            # Setup the agent for the active speaker
            active_speaker = await ctx.get("active_speaker")
            
            agent_configs = await ctx.get("agent_configs")
            agent_config = agent_configs[active_speaker]
            
            chat_history = await ctx.get("chat_history")
            llm = await ctx.get("llm")
            
            user_state = await ctx.get("user_state")
            user_state_str = "\n".join([f"{k}: {v}" for k, v in user_state.items()])
            
            system_prompt = agent_config.system_prompt
            if system_prompt:
                system_prompt = system_prompt.strip() + f"\n\nHere is the current user state:\n{user_state_str}"
            else:
                system_prompt = f"You are the {agent_config.name} agent.\n\nHere is the current user state:\n{user_state_str}"
            
            llm_input = [ChatMessage(role="system", content=system_prompt)] + chat_history
            
            # Inject the request transfer tool into the list of tools
            tools = [get_function_tool(RequestTransfer)]
            if agent_config.tools:
                tools.extend(agent_config.tools)
            
            # Get response from LLM
            response = await llm.achat_with_tools(tools, chat_history=llm_input)
            
            # Check for tool calls
            tool_calls = llm.get_tool_calls_from_response(response, error_on_no_tool_call=False)
            
            # If no tool calls, just return the response
            if not tool_calls:
                chat_history.append(response.message)
                await ctx.set("chat_history", chat_history)
                return StopEvent(
                    result={
                        "response": response.message.content,
                        "chat_history": chat_history,
                    }
                )
            
            # Store number of tool calls
            await ctx.set("num_tool_calls", len(tool_calls))
            
            # Process each tool call
            for tool_call in tool_calls:
                if tool_call.tool_name == "RequestTransfer":
                    # Agent wants to transfer
                    await ctx.set("active_speaker", None)
                    ctx.write_event_to_stream(
                        ProgressEvent(msg="Agent is requesting a transfer. Please hold.")
                    )
                    return OrchestratorEvent()
                    
                elif tool_call.tool_name in agent_config.tools_requiring_human_confirmation:
                    # Tool requires human confirmation
                    ctx.write_event_to_stream(
                        ToolRequestEvent(
                            prefix=f"Tool {tool_call.tool_name} requires human approval.",
                            tool_name=tool_call.tool_name,
                            tool_kwargs=tool_call.tool_kwargs,
                            tool_id=tool_call.tool_id,
                        )
                    )
                else:
                    # Tool can be called directly
                    ctx.send_event(
                        ToolCallEvent(tool_call=tool_call, tools=agent_config.tools)
                    )
            
            # Add response to chat history
            chat_history.append(response.message)
            await ctx.set("chat_history", chat_history)
            
            return ActiveSpeakerEvent()
            
        except Exception as e:
            logger.error(f"Error in speak_with_sub_agent: {e}")
            return StopEvent(
                result={
                    "response": f"I encountered an error while speaking with the agent: {str(e)}. Please try again.",
                    "chat_history": await ctx.get("chat_history", []),
                    "error": True
                }
            )

    @step
    async def handle_tool_approval(
        self, ctx: Context, ev: ToolApprovedEvent
    ) -> ToolCallEvent | ToolCallResultEvent:
        """Handles the approval or rejection of a tool call."""
        try:
            if ev.approved:
                # Tool was approved, execute it
                active_speaker = await ctx.get("active_speaker")
                agent_config = (await ctx.get("agent_configs"))[active_speaker]
                return ToolCallEvent(
                    tools=agent_config.tools,
                    tool_call=ToolSelection(
                        tool_id=ev.tool_id,
                        tool_name=ev.tool_name,
                        tool_kwargs=ev.tool_kwargs,
                    ),
                )
            else:
                # Tool was rejected, return rejection message
                return ToolCallResultEvent(
                    chat_message=ChatMessage(
                        role="tool",
                        content=ev.response or self.default_tool_reject_str,
                        additional_kwargs={"tool_call_id": ev.tool_id},
                    )
                )
        except Exception as e:
            logger.error(f"Error in handle_tool_approval: {e}")
            return ToolCallResultEvent(
                chat_message=ChatMessage(
                    role="tool",
                    content=f"Error processing tool approval: {str(e)}",
                    additional_kwargs={"tool_call_id": ev.tool_id},
                )
            )

    @step(num_workers=4)
    async def handle_tool_call(
        self, ctx: Context, ev: ToolCallEvent
    ) -> ToolCallResultEvent:
        """Handles the execution of a tool call."""
        try:
            tool_call = ev.tool_call
            tools_by_name = {tool.metadata.get_name(): tool for tool in ev.tools}
            
            tool = tools_by_name.get(tool_call.tool_name)
            additional_kwargs = {
                "tool_call_id": tool_call.tool_id,
                "name": tool.metadata.get_name() if tool else "unknown_tool",
            }
            
            # Check if tool exists
            if not tool:
                return ToolCallResultEvent(
                    chat_message=ChatMessage(
                        role="tool",
                        content=f"Tool {tool_call.tool_name} does not exist",
                        additional_kwargs=additional_kwargs,
                    )
                )
            
            # Execute tool
            try:
                if isinstance(tool, FunctionToolWithContext):
                    tool_output = await tool.acall(ctx, **tool_call.tool_kwargs)
                else:
                    tool_output = await tool.acall(**tool_call.tool_kwargs)
                
                # Create success message
                tool_msg = ChatMessage(
                    role="tool",
                    content=tool_output.content,
                    additional_kwargs=additional_kwargs,
                )
            except Exception as tool_error:
                # Create error message
                tool_msg = ChatMessage(
                    role="tool",
                    content=f"Encountered error in tool call: {tool_error}",
                    additional_kwargs=additional_kwargs,
                )
            
            # Log progress
            ctx.write_event_to_stream(
                ProgressEvent(
                    msg=f"Tool {tool_call.tool_name} called with {tool_call.tool_kwargs} returned {tool_msg.content}"
                )
            )
            
            return ToolCallResultEvent(chat_message=tool_msg)
            
        except Exception as e:
            logger.error(f"Error in handle_tool_call: {e}")
            return ToolCallResultEvent(
                chat_message=ChatMessage(
                    role="tool",
                    content=f"Encountered system error: {e}",
                    additional_kwargs={
                        "tool_call_id": ev.tool_call.tool_id if hasattr(ev, "tool_call") else "unknown",
                        "name": "error",
                    },
                )
            )

    @step
    async def aggregate_tool_results(
        self, ctx: Context, ev: ToolCallResultEvent
    ) -> ActiveSpeakerEvent:
        """Collects the results of all tool calls and updates the chat history."""
        try:
            # Get number of expected tool calls
            num_tool_calls = await ctx.get("num_tool_calls", 1)
            
            # Collect all tool call results
            results = ctx.collect_events(ev, [ToolCallResultEvent] * num_tool_calls)
            if not results:
                logger.warning("No tool call results collected")
                return ActiveSpeakerEvent()
            
            # Add results to chat history
            chat_history = await ctx.get("chat_history")
            for result in results:
                chat_history.append(result.chat_message)
            await ctx.set("chat_history", chat_history)
            
            return ActiveSpeakerEvent()
            
        except Exception as e:
            logger.error(f"Error in aggregate_tool_results: {e}")
            return ActiveSpeakerEvent()

    @step
    async def orchestrator(
        self, ctx: Context, ev: OrchestratorEvent
    ) -> ActiveSpeakerEvent | StopEvent:
        """Decides which agent to run next, if any."""
        try:
            # Get agent configs and chat history
            agent_configs = await ctx.get("agent_configs")
            chat_history = await ctx.get("chat_history")
            
            # Get user state
            user_state = await ctx.get("user_state")
            is_authenticated = user_state.get("session_token") is not None
            
            # If not authenticated, force auth agent
            if not is_authenticated:
                if "auth" in agent_configs:
                    await ctx.set("active_speaker", "auth")
                    ctx.write_event_to_stream(
                        ProgressEvent(msg="Transferring to authentication agent")
                    )
                    return ActiveSpeakerEvent()
                else:
                    # No auth agent available
                    auth_message = "Authentication is required, but no authentication agent is available."
                    return StopEvent(
                        result={
                            "response": auth_message,
                            "chat_history": chat_history + [ChatMessage(role="assistant", content=auth_message)],
                        }
                    )
            
            # Format agent descriptions
            agent_context_str = ""
            for agent_name, agent_config in agent_configs.items():
                agent_context_str += f"{agent_name}: {agent_config.description}\n"
            
            user_state_str = "\n".join([f"{k}: {v}" for k, v in user_state.items()])
            
            # Format system prompt
            system_prompt = self.orchestrator_prompt.format(
                agent_context_str=agent_context_str, user_state_str=user_state_str
            )
            
            # Prepare LLM input
            llm_input = [ChatMessage(role="system", content=system_prompt)] + chat_history
            llm = await ctx.get("llm")
            
            # Define tools
            tools = [get_function_tool(TransferToAgent)]
            
            # Get LLM response
            response = await llm.achat_with_tools(tools, chat_history=llm_input)
            
            # Check for tool calls
            tool_calls = llm.get_tool_calls_from_response(response, error_on_no_tool_call=False)
            
            # If no tool calls, return response
            if not tool_calls:
                chat_history.append(response.message)
                return StopEvent(
                    result={
                        "response": response.message.content,
                        "chat_history": chat_history,
                    }
                )
            
            # Get selected agent
            tool_call = tool_calls[0]
            selected_agent = tool_call.tool_kwargs["agent_name"]
            
            # Validate selected agent
            if selected_agent not in agent_configs:
                chat_history.append(ChatMessage(
                    role="assistant", 
                    content=f"I tried to transfer you to {selected_agent}, but that agent doesn't exist. Let me help you directly."
                ))
                return StopEvent(
                    result={
                        "response": f"I tried to transfer you to {selected_agent}, but that agent doesn't exist. Let me help you directly.",
                        "chat_history": chat_history,
                    }
                )
            
            # Set active speaker
            await ctx.set("active_speaker", selected_agent)
            
            # Log progress
            ctx.write_event_to_stream(
                ProgressEvent(msg=f"Transferring to agent {selected_agent}")
            )
            
            return ActiveSpeakerEvent()
            
        except Exception as e:
            logger.error(f"Error in orchestrator: {e}")
            return StopEvent(
                result={
                    "response": f"I encountered an error while deciding which agent to use: {str(e)}. Let me help you directly instead.",
                    "chat_history": await ctx.get("chat_history", []),
                    "error": True
                }
            )

    async def cleanup(self):
        """Cleanup resources."""
        try:
            await self._flush_conversation_buffer()
            try:
                await REDIS.aclose()
            except Exception as redis_error:
                logger.error(f"Redis cleanup error: {redis_error}")
            
            try:
                DB_POOL.closeall()
            except Exception as db_error:
                logger.error(f"Database cleanup error: {db_error}")
                
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    def __del__(self):
        """Ensure cleanup on deletion."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.cleanup())
        except Exception as e:
            logger.error(f"Cleanup error in __del__: {e}")
