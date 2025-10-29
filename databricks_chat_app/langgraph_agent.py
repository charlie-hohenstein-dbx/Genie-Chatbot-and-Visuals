"""
LangGraph Agent with MCP Tools - Following Official Databricks Pattern
Based on: https://docs.databricks.com/aws/en/notebooks/source/generative-ai/langgraph-mcp-tool-calling-agent.html
"""
import asyncio
from typing import Annotated, Any, List, Optional, Sequence, TypedDict
from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode
from pydantic import create_model
import nest_asyncio

nest_asyncio.apply()


class AgentState(TypedDict):
    """State of the agent conversation"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    custom_inputs: Optional[dict[str, Any]]
    custom_outputs: Optional[dict[str, Any]]


class MCPTool(BaseTool):
    """Custom LangChain tool that wraps MCP server functionality"""

    def __init__(
        self,
        name: str,
        description: str,
        args_schema: type,
        server_url: str,
        ws: WorkspaceClient,
    ):
        super().__init__(name=name, description=description, args_schema=args_schema)
        object.__setattr__(self, "server_url", server_url)
        object.__setattr__(self, "workspace_client", ws)

    def _run(self, **kwargs) -> str:
        """Execute the MCP tool"""
        mcp_client = DatabricksMCPClient(
            server_url=self.server_url, workspace_client=self.workspace_client
        )
        response = mcp_client.call_tool(self.name, kwargs)
        return "".join([c.text for c in response.content])


def get_managed_mcp_tools(ws: WorkspaceClient, server_url: str):
    """Get tools from a managed MCP server"""
    mcp_client = DatabricksMCPClient(server_url=server_url, workspace_client=ws)
    return mcp_client.list_tools()


def create_langchain_tool_from_mcp(mcp_tool, server_url: str, ws: WorkspaceClient):
    """Create a LangChain tool from an MCP tool definition"""
    schema = mcp_tool.inputSchema.copy()
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    # Map JSON schema types to Python types
    TYPE_MAPPING = {"integer": int, "number": float, "boolean": bool}
    field_definitions = {}
    
    for field_name, field_info in properties.items():
        field_type_str = field_info.get("type", "string")
        field_type = TYPE_MAPPING.get(field_type_str, str)

        if field_name in required:
            field_definitions[field_name] = (field_type, ...)
        else:
            field_definitions[field_name] = (field_type, None)

    # Dynamically create Pydantic schema
    args_schema = create_model(f"{mcp_tool.name}Args", **field_definitions)

    return MCPTool(
        name=mcp_tool.name,
        description=mcp_tool.description or f"Tool: {mcp_tool.name}",
        args_schema=args_schema,
        server_url=server_url,
        ws=ws,
    )


def create_mcp_tools(ws: WorkspaceClient, managed_server_urls: List[str]) -> List[MCPTool]:
    """Create LangChain tools from managed MCP servers"""
    tools = []

    for server_url in managed_server_urls:
        try:
            print(f"[LangGraph] Loading tools from: {server_url}")
            mcp_tools = get_managed_mcp_tools(ws, server_url)
            
            for mcp_tool in mcp_tools:
                tool = create_langchain_tool_from_mcp(mcp_tool, server_url, ws)
                tools.append(tool)
                print(f"[LangGraph] Loaded tool: {tool.name}")
                
        except Exception as e:
            print(f"[LangGraph] Error loading tools from {server_url}: {e}")

    return tools


class LangGraphMCPAgent:
    """LangGraph agent that uses MCP tools with Databricks model serving"""
    
    def __init__(
        self, 
        workspace_client: WorkspaceClient, 
        genie_space_id: str,
        llm_endpoint: str = "databricks-claude-3-7-sonnet"
    ):
        self.workspace_client = workspace_client
        self.host = workspace_client.config.host
        self.llm_endpoint = llm_endpoint
        self.genie_space_id = genie_space_id
        
        # Get OpenAI-compatible client for model serving
        self.llm = workspace_client.serving_endpoints.get_open_ai_client()
        
        # Configure MCP server URLs
        self.managed_mcp_urls = [
            f"{self.host}/api/2.0/mcp/genie/{genie_space_id}",
            f"{self.host}/api/2.0/mcp/functions/chat_app_demo/dev",
        ]
        
        # Initialize tools and build graph
        print(f"[LangGraph] Initializing agent with endpoint: {llm_endpoint}")
        self.tools = create_mcp_tools(self.workspace_client, self.managed_mcp_urls)
        self.agent = self._build_agent()
    
    def _build_agent(self):
        """Build the LangGraph agent workflow"""
        
        def should_continue(state: AgentState):
            """Decide whether to continue to tools or end"""
            messages = state["messages"]
            last_message = messages[-1]
            
            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                return "continue"
            return "end"
        
        def call_model(state: AgentState):
            """Call the LLM with tool definitions"""
            messages = state["messages"]
            print(f"[LangGraph] call_model invoked with {len(messages)} messages")
            
            # Add system prompt and convert messages to OpenAI format
            system_prompt = """You are a helpful assistant that can query data and create charts.

                WORKFLOW:
                1. If user asks for data or a chart, call query_space_01f0ab8c079d17b8a00584e70d2ac18c to get data
                2. If user wants a chart, call chat_app_demo__dev__genie_to_chart EXACTLY ONCE with:
                - genie_response_json: the ENTIRE raw JSON response from query_space
                - chart_type: "bar", "line", or "pie"
                3. IMMEDIATELY STOP after calling genie_to_chart and provide a brief summary

                CRITICAL RULES:
                - Use EXACT tool names: query_space_01f0ab8c079d17b8a00584e70d2ac18c and chat_app_demo__dev__genie_to_chart
                - Call genie_to_chart ONLY ONCE per request - NEVER call it multiple times
                - STOP calling tools after genie_to_chart returns - the chart is already created
                - Do NOT query data again or recreate charts - ONE chart is sufficient"""
            
            # Map LangChain message types to OpenAI roles
            def get_role(msg):
                if isinstance(msg, HumanMessage):
                    return "user"
                elif isinstance(msg, AIMessage):
                    return "assistant"
                elif hasattr(msg, 'type'):
                    # Map: human->user, ai->assistant, system->system
                    role_map = {"human": "user", "ai": "assistant", "system": "system"}
                    return role_map.get(msg.type, "user")
                return "user"
            
            full_messages = [
                {"role": "system", "content": system_prompt}
            ] + [
                {"role": get_role(m), "content": m.content}
                for m in messages
            ]
            
            # Convert tools to OpenAI format
            tools_format = []
            for tool in self.tools:
                # Get schema from args_schema
                schema = tool.args_schema.model_json_schema()
                tools_format.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": schema
                    }
                })
            
            # Call LLM
            print(f"[LangGraph] Calling LLM with {len(tools_format)} tools")
            response = self.llm.chat.completions.create(
                model=self.llm_endpoint,
                messages=full_messages,
                tools=tools_format if tools_format else None,
                tool_choice="auto",  # Let LLM decide when to use tools
                temperature=0.1
            )
            
            # Debug: Check what the LLM returned
            print(f"[LangGraph] LLM response content: {response.choices[0].message.content[:200] if response.choices[0].message.content else 'None'}")
            print(f"[LangGraph] Tool calls present: {bool(response.choices[0].message.tool_calls)}")
            
            # Convert to AIMessage
            ai_msg = AIMessage(content=response.choices[0].message.content or "")
            
            # Add tool calls if present
            if response.choices[0].message.tool_calls:
                import json
                ai_msg.tool_calls = [
                    {
                        "name": tc.function.name,
                        "args": json.loads(tc.function.arguments),
                        "id": tc.id
                    }
                    for tc in response.choices[0].message.tool_calls
                ]
            
            return {"messages": [ai_msg]}
        
        def call_tools(state: AgentState):
            """Execute tools and log results"""
            print(f"[LangGraph] Executing tools...")
            messages = state["messages"]
            last_message = messages[-1]
            
            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                for tool_call in last_message.tool_calls:
                    print(f"[LangGraph] Tool: {tool_call['name']}")
                    print(f"[LangGraph] Args: {tool_call['args']}")
            
            # Execute the tools
            tool_node = ToolNode(self.tools)
            result = tool_node.invoke(state)
            
            # Log the tool results
            if result and "messages" in result:
                for msg in result["messages"]:
                    if hasattr(msg, 'content'):
                        content_preview = str(msg.content)[:500] if msg.content else "None"
                        print(f"[LangGraph] Tool result (first 500 chars): {content_preview}")
            
            return result
        
        # Build the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", RunnableLambda(call_model))
        workflow.add_node("tools", call_tools)
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add edges
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "continue": "tools",
                "end": END
            }
        )
        workflow.add_edge("tools", "agent")
        
        # Compile with recursion limit to prevent infinite loops
        return workflow.compile()
    
    def chat(self, user_message: str) -> dict:
        """
        Process a user message through the LangGraph agent
        Returns: {"response": str, "messages": list, "error": str or None}
        """
        try:
            # Create initial state
            initial_state = {
                "messages": [HumanMessage(content=user_message)]
            }
            
            # Run the graph with recursion limit to prevent infinite loops
            print(f"[LangGraph] Processing: {user_message}")
            result = self.agent.invoke(
                initial_state,
                config={"recursion_limit": 20}
            )
            
            # Extract final response
            messages = result.get("messages", [])
            final_message = messages[-1] if messages else None
            
            if final_message:
                response_text = final_message.content if hasattr(final_message, 'content') else str(final_message)
                
                return {
                    "response": response_text,
                    "messages": messages,
                    "error": None
                }
            else:
                return {
                    "response": "No response generated",
                    "messages": [],
                    "error": "Empty response from agent"
                }
                
        except Exception as e:
            error_msg = str(e)
            print(f"[LangGraph] Error: {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "response": None,
                "messages": [],
                "error": error_msg
            }
