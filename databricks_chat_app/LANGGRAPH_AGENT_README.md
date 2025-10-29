# LangGraph Agent Architecture Guide

## Overview

`langgraph_agent.py` implements a **stateful AI agent** using LangGraph that orchestrates interactions between:
- **Databricks Model Serving** (Claude 3.7 Sonnet LLM)
- **Genie MCP Server** (natural language data queries)
- **Unity Catalog Functions MCP Server** (chart generation)

Based on the official [Databricks LangGraph MCP pattern](https://docs.databricks.com/aws/en/notebooks/source/generative-ai/langgraph-mcp-tool-calling-agent.html).

---

## Architecture Diagram

```
┌─────────────┐
│   User      │
│   Query     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│   LangGraphMCPAgent                 │
│   ┌─────────────────────────────┐   │
│   │  LangGraph Workflow (Graph) │   │
│   │                             │   │
│   │  ┌──────────┐              │   │
│   │  │  Agent   │              │   │
│   │  │  Node    │◄─────────┐   │   │
│   │  └────┬─────┘          │   │   │
│   │       │                │   │   │
│   │       ▼                │   │   │
│   │  ┌──────────┐          │   │   │
│   │  │  Tools   │──────────┘   │   │
│   │  │  Node    │              │   │
│   │  └────┬─────┘              │   │
│   │       │                    │   │
│   │       ▼                    │   │
│   │    [END]                   │   │
│   └─────────────────────────────┘   │
└────────┬────────────┬────────────────┘
         │            │
         ▼            ▼
┌────────────┐   ┌──────────────┐
│ Genie MCP  │   │ UC Functions │
│  Server    │   │  MCP Server  │
└────────────┘   └──────────────┘
```

---

## How the Agent Knows Which Tools to Call

### The Magic: LLM-Driven Tool Selection

**The agent doesn't hardcode logic** - instead, the **LLM (Claude) decides** which tools to call based on:
1. Tool descriptions from MCP servers
2. System prompt instructions
3. User's natural language request

### Step-by-Step: Tool Discovery → Execution

#### **Phase 1: Initialization (Lines 112-135)**

When you create a `LangGraphMCPAgent`:

```python
agent = LangGraphMCPAgent(
    workspace_client=workspace_client,
    genie_space_id="01f0ab8c..."
)
```

**What happens:**

```python
# Line 127-130: Configure MCP server URLs
self.managed_mcp_urls = [
    f"{host}/api/2.0/mcp/genie/{genie_space_id}",       # Genie queries
    f"{host}/api/2.0/mcp/functions/chat_app_demo/dev",  # UC functions
]

# Line 134: Discover ALL tools from MCP servers
self.tools = create_mcp_tools(self.workspace_client, self.managed_mcp_urls)
```

**Tools discovered** (example):
```
[LangGraph] Loaded tool: query_space_01f0ab8c079d17b8a00584e70d2ac18c
[LangGraph] Loaded tool: poll_response_01f0ab8c079d17b8a00584e70d2ac18c
[LangGraph] Loaded tool: chat_app_demo__dev__genie_to_chart
```

#### **Phase 2: Tool Schema Extraction (Lines 52-106)**

For **each MCP tool**, the agent extracts:

```python
# Line 54-55: Connect to MCP server and list tools
mcp_client = DatabricksMCPClient(server_url=server_url, workspace_client=ws)
mcp_tools = mcp_client.list_tools()

# Each MCP tool has:
# - name: "query_space_01f0ab8c079d17b8a00584e70d2ac18c"
# - description: "Query data from Genie space using natural language"
# - inputSchema: JSON schema defining parameters
```

**Example MCP Tool Schema**:
```json
{
  "name": "chat_app_demo__dev__genie_to_chart",
  "description": "Transform Genie response into a Plotly chart. Takes raw Genie JSON and chart type.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "genie_response_json": {
        "type": "string",
        "description": "Raw JSON from query_space tool"
      },
      "chart_type": {
        "type": "string", 
        "description": "bar, line, or pie"
      }
    },
    "required": ["genie_response_json", "chart_type"]
  }
}
```

**Conversion to LangChain** (Lines 58-86):
```python
# Create Pydantic model from JSON schema
args_schema = create_model(f"{mcp_tool.name}Args", **field_definitions)

# Wrap in MCPTool
return MCPTool(
    name=mcp_tool.name,
    description=mcp_tool.description,
    args_schema=args_schema,
    server_url=server_url,
    ws=ws
)
```

#### **Phase 3: Presenting Tools to the LLM (Lines 189-201)**

When the agent node runs, it converts **all tools to OpenAI function calling format**:

```python
# Line 190-201: Convert tools to OpenAI format
tools_format = []
for tool in self.tools:
    schema = tool.args_schema.model_json_schema()
    tools_format.append({
        "type": "function",
        "function": {
            "name": tool.name,                    # ← LLM sees this
            "description": tool.description,       # ← LLM reads this
            "parameters": schema                   # ← LLM uses this
        }
    })
```

**What the LLM sees** (simplified):
```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "query_space_01f0ab8c079d17b8a00584e70d2ac18c",
        "description": "Query data from Genie using natural language",
        "parameters": {"query": "string"}
      }
    },
    {
      "type": "function",
      "function": {
        "name": "chat_app_demo__dev__genie_to_chart",
        "description": "Transform Genie response into Plotly chart",
        "parameters": {
          "genie_response_json": "string",
          "chart_type": "string"
        }
      }
    }
  ]
}
```

#### **Phase 4: LLM Makes the Decision (Lines 205-211)**

The LLM receives:
1. **System prompt** with workflow instructions
2. **Conversation history** 
3. **All available tools** with descriptions
4. **User's message**: "Show me trips per year as a bar chart"

```python
# Line 205-211: Call LLM with tools
response = self.llm.chat.completions.create(
    model=self.llm_endpoint,
    messages=full_messages,           # System prompt + history
    tools=tools_format,               # All available tools
    tool_choice="auto",               # LLM decides!
    temperature=0.1
)
```

**The LLM thinks**:
> "User wants trips per year as a bar chart. 
> I see a tool called 'query_space_...' that can query data.
> I also see 'genie_to_chart' that creates charts.
> My system prompt says to first query data, then create chart.
> I'll call query_space first with query='trips per year'."

**LLM returns**:
```json
{
  "message": {
    "content": "I'll query the data first...",
    "tool_calls": [
      {
        "id": "call_123",
        "function": {
          "name": "query_space_01f0ab8c079d17b8a00584e70d2ac18c",
          "arguments": "{\"query\": \"Count trips per year\"}"
        }
      }
    ]
  }
}
```

#### **Phase 5: Tool Execution (Lines 234-256)**

The `call_tools` node receives the LLM's decision:

```python
# Line 240-243: Extract tool calls from AI message
if isinstance(last_message, AIMessage) and last_message.tool_calls:
    for tool_call in last_message.tool_calls:
        print(f"[LangGraph] Tool: {tool_call['name']}")
        print(f"[LangGraph] Args: {tool_call['args']}")
```

**Execute via ToolNode** (Line 246-247):
```python
tool_node = ToolNode(self.tools)  # Has all discovered tools
result = tool_node.invoke(state)   # Matches name → executes MCP call
```

**Behind the scenes** (in MCPTool._run, Lines 43-49):
```python
def _run(self, **kwargs) -> str:
    # Connect to the MCP server for this specific tool
    mcp_client = DatabricksMCPClient(
        server_url=self.server_url,       # e.g., /api/2.0/mcp/genie/...
        workspace_client=self.workspace_client
    )
    # Call the tool by name with arguments
    response = mcp_client.call_tool(self.name, kwargs)
    return "".join([c.text for c in response.content])
```

#### **Phase 6: Results Back to LLM**

The tool result is added to the conversation:

```
Messages: [
  HumanMessage("Show me trips per year as bar chart"),
  AIMessage(tool_calls=[...]),
  ToolMessage(content='{"query": "...", "statement_response": {...}}')  ← Result
]
```

**Graph loops back to agent node**, LLM sees the data and decides:
> "Perfect! I have the Genie response. Now I'll call genie_to_chart 
> with this data and chart_type='bar'."

---

### Key Insight: No Hardcoding!

**The agent never has code like**:
```python
if "chart" in user_message:
    call_chart_function()
```

Instead:
1. ✅ MCP servers **describe** what tools can do
2. ✅ LLM **reads** those descriptions
3. ✅ LLM **decides** which tools to call and when
4. ✅ Agent **executes** the LLM's decisions

This makes the agent:
- **Flexible**: Add new UC functions → LLM discovers them automatically
- **Intelligent**: LLM can chain tools in creative ways
- **Maintainable**: No brittle if/else logic to update

---

## Key Components

### 1. **AgentState** (Lines 21-25)
Tracks the conversation state as it flows through the graph.

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    custom_inputs: Optional[dict[str, Any]]
    custom_outputs: Optional[dict[str, Any]]
```

- **`messages`**: Full conversation history (user messages, AI responses, tool calls)
- **`add_messages`**: LangGraph reducer that appends new messages to history
- **`custom_inputs/outputs`**: Reserved for future extensibility

### 2. **MCPTool** (Lines 28-49)
Custom LangChain tool wrapper for MCP server functionality.

**Key Method**: `_run(self, **kwargs)` 
- Creates an MCP client connection
- Calls the specified tool on the MCP server
- Returns the text response

### 3. **Tool Loading** (Lines 89-106)
`create_mcp_tools()` discovers and wraps all available MCP tools:

```python
managed_server_urls = [
    f"{host}/api/2.0/mcp/genie/{genie_space_id}",      # Genie queries
    f"{host}/api/2.0/mcp/functions/catalog/schema",    # UC functions
]
```

**Process**:
1. Connect to each MCP server
2. Call `list_tools()` to discover available tools
3. Convert MCP tool schemas to LangChain-compatible Pydantic models
4. Wrap each tool in `MCPTool` class

---

## What Happens When You Call `self.agent.invoke()`?

### The Execution Engine

When you call `agent.chat("Show me trips per year")`, it eventually calls:

```python
# Line 295-298 in langgraph_agent.py
result = self.agent.invoke(
    initial_state,
    config={"recursion_limit": 20}
)
```

**What is `self.agent`?**

```python
# Line 135: Created during initialization
self.agent = self._build_agent()

# Line 280: _build_agent returns a compiled graph
return workflow.compile()
```

So `self.agent` is a **compiled StateGraph** - a finite state machine that knows:
- Which nodes exist (`agent`, `tools`)
- How they're connected (edges)
- Where to start (entry point)
- When to stop (`END`)

### What `.invoke()` Does: Step-by-Step Execution

#### **1. Initialize State**

```python
initial_state = {
    "messages": [HumanMessage(content=user_message)]
}
```

The graph receives this state and begins execution.

#### **2. Start at Entry Point**

```python
# Line 266: Entry point is set to "agent"
workflow.set_entry_point("agent")
```

So execution **always** starts at the `agent` node.

#### **3. Execute Node: `agent`**

```python
# The agent node runs the call_model function
workflow.add_node("agent", RunnableLambda(call_model))
```

**What happens in `call_model`:**
1. Takes current state (messages)
2. Formats them for OpenAI API
3. Sends to LLM with tool definitions
4. Returns new state with AI response

**State after this node:**
```python
{
    "messages": [
        HumanMessage("Show me trips per year"),
        AIMessage(tool_calls=[{name: "query_space", args: {...}}])
    ]
}
```

#### **4. Evaluate Conditional Edge**

```python
# Line 269-276: Conditional routing logic
workflow.add_conditional_edges(
    "agent",
    should_continue,  # ← This function decides
    {
        "continue": "tools",
        "end": END
    }
)
```

**The `should_continue` function checks:**
```python
# Line 140-147
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "continue"  # → Go to tools node
    return "end"          # → Go to END
```

**In our case:** Last message has `tool_calls`, so return `"continue"` → route to `tools` node.

#### **5. Execute Node: `tools`**

```python
workflow.add_node("tools", call_tools)
```

**What happens in `call_tools`:**
1. Extract tool calls from last AI message
2. Execute each tool via MCP
3. Add tool results to messages
4. Return updated state

**State after this node:**
```python
{
    "messages": [
        HumanMessage("Show me trips per year"),
        AIMessage(tool_calls=[...]),
        ToolMessage(content='{"query": "...", "data": [...]}')  # ← Added
    ]
}
```

#### **6. Follow Edge Back to `agent`**

```python
# Line 277: After tools, always go back to agent
workflow.add_edge("tools", "agent")
```

The graph **loops back** to the `agent` node with the updated state.

#### **7. Execute `agent` Node Again (Loop)**

Now the LLM sees:
- Original user message
- Its previous tool call
- The tool result

**The LLM decides:**
> "I have the data. User wanted a bar chart. 
> I'll call genie_to_chart now."

**New state:**
```python
{
    "messages": [
        HumanMessage("Show me trips per year"),
        AIMessage(tool_calls=[query_space]),
        ToolMessage(content='...genie data...'),
        AIMessage(tool_calls=[{name: "genie_to_chart", args: {...}}])  # ← New
    ]
}
```

#### **8. Conditional Edge → `tools` Again**

Still has `tool_calls`, so route to `tools` node again.

#### **9. Execute `tools` Node Again**

Execute `genie_to_chart`, add result to state.

#### **10. Back to `agent` Node**

#### **11. LLM Decides to Stop**

Now the LLM sees all the data and the chart has been created.

**The LLM returns:**
```python
AIMessage(content="Here's your bar chart showing trips per year.")
# NO tool_calls this time!
```

#### **12. Conditional Edge → `END`**

```python
def should_continue(state: AgentState):
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "continue"
    return "end"  # ← No tool calls, so END
```

#### **13. Graph Execution Complete**

`.invoke()` returns the **final state**:

```python
result = {
    "messages": [
        HumanMessage("Show me trips per year"),
        AIMessage(tool_calls=[query_space]),
        ToolMessage(content='...'),
        AIMessage(tool_calls=[genie_to_chart]),
        ToolMessage(content='{"plotly_json": {...}}'),
        AIMessage(content="Here's your bar chart...")
    ]
}
```

---

### Visual Representation of `.invoke()` Execution

```
invoke(initial_state) called
    ↓
┌─────────────────────────────────────────────────────────┐
│  Graph Execution Loop                                   │
│                                                          │
│  Iteration 1:                                            │
│    → agent node: call_model()                            │
│       LLM returns: tool_calls=[query_space]              │
│    → should_continue: "continue"                         │
│    → tools node: execute query_space                     │
│    → back to agent                                       │
│                                                          │
│  Iteration 2:                                            │
│    → agent node: call_model()                            │
│       LLM returns: tool_calls=[genie_to_chart]           │
│    → should_continue: "continue"                         │
│    → tools node: execute genie_to_chart                  │
│    → back to agent                                       │
│                                                          │
│  Iteration 3:                                            │
│    → agent node: call_model()                            │
│       LLM returns: "Here's your chart" (no tool_calls)   │
│    → should_continue: "end"                              │
│    → END                                                 │
│                                                          │
└─────────────────────────────────────────────────────────┘
    ↓
return final_state
```

---

### Why Use `.invoke()` Instead of a Regular Loop?

**A naive implementation might look like:**
```python
while True:
    response = call_llm(messages)
    if response.tool_calls:
        results = execute_tools(response.tool_calls)
        messages.append(results)
    else:
        break
```

**Problems with this approach:**
- ❌ No state management
- ❌ Hard to debug (where are we in the loop?)
- ❌ No recursion limits
- ❌ Can't pause/resume
- ❌ Can't visualize execution
- ❌ Hard to add branching logic

**LangGraph's `.invoke()` provides:**
- ✅ **State tracking**: Every node modifies state explicitly
- ✅ **Conditional routing**: `should_continue` decides flow
- ✅ **Recursion limits**: `config={"recursion_limit": 20}`
- ✅ **Debuggability**: Can trace which nodes ran
- ✅ **Checkpointing**: Can save/resume execution
- ✅ **Visualization**: Can render the graph
- ✅ **Human-in-the-loop**: Can pause for approval

---

### Config Options for `.invoke()`

```python
result = self.agent.invoke(
    initial_state,
    config={
        "recursion_limit": 20,        # Max iterations before stopping
        "configurable": {...},        # Custom config per run
        # "thread_id": "abc123",      # For checkpointing/memory
    }
)
```

**`recursion_limit`**: 
- Each time the graph transitions between nodes = 1 recursion
- In our example: agent → tools → agent → tools → agent → END = 5 recursions
- If limit hit, raises `RecursionLimitError`

**Why we set it to 20:**
- Most queries complete in 3-5 recursions
- Prevents infinite loops if LLM gets confused
- Fails fast rather than hanging forever

---

### What `.invoke()` Returns

```python
result = {
    "messages": [...],           # Full conversation history
    "custom_inputs": None,       # Reserved for future use
    "custom_outputs": None       # Reserved for future use
}
```

**In `chat()` method (Line 300-311), we extract:**
```python
messages = result.get("messages", [])
final_message = messages[-1]
response_text = final_message.content

return {
    "response": response_text,      # Just the final text
    "messages": messages,           # Full history for parsing
    "error": None
}
```

---

### Advanced: Streaming with `.stream()`

Instead of `.invoke()`, you can use `.stream()`:

```python
for chunk in self.agent.stream(initial_state):
    print(chunk)  # Each node execution as it happens
```

**Output:**
```python
{'agent': {'messages': [AIMessage(...)]}}
{'tools': {'messages': [ToolMessage(...)]}}
{'agent': {'messages': [AIMessage(...)]}}
# ...
```

This enables **real-time updates** in the UI as the agent works!

---

## The LangGraph Workflow

### Graph Structure (Lines 258-280)

```
START → agent → [decision] → tools → agent → [decision] → END
                    ↓                           ↓
                   END                         END
```

**Two Nodes**:
1. **`agent`**: Calls the LLM to decide next action
2. **`tools`**: Executes tools chosen by the LLM

**Conditional Logic**:
- If LLM returns tool calls → go to `tools` node
- If LLM returns only text → go to `END`

### Node 1: `call_model` (Lines 149-232)

**Responsibilities**:
1. Add system prompt with instructions
2. Convert conversation history to OpenAI format
3. Convert tools to OpenAI function calling format
4. Call LLM with tools available
5. Parse LLM response (text or tool calls)

**System Prompt Strategy**:
```
WORKFLOW:
1. Call query_space to get data
2. Call genie_to_chart ONCE to create chart
3. STOP and summarize
```

**Key Design Choice**: Explicit, step-by-step instructions prevent the agent from looping.

### Node 2: `call_tools` (Lines 234-256)

**Responsibilities**:
1. Extract tool calls from last AI message
2. Execute each tool via MCP
3. Add tool results to message history
4. Return to `agent` node for LLM to process results

---

## Execution Flow Example

### User: "Show me trips per year as a bar chart"

**Step 1: Initial Call to Agent Node**
```
State: { messages: [HumanMessage("Show me trips per year...")] }
↓
Agent calls LLM with tools
↓
LLM decides: call query_space_01f0ab8c079d17b8a00584e70d2ac18c
```

**Step 2: Tools Node**
```
State: { messages: [HumanMessage(...), AIMessage(tool_calls=[...])] }
↓
Execute: query_space("trips per year")
↓
Add ToolMessage with Genie JSON response
```

**Step 3: Back to Agent Node**
```
State: { messages: [HumanMessage(...), AIMessage(...), ToolMessage(...)] }
↓
LLM sees data, decides: call chat_app_demo__dev__genie_to_chart
```

**Step 4: Tools Node**
```
Execute: genie_to_chart(genie_response_json=..., chart_type="bar")
↓
Add ToolMessage with Plotly JSON
```

**Step 5: Back to Agent Node**
```
LLM sees chart created, generates summary text
↓
No more tool calls → END
```

---

## Important Features

### 1. **Recursion Limit** (Line 297)
```python
result = self.agent.invoke(initial_state, config={"recursion_limit": 20})
```
Prevents infinite loops if the agent gets stuck.

### 2. **Debug Logging**
Every step prints debug info:
- `[LangGraph] call_model invoked with N messages`
- `[LangGraph] Tool: tool_name`
- `[LangGraph] Tool result: ...`

### 3. **OpenAI Compatibility**
Uses `workspace_client.serving_endpoints.get_open_ai_client()` for structured tool calling support.

### 4. **Dynamic Tool Discovery**
Tools are discovered at runtime from MCP servers, not hardcoded.

---

## Configuration

### Model Endpoint
```python
LangGraphMCPAgent(
    workspace_client=workspace_client,
    genie_space_id="01f0ab8c079d17b8a00584e70d2ac18c",
    llm_endpoint="databricks-claude-3-7-sonnet"  # ← Change model here
)
```

**Supported Models**:
- `databricks-claude-3-7-sonnet` ✅ (Best tool calling)
- `databricks-meta-llama-3-1-405b-instruct` ✅
- `databricks-llama-4-maverick` ⚠️ (Unreliable tool calling)

### MCP Server URLs
```python
self.managed_mcp_urls = [
    f"{host}/api/2.0/mcp/genie/{genie_space_id}",
    f"{host}/api/2.0/mcp/functions/chat_app_demo/dev",  # ← Your UC catalog/schema
]
```

---

## Usage in Streamlit App

```python
from databricks_chat_app.langgraph_agent import LangGraphMCPAgent
from databricks.sdk import WorkspaceClient

# Initialize
workspace_client = WorkspaceClient()
agent = LangGraphMCPAgent(
    workspace_client=workspace_client,
    genie_space_id="your-genie-space-id"
)

# Chat
response = agent.chat("Show me sales by region as a pie chart")

# Response structure
{
    "response": "Here's the pie chart showing sales by region...",
    "messages": [...],  # Full conversation history
    "error": None
}
```

---

## Troubleshooting

### ❌ Agent Keeps Looping
**Symptom**: Hits recursion limit (20)

**Causes**:
1. LLM isn't detecting tool call success
2. Tool returns unexpected format
3. System prompt not explicit enough

**Fix**:
- Check `[LangGraph] Tool result` logs
- Ensure UC function returns expected JSON format
- Update system prompt with stricter instructions

### ❌ "Tool not found" Error
**Symptom**: Agent tries to call tool that doesn't exist

**Causes**:
1. MCP server URL incorrect
2. UC function not deployed
3. Tool name mismatch

**Fix**:
- Check `[LangGraph] Loaded tool:` logs at startup
- Verify UC function exists: `SELECT * FROM chat_app_demo.dev.genie_to_chart`
- Use exact tool name from logs in system prompt

### ❌ "Unsupported message role: human"
**Symptom**: OpenAI API error

**Fix**: Already handled in `get_role()` function (lines 171-180) which maps LangChain message types to OpenAI roles.

---

## Key Differences from Simple Agent

| Feature | Simple Agent | LangGraph Agent |
|---------|-------------|-----------------|
| **State Management** | Manual in loop | Built-in graph state |
| **Tool Execution** | Sequential | Parallel-capable |
| **Error Recovery** | Try/except | Graph rollback |
| **Debugging** | Print statements | LangSmith traces |
| **Extensibility** | Hard to extend | Add nodes/edges |
| **Async Support** | Limited | Full async |

---

## Advanced: Extending the Agent

### Add a New Node
```python
def validate_output(state: AgentState):
    """Custom validation logic"""
    # Check if chart data is valid
    # Return modified state
    return state

workflow.add_node("validator", validate_output)
workflow.add_edge("tools", "validator")
workflow.add_edge("validator", "agent")
```

### Add Memory
```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
workflow.compile(checkpointer=checkpointer)
```

### Add Human-in-the-Loop
```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model=llm,
    tools=tools,
    interrupt_before=["tools"],  # Pause before tool execution
)
```

---

## References

- **Official Databricks Pattern**: [LangGraph MCP Tool Calling](https://docs.databricks.com/aws/en/notebooks/source/generative-ai/langgraph-mcp-tool-calling-agent.html)
- **LangGraph Docs**: [langgraph-ai.github.io](https://langchain-ai.github.io/langgraph/)
- **Databricks MCP Docs**: [MCP Managed Servers](https://docs.databricks.com/aws/en/generative-ai/mcp/managed-mcp.html)

---

## Summary

`langgraph_agent.py` implements a **robust, stateful AI agent** that:
1. ✅ Discovers tools from MCP servers dynamically
2. ✅ Uses LangGraph for reliable orchestration
3. ✅ Supports structured tool calling via OpenAI API
4. ✅ Prevents infinite loops with recursion limits
5. ✅ Provides detailed debug logging
6. ✅ Handles async MCP calls correctly

The agent follows a simple but powerful pattern:
**Query Data (Genie) → Transform Data (UC Function) → Summarize (LLM) → End**

