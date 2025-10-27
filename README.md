# Genie Chatbot with MCP

A simple Streamlit chatbot that uses **Databricks MCP servers** to query data via Genie and generate interactive charts via Unity Catalog functions.

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables
Create a `.env` file:
```bash
DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
DATABRICKS_TOKEN="dapi..."
GENIE_SPACE_ID="01abc..."
```

### 3. Deploy UC Function
```bash
cd uc_functions/deploy
python create_chart_function_client.py
```
This creates `chat_app_demo.dev.generate_chart` in Unity Catalog.

### 4. Run Locally
```bash
cd databricks_chat_app
streamlit run app.py
```

---

## How It Works

```
User asks: "Show me sales by region"
    ↓
[Genie MCP] → Returns table data
    ↓
[UC Function MCP] → Generates interactive Plotly charts (bar, line, pie)
    ↓
Streamlit displays results
```

**MCP Servers Used:**
- **Genie**: `{host}/api/2.0/mcp/genie/{space_id}`
- **UC Functions**: `{host}/api/2.0/mcp/functions/chat_app_demo/dev`

---

## Deploy to Databricks

### Using Databricks Asset Bundles (DABs)

1. **Configure secrets** (replace `your-scope`):
   ```bash
   databricks secrets create-scope your-scope
   databricks secrets put-secret your-scope databricks-token
   databricks secrets put-secret your-scope genie-space-id
   ```

2. **Update `databricks.yml`** - change `your-scope` to your actual scope name

3. **Deploy**:
   ```bash
   databricks bundle validate
   databricks bundle deploy -t dev
   ```

4. **Access** your app in the workspace under **Apps** → `genie-chatbot-mcp`

---

## Project Structure

```
Genie-Chatbot-and-Visuals/
├── databricks.yml               # DABs deployment config
├── requirements.txt             # Dependencies
├── databricks_chat_app/
│   ├── app.py                  # Main Streamlit app (MCP-based)
│   ├── mcp_client.py           # MCP wrapper
│   ├── genie_chat.py           # CLI tool
│   └── app_old_sdk_version.py # Backup (original SDK version)
└── uc_functions/
    └── deploy/
        └── create_chart_function_client.py  # Deploy UC function
```

---

## What's Different (vs Old Version)

| Aspect | Old (SDK) | New (MCP) |
|--------|-----------|-----------|
| **Code** | 825 lines | 184 lines (78% less!) |
| **Genie** | Direct SDK | MCP server |
| **Charts** | Local matplotlib | UC function via MCP |
| **Chart Type** | Static images | Interactive Plotly |
| **Agent Ready** | ❌ | ✅ |
| **Layout** | Chat + Dashboard | Chat only (clean & focused) |

---

## Key Features

✅ **Ultra Simple**: 184 lines of clean, focused code  
✅ **Agent-Ready**: Uses MCP protocol for AI frameworks  
✅ **Interactive Charts**: Zoomable, hoverable Plotly visualizations  
✅ **Governed**: Charts generated via Unity Catalog functions with audit trails  
✅ **Flexible**: Easy to add more MCP servers (Vector Search, DBSQL, etc.)  
✅ **Clean UI**: Full-width chat interface without distractions

---

## Troubleshooting

**"No Genie tools available"**  
→ Check your `GENIE_SPACE_ID` and workspace access

**"Tool 'chat_app_demo__dev__generate_chart' not found"**  
→ Deploy the UC function: `python uc_functions/deploy/create_chart_function_client.py`

**"MCP server connection failed"**  
→ Verify authentication: `databricks auth login --host https://your-workspace.cloud.databricks.com`

---

## Architecture

The app uses two MCP servers that discover and call tools dynamically:

```python
# Genie MCP - natural language queries
genie_mcp = DatabricksMCPClient(
    server_url=f"{host}/api/2.0/mcp/genie/{genie_space_id}",
    workspace_client=workspace_client
)

# UC Function MCP - chart generation
chart_mcp = DatabricksMCPClient(
    server_url=f"{host}/api/2.0/mcp/functions/chat_app_demo/dev",
    workspace_client=workspace_client
)

# Discover and call tools
tools = mcp_client.list_tools()
result = mcp_client.call_tool(tool_name, parameters)
```

**Tool Naming Convention:**  
UC function `catalog.schema.function` → MCP tool `catalog__schema__function`

---

## Next Steps

- 🔗 Connect to additional MCP servers (Vector Search, DBSQL)
- 🤖 Integrate with AI frameworks (LangChain, OpenAI, CrewAI)
- 📊 Add more chart types to the UC function
- 🚀 Deploy to production with `databricks bundle deploy -t prod`

---

## Documentation

- **MCP Servers**: https://docs.databricks.com/aws/en/generative-ai/mcp/managed-mcp
- **Unity Catalog AI**: https://docs.unitycatalog.io/ai/client/
- **Databricks Apps**: https://docs.databricks.com/en/dev-tools/databricks-apps/
- **DABs**: https://docs.databricks.com/dev-tools/bundles/
