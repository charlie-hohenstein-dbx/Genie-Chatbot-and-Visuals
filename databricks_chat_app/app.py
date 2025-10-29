import sys
import os
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import plotly.io as pio
from dotenv import load_dotenv
from databricks.sdk import WorkspaceClient
from databricks_chat_app.langgraph_agent import LangGraphMCPAgent

# Load environment variables
dotenv_path = os.path.join(PROJECT_ROOT, '.env')
load_dotenv(dotenv_path=dotenv_path)

DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
GENIE_SPACE_ID = os.getenv("GENIE_SPACE_ID")

# Initialize agent
try:
    workspace_client = WorkspaceClient(
        host=DATABRICKS_HOST,
        token=DATABRICKS_TOKEN,
        auth_type='pat'
    )
    # Initialize LangGraph agent with LLM endpoint
    # Using Claude 3.7 Sonnet which has excellent tool calling capabilities
    agent = LangGraphMCPAgent(
        workspace_client=workspace_client, 
        genie_space_id=GENIE_SPACE_ID,
        llm_endpoint="databricks-claude-3-7-sonnet"
    )
    client_initialized = True
except Exception as e:
    print(f"Error initializing agent: {e}")
    client_initialized = False
    agent = None

# Streamlit Page Config - Force light theme
st.set_page_config(
    page_title="Agent Chat", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS - Optimized for light theme
st.markdown("""
<style>
    /* Light, modern background with subtle gradient */
    .stApp {
        background: linear-gradient(135deg, #f0f4f8 0%, #e8f0fe 100%) !important;
    }
    
    /* Modern header with gradient */
    .main-header-wrapper {
        padding: 2rem;
        border-bottom: 3px solid #e0e7ff;
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        box-shadow: 0 2px 12px rgba(0,0,0,0.05);
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
        margin-bottom: 2rem;
    }
    .main-header-wrapper h1 {
        background: linear-gradient(90deg, #1e40af, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.25rem;
        font-weight: 800;
        margin: 0;
    }
    
    /* Chat message styling - elegant cards */
    [data-testid="stChatMessage"] {
        border-radius: 18px;
        padding: 20px 24px;
        margin-bottom: 16px;
        max-width: 75%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    /* User messages - soft blue gradient */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        margin-left: auto;
        border: 1px solid #2563eb;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25);
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) p,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) div,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) span,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) * {
        color: white !important;
    }
    
    /* Assistant messages - clean white with subtle border */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background-color: #ffffff !important;
        border-left: 4px solid #8b5cf6 !important;
        box-shadow: 0 2px 10px rgba(139, 92, 246, 0.1);
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) p,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) div,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) span {
        color: #1e293b !important;
    }
    
    /* Chart expander styling - LIGHT BACKGROUND */
    [data-testid="stExpander"] {
        background-color: #ffffff !important;
        border: 2px solid #e5e7eb !important;
        border-radius: 12px !important;
    }
    
    [data-testid="stExpander"] summary {
        background-color: #f8fafc !important;
        border-radius: 10px !important;
        padding: 14px 18px !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        color: #000000 !important;
    }
    
    [data-testid="stExpander"] summary:hover {
        background-color: #e5e7eb !important;
    }
    
    [data-testid="stExpander"] div[role="button"] {
        background-color: transparent !important;
    }
    
    [data-testid="stExpander"] div[role="button"] p {
        color: #000000 !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stExpander"] svg {
        fill: #000000 !important;
    }
    
    .streamlit-expanderContent {
        background-color: #ffffff !important;
        padding: 20px !important;
    }
    
    /* Button styling - gradient */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 14px 28px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
        transform: translateY(-2px);
    }
    
    /* Error message styling */
    .stAlert {
        background-color: #fef2f2;
        color: #991b1b;
        border: 2px solid #fee2e2;
        border-radius: 12px;
        padding: 16px;
    }
    
    /* Ensure all text is readable with high contrast */
    p, div, span, h1, h2, h3, h4, h5, h6 {
        color: #1e293b !important;
    }
    
    /* Code blocks */
    code {
        background-color: #f1f5f9 !important;
        color: #475569 !important;
        padding: 0.25rem 0.5rem !important;
        border-radius: 6px !important;
        font-size: 0.9em !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('''
<div class="main-header-wrapper">
    <h1>Genie Chat with MCP</h1>
</div>
''', unsafe_allow_html=True)

if not client_initialized:
    st.error("Agent initialization failed. Check your environment variables and model serving endpoint access.")
else:
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hi there! Ask me anything about your data to get started.",
            "table_data": None,
            "charts": None
        })

    # Display message history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display table data if present
            if message.get("table_data"):
                table_data = message["table_data"]
                with st.expander("Table Data", expanded=True):
                    df = pd.DataFrame(table_data["data"], columns=table_data["columns"])
                    st.dataframe(df)
            
            # Display charts if present
            if message.get("charts"):
                for idx, chart_info in enumerate(message["charts"]):
                    chart_type = chart_info.get("chart_type", "unknown")
                    plotly_json = chart_info.get("plotly_json")
                    
                    if plotly_json:
                        # Generate unique key for this chart
                        msg_idx = st.session_state.messages.index(message)
                        chart_key = f"chart_{msg_idx}_{idx}"
                        
                        with st.expander(f"{chart_type.capitalize()} Chart", expanded=True):
                            # Use plotly.io.from_json with skip_invalid to handle version incompatibilities
                            fig_json_str = json.dumps(plotly_json)
                            fig = pio.from_json(fig_json_str, skip_invalid=True)
                            
                            # Update layout for light theme with HIGH CONTRAST
                            fig.update_layout(
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                font=dict(color='#000000', size=14, family='Arial, sans-serif'),
                                title_font=dict(color='#000000', size=18, family='Arial, sans-serif', weight='bold'),
                                xaxis=dict(
                                    gridcolor='#d1d5db',
                                    zerolinecolor='#9ca3af',
                                    color='#000000',
                                    title_font=dict(size=14, color='#000000', weight='bold'),
                                    tickfont=dict(size=12, color='#000000')
                                ),
                                yaxis=dict(
                                    gridcolor='#d1d5db',
                                    zerolinecolor='#9ca3af',
                                    color='#000000',
                                    title_font=dict(size=14, color='#000000', weight='bold'),
                                    tickfont=dict(size=12, color='#000000')
                                )
                            )
                            
                            st.plotly_chart(fig, use_container_width=True, key=chart_key)

    # Chat input
    if prompt := st.chat_input("What is your question?"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt, "table_data": None, "charts": None})
        
        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown("🤖 LangGraph agent is thinking...")
            
            # Call agent
            agent_response = agent.chat(prompt)
            
            if agent_response.get("error"):
                response_text = f"Error: {agent_response['error']}"
                table_data = None
                charts = None
            else:
                response_text = agent_response.get("response", "Agent processed your request")
                
                # Extract table data and charts from messages
                table_data = None
                charts = []
                
                # Look through all messages for tool results
                for msg in agent_response.get("messages", []):
                    if hasattr(msg, 'content') and isinstance(msg.content, str):
                        # Try to parse message content as JSON
                        try:
                            result_data = json.loads(msg.content)
                            
                            # Check if this is table data from Genie
                            if isinstance(result_data, dict) and "columns" in result_data and "data" in result_data:
                                table_data = result_data
                            
                            # Check if this is a Databricks UC function result (has rows/columns wrapper)
                            if isinstance(result_data, dict) and "rows" in result_data and "columns" in result_data:
                                # Extract the actual result from rows[0][0]
                                if result_data["rows"] and result_data["rows"][0]:
                                    chart_json_str = result_data["rows"][0][0]
                                    # Parse the nested JSON string
                                    chart_data = json.loads(chart_json_str)
                                    # Check if this is chart data
                                    if isinstance(chart_data, dict) and "plotly_json" in chart_data:
                                        charts.append(chart_data)
                            
                            # Check if this is a chart from UC function (direct format)
                            elif isinstance(result_data, dict) and "plotly_json" in result_data:
                                charts.append(result_data)
                        except Exception as e:
                            print(f"[DEBUG] Error parsing message: {e}")
                            pass
            
            thinking_placeholder.empty()
            
            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "table_data": table_data,
                "charts": charts if charts else None
            })
            
            st.rerun()

