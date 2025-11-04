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
from databricks_chat_app.agent_endpoint_client import AgentEndpointClient

# Load environment variables
dotenv_path = os.path.join(PROJECT_ROOT, '.env')
load_dotenv(dotenv_path=dotenv_path)

# Get agent endpoint name from environment
AGENT_ENDPOINT_NAME = os.getenv("AGENT_ENDPOINT_NAME", "agents_bo_cheng_dnb_demos-agents-managed-mcp-model")

# Initialize Databricks workspace client
# This handles both user identity (Databricks Apps) and PAT (local dev)
try:
    workspace_client = WorkspaceClient()
    
    # Initialize agent endpoint client
    agent = AgentEndpointClient(
        agent_endpoint_name=AGENT_ENDPOINT_NAME,
        workspace_client=workspace_client
    )
    
    client_initialized = True
    print(f"✅ Connected to agent endpoint: {AGENT_ENDPOINT_NAME}")
    
except Exception as e:
    print(f"❌ Error initializing agent client: {e}")
    import traceback
    traceback.print_exc()
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
    <h1>Chat with your Agent</h1>
</div>
''', unsafe_allow_html=True)

if not client_initialized:
    st.error("Agent endpoint client initialization failed. Check your environment variables and ensure the agent endpoint is deployed.")
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
            thinking_placeholder.markdown("🤖 Agent endpoint is processing...")
            
            # Call agent endpoint
            agent_response = agent.chat(prompt)
            
            if agent_response.get("error"):
                response_text = f"Error: {agent_response['error']}"
                table_data = None
                charts = None
            else:
                response_text = agent_response.get("response", "Agent processed your request")
                
                # Use already-parsed data from agent endpoint client
                table_data = agent_response.get("table_data")
                charts = agent_response.get("charts") or []
            
            thinking_placeholder.empty()
            
            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "table_data": table_data,
                "charts": charts if charts else None
            })
            
            st.rerun()

