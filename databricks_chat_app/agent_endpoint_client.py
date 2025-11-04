"""
Client for calling Databricks Agent Endpoint via MLflow Deployments
Uses MLflow client which handles auth automatically via WorkspaceClient
"""
import json
from typing import Dict, Any
from mlflow.deployments import get_deploy_client


class AgentEndpointClient:
    """Client for calling Databricks Agent Endpoint via MLflow Deployments"""
    
    def __init__(self, agent_endpoint_name: str, workspace_client):
        """
        Initialize agent endpoint client
        
        Args:
            agent_endpoint_name: Name of the deployed agent endpoint
            workspace_client: Databricks WorkspaceClient instance (handles auth automatically)
        """
        self.agent_endpoint_name = agent_endpoint_name
        self.deploy_client = get_deploy_client("databricks")
        
        print(f"[AgentClient] Initialized with endpoint: {agent_endpoint_name}")
        
    def chat(self, user_message: str) -> Dict[str, Any]:
        """
        Send a message to the agent endpoint
        
        Args:
            user_message: User's question
            
        Returns:
            Dict with 'response', 'messages', 'charts', 'table_data', and 'error' keys
        """
        print(f"[AgentClient] Sending message to agent endpoint...")
        
        try:
            # Call agent endpoint with user message using MLflow Deployments Client
            # Agent endpoint expects "input" (array of messages), not "messages"
            response = self.deploy_client.predict(
                endpoint=self.agent_endpoint_name,
                inputs={
                    "input": [{"role": "user", "content": user_message}]
                }
            )
            
            print(f"[AgentClient] Received response from agent endpoint")
            print(f"[AgentClient] Raw response type: {type(response)}")
            print(f"[AgentClient] Raw response keys: {response.keys() if isinstance(response, dict) else 'Not a dict'}")
            if isinstance(response, dict):
                print(f"[AgentClient] Response sample: {str(response)[:500]}")
            
            # Parse agent response
            result = self._parse_agent_response(response)
            print(f"[AgentClient] Parsed response: {len(result.get('charts', []))} charts, table_data: {bool(result.get('table_data'))}")
            
            return result
            
        except Exception as e:
            error_msg = f"Error calling agent endpoint: {str(e)}"
            print(f"[AgentClient] ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "response": None,
                "messages": [],
                "charts": [],
                "table_data": None,
                "error": error_msg
            }
    
    def _parse_agent_response(self, raw_response: dict) -> Dict[str, Any]:
        """
        Parse agent endpoint response to extract summary, charts, and table data
        
        Agent response format: {"object": "response", "output": [...], "id": "..."}
        """
        output_array = raw_response.get("output", [])
        print(f"[AgentClient] Parsing {len(output_array)} output items...")
        
        return {
            "response": self._extract_summary(output_array),
            "messages": output_array,
            "charts": self._extract_charts(output_array),
            "table_data": self._extract_genie_table(output_array),
            "error": None
        }
    
    def _extract_summary(self, output_array: list) -> str:
        """Extract the final text summary from agent output"""
        for item in reversed(output_array):  # Start from end - summary is usually last
            if text := self._get_text_from_item(item):
                print(f"[AgentClient] Found summary: {text[:100]}...")
                return text
        
        return "Agent processed your request"
    
    def _get_text_from_item(self, item: dict) -> str:
        """Extract text from various item formats"""
        item_type = item.get("type")
        
        # Simple text type
        if item_type == "text":
            return item.get("text", "")
        
        # Message type with nested content
        if item_type == "message":
            content = item.get("content", "")
            
            # Content as string
            if isinstance(content, str):
                return content
            
            # Content as list of objects with 'text' field
            if isinstance(content, list) and content:
                first_item = content[0]
                if isinstance(first_item, dict):
                    return first_item.get("text", "")
        
        return ""
    
    def _extract_charts(self, output_array: list) -> list:
        """Extract all Plotly charts from function call outputs"""
        charts = []
        
        for item in output_array:
            if item.get("type") != "function_call_output":
                continue
            
            if chart := self._parse_chart_output(item.get("output", "")):
                charts.append(chart)
                print(f"[AgentClient] Found chart: {chart.get('chart_type', 'unknown')}")
        
        return charts
    
    def _parse_chart_output(self, output_str: str) -> dict:
        """Parse a single function output for chart data"""
        try:
            data = json.loads(output_str) if isinstance(output_str, str) else output_str
            
            if not isinstance(data, dict):
                return None
            
            # Direct chart format: {"plotly_json": {...}, "chart_type": "..."}
            if "plotly_json" in data:
                return data
            
            # UC function wrapper format: {"rows": [["{...}"]], "columns": [...]}
            if "rows" in data and "columns" in data:
                if data["rows"] and data["rows"][0]:
                    chart_json = json.loads(data["rows"][0][0])
                    if "plotly_json" in chart_json:
                        return chart_json
            
            return None
            
        except (json.JSONDecodeError, IndexError, KeyError, TypeError) as e:
            print(f"[AgentClient] Error parsing chart: {e}")
            return None
    
    def _extract_genie_table(self, output_array: list) -> dict:
        """Extract table data from Genie function call outputs"""
        for item in output_array:
            if item.get("type") != "function_call_output":
                continue
            
            if table := self._parse_genie_output(item.get("output", "")):
                print(f"[AgentClient] Found Genie table: {len(table['data'])} rows")
                return table
        
        return None
    
    def _parse_genie_output(self, output_str: str) -> dict:
        """Parse Genie's complex response into simple table format"""
        try:
            # Parse outer JSON
            data = json.loads(output_str) if isinstance(output_str, str) else output_str
            
            if not isinstance(data, dict) or "content" not in data:
                return None
            
            # Parse nested content
            content = data["content"]
            content_data = json.loads(content) if isinstance(content, str) else content
            
            # Check for Genie statement_response
            if "statement_response" not in content_data:
                return None
            
            # Transform complex format to simple format
            return self._transform_genie_response(content_data)
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"[AgentClient] Error parsing Genie output: {e}")
            return None
    
    def _transform_genie_response(self, genie_response: dict) -> dict:
        """
        Transform Genie's nested response to simple table format
        
        FROM: {"statement_response": {"result": {"data_array": [...]}, "manifest": {...}}}
        TO:   {"data": [[...]], "columns": [...]}
        """
        try:
            stmt_resp = genie_response["statement_response"]
            result = stmt_resp["result"]
            schema = stmt_resp["manifest"]["schema"]
            
            # Extract columns
            columns = [col["name"] for col in schema["columns"]]
            
            # Extract data rows
            data = [
                [val["string_value"] for val in row["values"]]
                for row in result["data_array"]
            ]
            
            return {"data": data, "columns": columns}
            
        except (KeyError, TypeError) as e:
            print(f"[AgentClient] Error transforming Genie response: {e}")
            return None

