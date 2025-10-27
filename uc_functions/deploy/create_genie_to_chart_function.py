"""
Create the genie_to_chart UC Function using the Databricks Function Client
This function transforms Genie MCP responses into Plotly charts
"""

from unitycatalog.ai.core.databricks import DatabricksFunctionClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the Databricks client
client = DatabricksFunctionClient()

# Define the function
def genie_to_chart(genie_response_json: str, chart_type: str) -> str:
    """
    Transform Genie MCP response into a Plotly chart.
    This function handles all data extraction and transformation from Genie's nested JSON format.
    
    Args:
        genie_response_json (str): Raw JSON string from Genie MCP query_space tool
        chart_type (str): Type of chart - "bar", "line", or "pie"
    
    Chart Types:
        - "bar": Compare values across categories. Best for comparing quantities.
        - "line": Visualize trends over time. Best for time series data.
        - "pie": Show proportions and percentages. Best for part-to-whole relationships.
    
    Returns:
        str: Plotly JSON string for direct rendering with Plotly.js
    """
    import json
    import plotly.express as px
    import pandas as pd
    
    # Parse Genie response - handle both full wrapper and extracted content
    genie_data = json.loads(genie_response_json)
    
    # If there's a 'content' field, parse it (double-parse scenario)
    if 'content' in genie_data:
        content_data = json.loads(genie_data['content']) if isinstance(genie_data['content'], str) else genie_data['content']
    else:
        # Already the content (no wrapper)
        content_data = genie_data
    
    # Extract columns and rows from statement_response
    columns_info = content_data['statement_response']['manifest']['schema']['columns']
    column_names = [col['name'] for col in columns_info]
    
    data_array = content_data['statement_response']['result']['data_array']
    rows = [[value['string_value'] for value in row['values']] for row in data_array]
    
    # Create DataFrame
    df = pd.DataFrame(rows, columns=column_names)
    
    # Convert all columns to appropriate types
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')
    
    # Column selection: string columns for X, last numeric for Y
    string_cols = df.select_dtypes(include=['object', 'string']).columns
    numeric_cols = df.select_dtypes(include='number').columns
    
    x_col = string_cols[0] if len(string_cols) > 0 else df.columns[0]
    y_col = numeric_cols[-1] if len(numeric_cols) > 0 else df.columns[-1]
    
    # Generate chart
    chart_functions = {
        "bar": lambda: px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}"),
        "line": lambda: px.line(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}", markers=True),
        "pie": lambda: px.pie(df, names=x_col, values=y_col, title=f"{y_col} by {x_col}")
    }
    
    chart_type_key = chart_type.lower().strip()
    if chart_type_key not in chart_functions:
        raise ValueError(f"Unsupported chart type: {chart_type}")
    
    fig = chart_functions[chart_type_key]()
    
    # Return structured response with Plotly JSON
    import json as json_lib
    response = {
        "plotly_json": json_lib.loads(fig.to_json()),
        "chart_type": chart_type_key
    }
    return json_lib.dumps(response)

# Example genie_response_json format:
# {
#   "content": "{\"query\":\"SELECT YEAR(`tpep_pickup_datetime`) AS year, COUNT(*) AS trip_count\\nFROM `samples`.`nyctaxi`.`trips`\\nWHERE `tpep_pickup_datetime` IS NOT NULL\\nGROUP BY YEAR(`tpep_pickup_datetime`)\\nORDER BY year\",\"statement_response\":{\"statement_id\":\"01f0aeff-0b10-1aa4-b756-53fa9090a939\",\"status\":{\"state\":\"SUCCEEDED\"},\"manifest\":{\"format\":\"JSON_ARRAY\",\"schema\":{\"column_count\":2,\"columns\":[{\"name\":\"year\",\"type_text\":\"INT\",\"type_name\":\"INT\",\"position\":0},{\"name\":\"trip_count\",\"type_text\":\"BIGINT\",\"type_name\":\"LONG\",\"position\":1}]}},\"result\":{\"data_array\":[{\"values\":[{\"string_value\":\"2016\"},{\"string_value\":\"21932\"}]}]}}}",
#   "conversationId": "01f0aefcf2911c0e9e457bbd46321cce",
#   "messageId": "01f0aefcf299138fa9657416761146f1"
# }
#
# Note: The 'content' field is a JSON STRING (not object) that needs to be parsed again.
# After parsing content, the structure is:
# {
#   "query": "SELECT ...",
#   "statement_response": {
#     "manifest": {
#       "schema": {
#         "columns": [{"name": "year", ...}, {"name": "trip_count", ...}]
#       }
#     },
#     "result": {
#       "data_array": [
#         {"values": [{"string_value": "2016"}, {"string_value": "21932"}]}
#       ]
#     }
#   }
# }


def main():
    """Create the function in Unity Catalog"""
    print("=" * 70)
    print("Creating UC Function: genie_to_chart")
    print("=" * 70)
    
    catalog = os.getenv("UC_CATALOG", "chat_app_demo")
    schema = os.getenv("UC_SCHEMA", "dev")
    
    print(f"\nTarget: {catalog}.{schema}")
    print(f"Function: genie_to_chart")
    print("\nThis function:")
    print("  - Takes raw Genie MCP JSON response")
    print("  - Extracts and transforms data automatically")
    print("  - Returns Plotly chart JSON")
    print("  - No manual JSON parsing required by agent!")
    
    try:
        function_info = client.create_python_function(
            func=genie_to_chart,
            catalog=catalog,
            schema=schema,
            replace=True
        )
        
        print(f"\n{'='*70}")
        print("✅ SUCCESS!")
        
    except Exception as e:
        print(f"\n{'='*70}")
        print("❌ ERROR")
        print(f"{'='*70}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
