-- ============================================================================
-- UC Function: generate_chart (UNIFIED)
-- Description: Single function to generate any type of chart visualization
-- ============================================================================
-- 
-- Catalog: chat_app_demo
-- Schema: dev
--
-- This unified function replaces the 4 individual chart functions with a 
-- single, more agent-friendly interface.
--
-- Run this in:
--   - Databricks SQL Editor
--   - Databricks Notebook (with %sql)
--   - Via CLI: databricks sql execute -f create_unified_chart_function.sql
-- ============================================================================

CREATE OR REPLACE FUNCTION chat_app_demo.dev.generate_chart(
    table_data_json STRING,
    chart_type STRING
)
RETURNS STRING
LANGUAGE PYTHON
COMMENT 'Universal chart generation function. Creates interactive Plotly visualizations from tabular data. 
PARAMETERS:
  - table_data_json: JSON with "columns" (array of column names) and "data" (array of row arrays)
  - chart_type: Type of chart to generate - "bar", "pie", or "line"

CHART TYPES:
  - "bar": Compare values across categories, show distributions. Best for comparing quantities, categorical data.
  - "pie": Show proportions and percentages. Best for part-to-whole relationships.
  - "line": Visualize trends over time or sequences. Best for time series, progression tracking.

RETURNS: Plotly JSON string that can be directly rendered with Plotly.js using Plotly.newPlot()'
AS $$
import plotly.express as px
import pandas as pd
import json

def generate_chart(table_data_json, chart_type):
    """Simplified chart generation using Plotly Express"""
    # Parse input
    table_data = json.loads(table_data_json)
    df = pd.DataFrame(table_data.get("data"), columns=table_data.get("columns"))
    
    # Auto-detect column types
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')
    
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    string_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    
    # Determine x and y
    x_col = string_cols[0] if string_cols else None
    y_cols = numeric_cols
    
    # Generate chart
    chart_type = chart_type.lower().strip()
    
    if chart_type == "bar":
        fig = px.bar(df, x=x_col, y=y_cols, title="Bar Chart")
    elif chart_type == "line":
        fig = px.line(df, x=x_col, y=y_cols, title="Line Chart", markers=True)
    elif chart_type == "pie":
        fig = px.pie(df, names=string_cols[0], values=numeric_cols[0], title="Pie Chart")
    
    return str(fig.to_json())

return generate_chart(table_data_json, chart_type)
$$;

-- ============================================================================
-- TEST THE FUNCTION
-- ============================================================================

-- Test bar chart
-- SELECT chat_app_demo.dev.generate_chart(
--   '{"columns": ["Region", "Sales"], "data": [["North", 100], ["South", 150], ["East", 120]]}',
--   'bar'
-- );

-- Test pie chart
-- SELECT chat_app_demo.dev.generate_chart(
--   '{"columns": ["Region", "Sales"], "data": [["North", 100], ["South", 150], ["East", 120]]}',
--   'pie'
-- );

-- Test line chart
-- SELECT chat_app_demo.dev.generate_chart(
--   '{"columns": ["Month", "Sales"], "data": [["Jan", 100], ["Feb", 150], ["Mar", 120]]}',
--   'line'
-- );

-- ============================================================================

