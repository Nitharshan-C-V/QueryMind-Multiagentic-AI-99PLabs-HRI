import os
import streamlit as st
import pandas as pd
import pickle
import base64
from io import BytesIO, StringIO
import sys
import operator
from typing import Literal, Sequence, TypedDict, Annotated, List, Dict, Tuple
import tempfile
import shutil
import plotly.io as pio
import io
import re
import json
import openai
# from fpdf import FPDF
import base64
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from PIL import Image as PILImage

# Import LangChain and LangGraph components
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_experimental.utilities import PythonREPL
from langgraph.prebuilt import ToolInvocation, ToolExecutor
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langgraph.graph import StateGraph, END
from reportlab.platypus import PageBreak


# Initialize session state for AI provider settings
if 'ai_provider' not in st.session_state:
    st.session_state.ai_provider = "openai"
    
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
    
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "gpt-4"
    
# Define model options for each provider
OPENAI_MODELS = ["gpt-4", "gpt-4-turbo", "gpt-4-mini", "gpt-3.5-turbo"]
GROQ_MODELS = ["llama3.3-70b-versatile", "gemma2-9b-it", "llama-3-8b-8192"]

# Create temporary directory for file storage
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()
    st.session_state.images_dir = os.path.join(st.session_state.temp_dir, "images/plotly_figures/pickle")
    os.makedirs(st.session_state.images_dir, exist_ok=True)
    print(f"Created temporary directory: {st.session_state.temp_dir}")
    print(f"Created images directory: {st.session_state.images_dir}")

# Define the system prompt
SYSTEM_PROMPT = """## Role
You are a professional data scientist helping a non-technical user understand, analyze, and visualize their data.

## Capabilities
1. **Execute python code** using the `complete_python_task` tool.

## Goals
1. Understand the user's objectives clearly.
2. Take the user on a data analysis journey, iterating to find the best way to visualize or analyse their data to solve their problems.
3. Investigate if the goal is achievable by running Python code via the `python_code` field.
4. Gain input from the user at every step to ensure the analysis is on the right track and to understand business nuances.

## Code Guidelines
- **ALL INPUT DATA IS LOADED ALREADY**, so use the provided variable names to access the data.
- **VARIABLES PERSIST BETWEEN RUNS**, so reuse previously defined variables if needed.
- **TO SEE CODE OUTPUT**, use `print()` statements. You won't be able to see outputs of `pd.head()`, `pd.describe()` etc. otherwise.
- **ONLY USE THE FOLLOWING LIBRARIES**:
  - `pandas`
  - `sklearn` (including all major ML models)
  - `plotly`
  - `numpy`
  
All these libraries are already imported for you.

## Machine Learning Guidelines
- For regression tasks:
  - Linear Regression: `LinearRegression`
  - Logistic Regression: `LogisticRegression`
  - Ridge Regression: `Ridge`
  - Lasso Regression: `Lasso`
  - Random Forest Regression: `RandomForestRegressor`
  
- For classification tasks:
  - Logistic Regression: `LogisticRegression`
  - Decision Trees: `DecisionTreeClassifier`
  - Random Forests: `RandomForestClassifier`
  - Support Vector Machines: `SVC`
  - K-Nearest Neighbors: `KNeighborsClassifier`
  - Naive Bayes: `GaussianNB`
  
- For clustering:
  - K-Means: `KMeans`
  - DBSCAN: `DBSCAN`
  
- For dimensionality reduction:
  - PCA: `PCA`
  
- Always preprocess data appropriately:
  - Scale numerical features with `StandardScaler` or `MinMaxScaler`
  - Encode categorical variables with `OneHotEncoder` when needed
  - Handle missing values with `SimpleImputer`
  
- Always split data into training and testing sets using `train_test_split`
- Evaluate models using appropriate metrics:
  - For regression: `mean_squared_error`, `mean_absolute_error`, `r2_score`
  - For classification: `accuracy_score`, `confusion_matrix`, `classification_report`
  - For clustering: `silhouette_score`
  
- Consider using `cross_val_score` for more robust evaluation
- Visualize ML results with plotly when possible

## Plotting Guidelines
- Always use the `plotly` library for plotting.
- Store all plotly figures inside a `plotly_figures` list, they will be saved automatically.
- Do not try and show the plots inline with `fig.show()`.
"""

# Define the State class
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    input_data: Annotated[List[Dict], operator.add]
    intermediate_outputs: Annotated[List[dict], operator.add]
    current_variables: dict
    output_image_paths: Annotated[List[str], operator.add]

# Initialize session state variables
if 'in_memory_datasets' not in st.session_state:
    st.session_state.in_memory_datasets = {}

if 'persistent_vars' not in st.session_state:
    st.session_state.persistent_vars = {}

if 'dataset_metadata_list' not in st.session_state:
    st.session_state.dataset_metadata_list = []

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'dashboard_plots' not in st.session_state:
    st.session_state.dashboard_plots = [None, None, None, None]

if 'columns' not in st.session_state:
    st.session_state.columns = ["No columns available"]

if 'custom_plots_to_save' not in st.session_state:
    st.session_state.custom_plots_to_save = {}

# Set up the tools
repl = PythonREPL()
plotly_saving_code = """import pickle

import uuid
import os
for figure in plotly_figures:
    pickle_filename = f"{images_dir}/{uuid.uuid4()}.pickle"
    with open(pickle_filename, 'wb') as f:
        pickle.dump(figure, f)
"""

@tool
def complete_python_task(
    graph_state: Annotated[dict, InjectedState],
    thought: str,
    python_code: str
) -> Tuple[str, dict]:
    """Execute Python code for data analysis and visualization."""

    current_variables = graph_state.get("current_variables", {})

    # Load datasets from in-memory storage
    for input_dataset in graph_state.get("input_data", []):
        var_name = input_dataset.get("variable_name")
        if var_name and var_name not in current_variables and var_name in st.session_state.in_memory_datasets:
            print(f"Loading {var_name} from in-memory storage")
            current_variables[var_name] = st.session_state.in_memory_datasets[var_name]
    current_image_pickle_files = os.listdir(st.session_state.images_dir)

    try:
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = StringIO()
       
        # Execute the code and capture the result
        exec_globals = globals().copy()
        exec_globals.update(st.session_state.persistent_vars)
        exec_globals.update(current_variables)
        
        # Add scikit-learn modules to execution environment
        import sklearn
        import numpy as np
        
        # Import scikit-learn components
        from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso # type: ignore
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier # type: ignore
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        from sklearn.svm import SVC, SVR
        from sklearn.naive_bayes import GaussianNB
        from sklearn.decomposition import PCA
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        from sklearn.cluster import KMeans, DBSCAN
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
        from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
        from sklearn.metrics import (
            accuracy_score, confusion_matrix, classification_report,
            mean_squared_error, r2_score, mean_absolute_error, silhouette_score
        )
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        
        # Update execution globals with all ML components
        exec_globals.update({
            "plotly_figures": [], 
            "images_dir": st.session_state.images_dir,
            "np": np,
            # Linear models
            "LinearRegression": LinearRegression,
            "LogisticRegression": LogisticRegression,
            "Ridge": Ridge,
            "Lasso": Lasso,
            # Tree-based models
            "DecisionTreeClassifier": DecisionTreeClassifier,
            "DecisionTreeRegressor": DecisionTreeRegressor,
            "RandomForestClassifier": RandomForestClassifier,
            "RandomForestRegressor": RandomForestRegressor,
            "GradientBoostingClassifier": GradientBoostingClassifier,
            # SVM models
            "SVC": SVC,
            "SVR": SVR,
            # Other models
            "GaussianNB": GaussianNB,
            "PCA": PCA,
            "KNeighborsClassifier": KNeighborsClassifier,
            "KNeighborsRegressor": KNeighborsRegressor,
            "KMeans": KMeans,
            "DBSCAN": DBSCAN,
            # Preprocessing
            "StandardScaler": StandardScaler,
            "MinMaxScaler": MinMaxScaler,
            "OneHotEncoder": OneHotEncoder,
            "SimpleImputer": SimpleImputer,
            # Model selection and evaluation
            "train_test_split": train_test_split,
            "cross_val_score": cross_val_score,
            "GridSearchCV": GridSearchCV,
            "accuracy_score": accuracy_score,
            "confusion_matrix": confusion_matrix,
            "classification_report": classification_report,
            "mean_squared_error": mean_squared_error,
            "r2_score": r2_score,
            "mean_absolute_error": mean_absolute_error,
            "silhouette_score": silhouette_score,
            # Pipeline
            "Pipeline": Pipeline
        })
        
        exec(python_code, exec_globals)

        st.session_state.persistent_vars.update({k: v for k, v in exec_globals.items() if k not in globals()})

        # Get the captured stdout
        output = sys.stdout.getvalue()

        # Restore stdout
        sys.stdout = old_stdout

        updated_state = {
            "intermediate_outputs": [{"thought": thought, "code": python_code, "output": output}],
            "current_variables": st.session_state.persistent_vars
        }

        if 'plotly_figures' in exec_globals and exec_globals['plotly_figures']:
            exec(plotly_saving_code, exec_globals)
           
            # Check if any images were created
            new_image_folder_contents = os.listdir(st.session_state.images_dir)
            new_image_files = [file for file in new_image_folder_contents if file not in current_image_pickle_files]
           
            if new_image_files:
                updated_state["output_image_paths"] = new_image_files
            st.session_state.persistent_vars["plotly_figures"] = []
        return output, updated_state

    except Exception as e:
        sys.stdout = old_stdout  # Restore stdout in case of error
        print(f"Error in complete_python_task: {str(e)}")
        return str(e), {"intermediate_outputs": [{"thought": thought, "code": python_code, "output": str(e)}]}

# Function to initialize the LLM based on selected provider and model
def initialize_llm():
    api_key = st.session_state.api_key
    model = st.session_state.selected_model
    
    if not api_key:
        return None
    
    try:
        if st.session_state.ai_provider == "openai":
            os.environ["OPENAI_API_KEY"] = api_key
            return ChatOpenAI(model=model, temperature=0)
        elif st.session_state.ai_provider == "groq":
            os.environ["GROQ_API_KEY"] = api_key
            # For Groq, set the base URL and use the model
            from langchain_groq import ChatGroq
            return ChatGroq(model=model, temperature=0)
    except Exception as e:
        print(f"Error initializing LLM: {str(e)}")
        return None

# Set up the tools
tools = [complete_python_task]
tool_executor = ToolExecutor(tools)

# Load the prompt template
chat_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("placeholder", "{messages}"),
])

def create_data_summary(state: AgentState) -> str:
    summary = ""
    variables = []
   
    # Add sample data for each dataset
    for d in state.get("input_data", []):
        var_name = d.get("variable_name")
        if var_name:
           
            variables.append(var_name)
            summary += f"\n\nVariable: {var_name}\n"
            summary += f"Description: {d.get('data_description', 'No description')}\n"

            # Add sample data if available
            if var_name in st.session_state.in_memory_datasets:
                df = st.session_state.in_memory_datasets[var_name]
                summary += "\nSample Data (first 5 rows):\n"
                summary += df.head(5).to_string()

    if "current_variables" in state:
        remaining_variables = [v for v in state["current_variables"] if v not in variables and not v.startswith("_")]
       
        for v in remaining_variables:
           
            var_value = state["current_variables"].get(v)

            if isinstance(var_value, pd.DataFrame):
                summary += f"\n\nVariable: {v} (DataFrame with shape {var_value.shape})"
            else:
                summary += f"\n\nVariable: {v}"
    return summary

def route_to_tools(state: AgentState) -> Literal["tools", "__end__"]:
    """Determine if we should route to tools or end the chain"""
    if messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
   
    return "__end__"

def call_model(state: AgentState):
    """Call the LLM to get a response"""
    current_data_template = """The following data is available:\n{data_summary}"""
    current_data_message = HumanMessage(
        content=current_data_template.format(data_summary=create_data_summary(state))
    )
    messages = [current_data_message] + state["messages"]
    
    # Get the initialized LLM
    llm = initialize_llm()
    if llm is None:
        return {"messages": [AIMessage(content="Please configure a valid API key and model in the settings tab.")]}
    
    # Create the model with bound tools
    model = llm.bind_tools(tools)
    model = chat_template | model
    
    llm_outputs = model.invoke({"messages": messages})
    return {"messages": [llm_outputs], "intermediate_outputs": [current_data_message.content]}

def call_tools(state: AgentState):
    """Execute tools called by the LLM"""
    last_message = state["messages"][-1]
    tool_invocations = []

    if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls'):
        tool_invocations = [
            ToolInvocation(
                tool=tool_call["name"],
                tool_input={**tool_call["args"], "graph_state": state}
            ) for tool_call in last_message.tool_calls
        ]
    responses = tool_executor.batch(tool_invocations, return_exceptions=True)

    tool_messages = []
    state_updates = {}

    for tc, response in zip(last_message.tool_calls, responses):
        if isinstance(response, Exception):
            print(f"Exception in tool execution: {str(response)}")
            tool_messages.append(ToolMessage(                
                content=f"Error: {str(response)}",
                name=tc["name"],
                tool_call_id=tc["id"]
            ))
            continue

        message, updates = response
        tool_messages.append(ToolMessage(
            content=str(message),
            name=tc["name"],
            tool_call_id=tc["id"]
        ))

        # Merge updates instead of overwriting
        for key, value in updates.items():
            if key in state_updates:
                if isinstance(value, list) and isinstance(state_updates[key], list):
                    state_updates[key].extend(value)
                elif isinstance(value, dict) and isinstance(state_updates[key], dict):
                    state_updates[key].update(value)
                else:
                    state_updates[key] = value
            else:
                state_updates[key] = value

    if 'messages' not in state_updates:
        state_updates["messages"] = []

    state_updates["messages"] = tool_messages
    return state_updates

# Set up the graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", call_tools)
workflow.add_conditional_edges(
    "agent",
    route_to_tools,
    {
        "tools": "tools",
        "__end__": END
    }
)
workflow.add_edge("tools", "agent")
workflow.set_entry_point("agent")

chain = workflow.compile()

def process_file_upload(files):
    """Process uploaded files and return dataframe previews and column names"""
    st.session_state.in_memory_datasets = {}  # Clear previous datasets
    st.session_state.dataset_metadata_list = []  # Clear previous metadata
    st.session_state.persistent_vars.clear()  # Clear persistent variables for new session

    if not files:
        return "No files uploaded.", [], ["No columns available"]

    results = []
    all_columns = []  # Track all columns from all datasets

    for file in files:
        try:
            # Use file object directly
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file)
            else:
                results.append(f"Unsupported file format: {file.name}. Please upload CSV or Excel files.")
                continue

            var_name = file.name.split('.')[0].replace('-', '_').replace(' ', '_').lower()
            st.session_state.in_memory_datasets[var_name] = df

            # Collect all columns
            all_columns.extend(df.columns.tolist())

            # Create dataset metadata
            dataset_metadata = {
                "variable_name": var_name,
                "data_path": "in_memory",
                "data_description": f"Dataset containing {df.shape[0]} rows and {df.shape[1]} columns. Columns: {', '.join(df.columns.tolist())}",
                "original_filename": file.name
            }

            st.session_state.dataset_metadata_list.append(dataset_metadata)

            # Return preview of the dataset
            preview = f"### Dataset: {file.name}\nVariable name: `{var_name}`\n\n"
            preview += df.head(10).to_markdown()
            results.append(preview)
            print(f"Successfully processed {file.name}")

        except Exception as e:
            print(f"Error processing {file.name}: {str(e)}")
            results.append(f"Error processing {file.name}: {str(e)}")

    # Get unique columns
    unique_columns = []
    seen = set()

    for col in all_columns:
        if col not in seen:
            seen.add(col)
            unique_columns.append(col)

    if not unique_columns:
        unique_columns = ["No columns available"]

    print(f"Found {len(unique_columns)} unique columns across datasets")
    return "\n\n".join(results), st.session_state.dataset_metadata_list, unique_columns

def get_columns():
    """Directly gets columns from in-memory datasets"""
    all_columns = []

    for var_name, df in st.session_state.in_memory_datasets.items():
        if isinstance(df, pd.DataFrame):
            all_columns.extend(df.columns.tolist())

    # Remove duplicates while preserving order
    unique_columns = []
    seen = set()
   
    for col in all_columns:
        if col not in seen:
            seen.add(col)
            unique_columns.append(col)

    if not unique_columns:
        unique_columns = ["No columns available"]

    print(f"Populating dropdowns with {len(unique_columns)} columns")
    return unique_columns

# === FUNCTIONS ===
import openai
import pandas as pd
import json
import re

def standard_clean(df):
    df.columns = [re.sub(r'\W+', '_', col).strip().lower() for col in df.columns]
    df.drop_duplicates(inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    df.dropna(axis=0, how='all', inplace=True)
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip()
    return df

def query_openai(prompt):
    try:
        # Use the configured API key and model from session state
        api_key = st.session_state.api_key
        model = st.session_state.selected_model
        
        if st.session_state.ai_provider == "openai":
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return response.choices[0].message.content
        elif st.session_state.ai_provider == "groq":
            from groq import Groq
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return response.choices[0].message.content
    except Exception as e:
        print(f"API Error: {e}")
        return "{}"

def llm_suggest_cleaning(df):
    sample = df.head(10).to_csv(index=False)
    prompt = f"""
You are a professional data wrangler. Below is a sample of a messy dataset.

Return a Python dictionary with the following keys:

1. rename_columns – fix unclear or inconsistent column names
2. convert_types – correct datatypes: int, float, str, or date
3. fill_missing – use 'mean', 'median', 'mode', or a constant like 'Unknown' or 0
4. value_map – map inconsistent values (e.g., yes/Yes/Y → Yes)

Do not drop any rows or columns. Your output must be a valid Python dict.

Example:
{{
  "rename_columns": {{"dob": "date_of_birth"}},
  "convert_types": {{"age": "int", "salary": "float", "signup_date": "date"}},
  "fill_missing": {{"gender": "mode", "salary": -1}},
  "value_map": {{
    "gender": {{"M": "Male", "F": "Female"}},
    "subscribed": {{"Y": "Yes", "N": "No"}}
  }}
}}
Apart from these mentioned steps, study the data and also do whatever things are good and needed for that particular dataset and do the cleaning.
Sample data:
{sample}
"""
    raw_response = query_openai(prompt)
    try:
        suggestions = eval(raw_response)
        return suggestions
    except:
        print("Could not parse suggestions.")
        return {
            "rename_columns": {},
            "convert_types": {},
            "fill_missing": {},
            "value_map": {}
        }

def apply_suggestions(df, suggestions):
    df.rename(columns=suggestions.get("rename_columns", {}), inplace=True)

    for col, dtype in suggestions.get("convert_types", {}).items():
        if col not in df.columns:
            continue
        try:
            if dtype == "int":
                df[col] = pd.to_numeric(df[col], errors='coerce').astype("Int64")
            elif dtype == "float":
                df[col] = pd.to_numeric(df[col], errors='coerce')
            elif dtype == "str":
                df[col] = df[col].astype(str)
            elif dtype == "date":
                df[col] = pd.to_datetime(df[col], errors='coerce')
        except:
            print(f"Failed to convert {col} to {dtype}")

    for col, method in suggestions.get("fill_missing", {}).items():
        if col not in df.columns:
            continue
        try:
            if method == "mean":
                df[col].fillna(df[col].mean(), inplace=True)
            elif method == "median":
                df[col].fillna(df[col].median(), inplace=True)
            elif method == "mode":
                df[col].fillna(df[col].mode().iloc[0], inplace=True)
            elif isinstance(method, str):
                df[col].fillna(method, inplace=True)
        except:
            print(f"Could not fill missing values for {col}")

    for col, mapping in suggestions.get("value_map", {}).items():
        if col in df.columns:
            try:
                df[col] = df[col].replace(mapping)
            except:
                print(f"Could not map values in {col}")

    return df

def capture_dashboard_screenshot():
    """Capture the entire dashboard as a single image"""
    try:
        # Create a figure that combines all dashboard plots
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
       
        # Create a 2x2 subplot
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=["Visualization 1", "Visualization 2",
                                          "Visualization 3", "Visualization 4"])

        # Add each plot from the dashboard to the combined figure
        for i, plot in enumerate(st.session_state.dashboard_plots):
            if plot is not None:
                row = (i // 2) + 1
                col = (i % 2) + 1

                # Extract traces from the original figure and add to our subplot
                for trace in plot.data:
                    fig.add_trace(trace, row=row, col=col)
               
                # Copy layout properties for each subplot
                for axis_type in ['xaxis', 'yaxis']:
                    axis_name = f"{axis_type}{i+1 if i > 0 else ''}"
                    subplot_name = f"{axis_type}{row}{col}"

                    # Copy axis properties if they exist
                    if hasattr(plot.layout, axis_name):
                        axis_props = getattr(plot.layout, axis_name)
                        fig.update_layout({subplot_name: axis_props})
       
        # Update layout for better appearance
        fig.update_layout(
            height=800,
            width=1000,
            title_text="Dashboard Overview",
            showlegend=False,
        )

        # Save to a temporary file
        dashboard_path = f"{st.session_state.temp_dir}/dashboard_combined.png"
        fig.write_image(dashboard_path, scale=2)  # Higher scale for better resolution
        return dashboard_path

    except Exception as e:
        import traceback
        print(f"Error capturing dashboard: {str(e)}")
        print(traceback.format_exc())
        return None
    
def generate_enhanced_pdf_report():
    """Generate an enhanced PDF report with proper handling of base64 image data"""
    try:
        # Create a buffer for the PDF
        buffer = io.BytesIO()
        
        # Create the PDF document
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                               leftMargin=36, rightMargin=36,
                               topMargin=36, bottomMargin=36)

        # Create custom styles with better formatting
        styles = getSampleStyleSheet()
        
        # Add custom styles with improved formatting
        styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=styles['Heading1'],
            fontSize=24,
            alignment=1,  # Centered
            spaceAfter=20,
            textColor='#2C3E50'  # Dark blue color
        ))
        
        styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=styles['Heading2'],
            fontSize=16,
            spaceBefore=15,
            spaceAfter=10,
            textColor='#2C3E50',
            borderWidth=1,
            borderColor='#95A5A6',
            borderPadding=5,
            borderRadius=5
        ))
        
        styles.add(ParagraphStyle(
            name='SubHeader',
            parent=styles['Heading3'],
            fontSize=14,
            spaceBefore=10,
            spaceAfter=8,
            textColor='#34495E',
            fontWeight='bold'
        ))
        styles.add(ParagraphStyle(
                name='UserMessage',
                parent=styles['Normal'],
                fontSize=11,
                leftIndent=10,
                spaceBefore=8,
                spaceAfter=4
            ))
            
        styles.add(ParagraphStyle(
            name='AssistantMessage',
            parent=styles['Normal'],
            fontSize=11,
            leftIndent=10,
            spaceBefore=4,
            spaceAfter=12,
            textColor='#2980B9'
        ))
        
        styles.add(ParagraphStyle(
            name='Timestamp',
            parent=styles['Italic'],
            fontSize=10,
            textColor='#7F8C8D',
            alignment=2  # Right aligned
        ))

        # Create the document content
        elements = []
        
        # Add title
        elements.append(Paragraph('Data Analysis Report', styles['ReportTitle']))
        
        # Add timestamp
        elements.append(Paragraph(f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                              styles['Timestamp']))
        elements.append(Spacer(1, 0.5*inch))
        
        # Add conversation history with better formatting
        elements.append(Paragraph('Analysis Conversation History', styles['SectionHeader']))
        
        if st.session_state.chat_history:
            for i, (user_msg, assistant_msg) in enumerate(st.session_state.chat_history):
                # Format user message with proper styling
                elements.append(Paragraph(f'<b>You:</b>', styles['SubHeader']))
                user_msg_formatted = user_msg.replace('\n', '<br/>')
                elements.append(Paragraph(user_msg_formatted, styles['UserMessage']))
                
                # Process assistant message to handle visualization
                # Look for markdown image syntax with base64 data
                base64_pattern = r'!\[Visualization\]\(data:image\/png;base64,([^\)]+)\)'
                
                # Check if the message contains visualizations
                if '### Visualizations' in assistant_msg or re.search(base64_pattern, assistant_msg):
                    # Split the message at the Visualizations header if it exists
                    if '### Visualizations' in assistant_msg:
                        parts = assistant_msg.split('### Visualizations', 1)
                        text_part = parts[0]
                        viz_part = "### Visualizations" + parts[1] if len(parts) > 1 else ""
                    else:
                        # If no header but still has visualization
                        match = re.search(base64_pattern, assistant_msg)
                        text_part = assistant_msg[:match.start()]
                        viz_part = assistant_msg[match.start():]
                    
                    # Format the text part
                    elements.append(Paragraph(f'<b>Assistant:</b>', styles['SubHeader']))
                    text_part = text_part.replace('\n', '<br/>')
                    elements.append(Paragraph(text_part, styles['AssistantMessage']))
                    
                    # Process visualizations
                    matches = re.findall(base64_pattern, viz_part)
                    for j, base64_data in enumerate(matches):
                        try:
                            # Decode the base64 image
                            image_data = base64.b64decode(base64_data)
                            
                            # Create a temporary file for the image
                            temp_img_path = f"{st.session_state.temp_dir}/chat_viz_{i}_{j}.png"
                            
                            with open(temp_img_path, 'wb') as f:
                                f.write(image_data)
                            
                            # Add the image to the PDF
                            elements.append(Paragraph(f'<b>Visualization:</b>', styles['SubHeader']))
                            elements.append(Spacer(1, 0.1*inch))
                            img = Image(temp_img_path, width=6*inch, height=4*inch)
                            elements.append(img)
                            elements.append(Spacer(1, 0.2*inch))
                        except Exception as e:
                            print(f"Error processing base64 image: {str(e)}")
                            elements.append(Paragraph(f"[Error displaying visualization: {str(e)}]", 
                                                    styles['Normal']))
                else:
                    # No visualizations, just format the text
                    elements.append(Paragraph(f'<b>Assistant:</b>', styles['SubHeader']))
                    assistant_msg_formatted = assistant_msg.replace('\n', '<br/>')
                    if len(assistant_msg_formatted) > 1500:
                        assistant_msg_formatted = assistant_msg_formatted[:1500] + '...'
                    elements.append(Paragraph(assistant_msg_formatted, styles['AssistantMessage']))
                
                elements.append(Spacer(1, 0.2*inch))
        else:
            elements.append(Paragraph('No conversation history available.', styles['Normal']))
        
        # Force a page break before the dashboard
        elements.append(PageBreak())
        
        # Add dashboard section header
        elements.append(Paragraph('Dashboard Overview', styles['SectionHeader']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Capture the dashboard as a single image
        dashboard_img_path = capture_dashboard_screenshot()
        
        if dashboard_img_path:
            # Calculate available width
            available_width = doc.width
            
            # Create PIL image to get dimensions
            pil_img = PILImage.open(dashboard_img_path)
            img_width, img_height = pil_img.size
            
            # Calculate scaling factor to fit within page width
            scale_factor = available_width / img_width
            
            # Calculate new height based on aspect ratio
            new_height = img_height * scale_factor
            
            # Add the image with scaled dimensions
            img = Image(dashboard_img_path, width=available_width, height=new_height)
            elements.append(img)
        else:
            # Fallback: Add individual plots if combined dashboard fails
            plot_count = 0
            for i, plot in enumerate(st.session_state.dashboard_plots):
                if plot is not None:
                    plot_count += 1
                    
                    # Convert plotly figure to image
                    img_bytes = io.BytesIO()
                    plot.write_image(img_bytes, format='png', width=500, height=300)
                    img_bytes.seek(0)
                    
                    # Create a temporary file for the image
                    temp_img_path = f"{st.session_state.temp_dir}/plot_{i}.png"
                    
                    with open(temp_img_path, 'wb') as f:
                        f.write(img_bytes.getvalue())
                    
                    # Add to PDF with appropriate caption and formatting
                    elements.append(Paragraph(f'Dashboard Visualization {i+1}', styles['SubHeader']))
                    elements.append(Spacer(1, 0.1*inch))
                    
                    # Add the image with proper scaling
                    img = Image(temp_img_path, width=6.5*inch, height=4*inch)
                    elements.append(img)
                    elements.append(Spacer(1, 0.3*inch))
            
            if plot_count == 0:
                elements.append(Paragraph('No visualizations have been added to the dashboard.',
                                        styles['Normal']))
        
        # Build the PDF with improved formatting
        doc.build(elements)
        
        # Get the value of the buffer
        pdf_value = buffer.getvalue()
        buffer.close()
        
        return pdf_value
        
    except Exception as e:
        import traceback
        print(f"Error generating enhanced PDF report: {str(e)}")
        print(traceback.format_exc())
        return None

def chat_with_workflow(message, history, dataset_info):
    """Send user query to the workflow and get response"""
    
    if not dataset_info:
        return "Please upload at least one dataset before asking questions."

    # Check if we have a valid API key and model
    if not st.session_state.api_key:
        return "Please set up your API key and model in the Settings tab before chatting."

    print(f"Chat with workflow called with {len(dataset_info)} datasets")

    try:
        # Extract chat history for context (last 3 exchanges)
        max_history = 3
        previous_messages = []
        
        if history:
            start_idx = max(0, len(history) - max_history)
            recent_history = history[start_idx:]
            
            for exchange in recent_history:
                if exchange[0]:  # User message
                    previous_messages.append(HumanMessage(content=exchange[0]))
                if exchange[1]:  # AI response
                    previous_messages.append(AIMessage(content=exchange[1]))

        # Initialize the workflow state
        state = AgentState(
            messages=previous_messages + [HumanMessage(content=message)],
            input_data=dataset_info,
            intermediate_outputs=[],
            current_variables=st.session_state.persistent_vars,
            output_image_paths=[]
        )

        # Execute the workflow
        print("Executing workflow...")
        result = chain.invoke(state)
        print("Workflow execution completed")

        # Extract messages from the result
        messages = result["messages"]

        # Format the response - only get the latest response
        response = ""
        if messages:
            latest_message = messages[-1]  # Get only the last message
            if hasattr(latest_message, "content"):
                content = latest_message.content
                
                # Clean up the response
                # Remove any instances where the user's message is repeated
                if message in content:
                    content = content.split(message)[-1].strip()
                
                # Remove any chat history markers
                content_lines = content.split('\n')
                filtered_lines = [line for line in content_lines 
                                if not line.strip().startswith(("You:", "User:", "Human:", "Assistant:"))]
                content = '\n'.join(filtered_lines)
                
                response = content.strip() + "\n\n"

        # Handle visualizations
        if "output_image_paths" in result and result["output_image_paths"]:
            response += "### Visualizations\n\n"
            for img_path in result["output_image_paths"]:
                try:
                    full_path = os.path.join(st.session_state.images_dir, img_path)
                    with open(full_path, 'rb') as f:
                        fig = pickle.load(f)

                    # Convert plotly figure to image
                    img_bytes = BytesIO()
                    fig.update_layout(width=800, height=500)
                    pio.write_image(fig, img_bytes, format='png')
                    img_bytes.seek(0)

                    # Convert to base64 for markdown image
                    b64_img = base64.b64encode(img_bytes.read()).decode()
                    response += f"![Visualization](data:image/png;base64,{b64_img})\n\n"
                except Exception as e:                    
                    response += f"Error loading visualization: {str(e)}\n\n"
                    
        return response

    except Exception as e:
        import traceback
        print(f"Error in chat_with_workflow: {str(e)}")
        print(traceback.format_exc())
        return f"Error executing workflow: {str(e)}"

def auto_generate_dashboard(dataset_info):
    """Generate an automatic dashboard with four plots"""

    if not dataset_info:
        return "Please upload a dataset first.", [None, None, None, None]

    prompt = """
    You are a data visualization expert. Given a dataset, identify the top 4 most insightful plots using statistical reasoning or patterns (correlation, distribution, trends).

    Use plotly and store the plots in a list named plotly_figures.

    Include multivariate plots using color/size/facets when helpful.
    """

    state = AgentState(
        messages=[HumanMessage(content=prompt)],
        input_data=dataset_info,
        intermediate_outputs=[],
        current_variables=st.session_state.persistent_vars,
        output_image_paths=[]
    )

    result = chain.invoke(state)
    figures = []

    if "output_image_paths" in result:
        for img_path in result["output_image_paths"][:4]:
            try:
                full_path = os.path.join(st.session_state.images_dir, img_path)
                with open(full_path, 'rb') as f:
                    fig = pickle.load(f)
                    figures.append(fig)
            except Exception as e:
                print(f"Error loading figure: {e}")

    while len(figures) < 4:
        figures.append(None)

    st.session_state.dashboard_plots = figures
    return "Dashboard generated!", figures

def generate_custom_plots_with_llm(dataset_info, x_col, y_col, facet_col):
    """Generate custom plots based on user-selected columns"""

    if not dataset_info or not x_col or not y_col:
        return [None, None, None]

    prompt = f"""
    You are a data visualization expert.

    Create 3 insightful visualizations using Plotly based on:

    - X-axis: {x_col}
    - Y-axis: {y_col}
    - Facet (optional): {facet_col if facet_col != 'None' else 'None'}

    Try to find interesting relationships, trends, or clusters using appropriate chart types.

    Use `plotly_figures` list and avoid using fig.show().
    """

    state = AgentState(
        messages=[HumanMessage(content=prompt)],
        input_data=dataset_info,
        intermediate_outputs=[],
        current_variables=st.session_state.persistent_vars,
        output_image_paths=[]
    )

    result = chain.invoke(state)
    figures = []

    if "output_image_paths" in result:
        for img_path in result["output_image_paths"][:3]:
            try:
                full_path = os.path.join(st.session_state.images_dir, img_path)
                with open(full_path, 'rb') as f:
                    fig = pickle.load(f)
                    figures.append(fig)
            except Exception as e:
                print(f"Error loading figure: {e}")
               
    while len(figures) < 3:
        figures.append(None)
    return figures

def remove_plot(index):
    """Remove a plot from the dashboard"""
    if 0 <= index < len(st.session_state.dashboard_plots):
        st.session_state.dashboard_plots[index] = None

def respond(message):
    """Handle chat message response"""
    if not st.session_state.dataset_metadata_list:
        bot_message = "Please upload at least one dataset before asking questions."
    else:
        bot_message = chat_with_workflow(message, st.session_state.chat_history, st.session_state.dataset_metadata_list)

    st.session_state.chat_history.append((message, bot_message))
    st.rerun()

def save_plot_to_dashboard(plot_index):
    """Callback for the Add Plot button"""
    for i in range(len(st.session_state.dashboard_plots)):
        if st.session_state.dashboard_plots[i] is None:
            # Found an empty slot
            st.session_state.dashboard_plots[i] = st.session_state.custom_plots_to_save[plot_index]
            return

# New function to check if settings are valid
def is_settings_valid():
    """Check if API key and model are configured"""
    return st.session_state.api_key != ""

# Streamlit UI with left panel settings
st.set_page_config(page_title="Data Analysis Assistant", layout="wide")

# Create side panel for settings
with st.sidebar:
    st.header("Settings")
    st.info("⚠️ Configure your API settings before uploading data")
    
    # AI Provider selection
    provider = st.radio("Select AI Provider", 
                         options=["OpenAI", "Groq"],
                         index=0 if st.session_state.ai_provider == "openai" else 1,
                         horizontal=True)
    
    # Update session state based on selection
    st.session_state.ai_provider = provider.lower()
    
    # API Key input
    api_key = st.text_input("Enter API Key", 
                           value=st.session_state.api_key, 
                           type="password",
                           help="Your API key for the selected provider")
    
    # Display different model options based on provider
    if st.session_state.ai_provider == "openai":
        model_options = OPENAI_MODELS
        model_help = "GPT-4 provides the best results but is slower. GPT-3.5-Turbo is faster but less capable."
    else:  # groq
        model_options = GROQ_MODELS
        model_help = "Llama 3.3 70B is most capable. Gemma 2 9B offers good balance. Llama 3 8B is fastest."
    
    # Model selection
    selected_model = st.selectbox("Select Model", 
                                 options=model_options,
                                 index=model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0,
                                 help=model_help)
    
    # Save button
    if st.button("Save Settings"):
        st.session_state.api_key = api_key
        st.session_state.selected_model = selected_model
        
        # Test the API key and model
        try:
            # Initialize LLM using the provided settings
            test_llm = initialize_llm()
            if test_llm:
                st.success(f"✅ Successfully configured {provider} with model: {selected_model}")
            else:
                st.error("Failed to initialize the AI provider. Please check your API key and model selection.")
        except Exception as e:
            st.error(f"Error testing settings: {str(e)}")
    
    # Display current settings
    st.subheader("Current Settings")
    settings_info = f"""
    - **Provider**: {st.session_state.ai_provider.upper()}
    - **Model**: {st.session_state.selected_model}
    - **API Key**: {'✅ Set' if st.session_state.api_key else '❌ Not Set'}
    """
    st.markdown(settings_info)
    
    # Provider-specific information
    if st.session_state.ai_provider == "openai":
        with st.expander("OpenAI Models Info"):
            st.info("""
            - **GPT-4**: Most powerful model, best for complex analysis and detailed explanations
            - **GPT-4-Turbo**: Faster than GPT-4 with similar capabilities
            - **GPT-4-Mini**: Economical option with good performance for standard tasks
            - **GPT-3.5-Turbo**: Fastest option, suitable for basic analysis and visualization
            """)
    else:
        with st.expander("Groq Models Info"):
            st.info("""
            - **llama3.3-70b-versatile**: Most powerful model for comprehensive analysis
            - **gemma2-9b-it**: Good balance of speed and capabilities
            - **llama-3-8b-8192**: Fastest option for basic analysis tasks
            """)
    
    # Integration instructions
    with st.expander("How to get API Keys"):
        if st.session_state.ai_provider == "openai":
            st.markdown("""
            ### Getting an OpenAI API Key
            
            1. Go to [OpenAI's platform](https://platform.openai.com)
            2. Sign up or log in to your account
            3. Navigate to the API section
            4. Create a new API key
            5. Copy the key and paste it above
            
            Note: OpenAI API usage incurs charges based on tokens used.
            """)
        else:
            st.markdown("""
            ### Getting a Groq API Key
            
            1. Go to [Groq's website](https://console.groq.com/keys)
            2. Sign up or log in to your account
            3. Navigate to API Keys section
            4. Create a new API key
            5. Copy the key and paste it above
            
            Note: Check Groq's pricing page for current rates.
            """)

# Main content
st.title("Data Analysis Assistant")
st.markdown("Upload your datasets, ask questions, and generate visualizations to gain insights.")

# Check if settings are valid before showing tabs
if not is_settings_valid():
    st.warning("⚠️ Please configure your API key in the sidebar settings panel before proceeding.")
    st.stop()

# Create main tabs - only if settings are valid
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Upload Datasets", "Data Cleaning", "Chat with AI Assistant", "Auto Dashboard Generator", "Generate Report"])

with tab1:
    st.header("Upload Datasets")
    uploaded_files = st.file_uploader("Upload CSV or Excel Files",
                                    accept_multiple_files=True,
                                    type=['csv', 'xlsx', 'xls'])

    if uploaded_files and st.button("Process Uploaded Files"):
        with st.spinner("Processing files..."):
            preview, metadata_list, columns = process_file_upload(uploaded_files)
            st.session_state.columns = columns
           
            # Display basic information about processed files
            st.success(f"✅ Successfully processed {len(uploaded_files)} file(s)")
           
            # Show detailed preview for each dataset
            st.subheader("Dataset Previews")
           
            for dataset_name, df in st.session_state.in_memory_datasets.items():
                with st.expander(f"Preview: {dataset_name}"):
                    # Display dataset info
                    st.write(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")
                   
                    # Display column information
                    col_info = pd.DataFrame({
                        'Column Name': df.columns,
                        'Data Type': df.dtypes.astype(str),
                        'Non-Null Count': df.count().values,
                        'Sample Values': [', '.join(df[col].dropna().astype(str).head(3).tolist()) for col in df.columns]
                    })
                   
                    # Show column information in a compact table
                    st.write("**Column Information:**")
                    st.dataframe(col_info, use_container_width=True)
                   
                    # Show actual data preview
                    st.write("**Data Preview (First 10 rows):**")
                    st.dataframe(df.head(10), use_container_width=True)
           
            # Provide hint for the next steps
            st.info("👆 Click on the dataset names above to see detailed previews. Then proceed to the Data Cleaning tab to clean your data or Chat with AI Assistant to analyze it.")

with tab2:
    st.header("Data Cleaning")

    if 'cleaning_done' not in st.session_state:
        st.session_state.cleaning_done = False

    if 'cleaned_datasets' not in st.session_state:
        st.session_state.cleaned_datasets = {}

    if 'cleaning_summaries' not in st.session_state:
        st.session_state.cleaning_summaries = {}

    if st.session_state.get("in_memory_datasets"):
        if not st.session_state.cleaning_done:
            if st.button("Run Data Cleaning"):
                with st.spinner("Running LLM-assisted cleaning..."):
                    for name, df in st.session_state.in_memory_datasets.items():
                        raw_df = df.copy()
                        df_std = standard_clean(raw_df.copy())
                        suggestions = llm_suggest_cleaning(df_std.copy())
                        df_clean = apply_suggestions(df_std.copy(), suggestions)
                        st.session_state.cleaned_datasets[name] = df_clean
                        st.session_state.cleaning_summaries[name] = suggestions
                    st.session_state.cleaning_done = True
                    st.rerun()
            else:
                st.info("Click Run Data Cleaning to clean your datasets using the LLM.")
        else:
            for name, df_clean in st.session_state.cleaned_datasets.items():
                raw_df = st.session_state.in_memory_datasets[name]

                st.subheader(f"Dataset: {name}")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("Original Data (First 5 Rows)")
                    st.dataframe(raw_df.head())

                with col2:
                    st.markdown("Cleaned Data (First 5 Rows)")
                    st.dataframe(df_clean.head())

                st.markdown("Summary of Cleaning Actions")
                suggestions = st.session_state.cleaning_summaries[name]
                summary_text = ""

                if suggestions:
                    for key, value in suggestions.items():
                        summary_text += f"**{key}**: {json.dumps(value, indent=2)}\n\n"
                    st.markdown(summary_text)

                st.markdown("Refine the Cleaning (Natural Language Instructions)")
                user_input = st.text_input("Example: Convert 'dob' to datetime and fill missing with '2000-01-01'",
                                           key=f"user_input_{name}")

                if f'corrections_{name}' not in st.session_state:
                    st.session_state[f'corrections_{name}'] = []

                if st.button("Apply Correction", key=f'apply_correction_{name}'):
                    if user_input.strip():
                        correction_prompt = f"""
You are a data cleaning expert. Below is a previously cleaned dataset with these actions:

{summary_text}

The user now wants the following additional instruction:
\"{user_input.strip()}\"

Write only the Python code that modifies the pandas DataFrame `df` accordingly.
Do not include explanations or markdown.
"""
                        correction_code = query_openai(correction_prompt)

                        try:
                            df = st.session_state.cleaned_datasets[name].copy()
                            local_vars = {"df": df}
                            exec(correction_code, {}, local_vars)
                            df_updated = local_vars["df"]

                            st.session_state.cleaned_datasets[name] = df_updated
                            st.session_state[f'corrections_{name}'].append((user_input, correction_code))
                            st.success("Correction applied.")
                            st.rerun()

                        except Exception as e:
                            st.error(f"Failed to apply correction: {str(e)}")

                if st.session_state[f'corrections_{name}']:
                    st.markdown("Applied Corrections")
                    for i, (msg, code) in enumerate(st.session_state[f'corrections_{name}']):
                        st.markdown(f"**Instruction:** {msg}")
                        st.code(code, language='python')

            col1, col2 = st.columns([1, 2])
            with col1:
                if st.button("Reset Cleaning and Re-run"):
                    st.session_state.cleaning_done = False
                    st.rerun()

            with col2:
                if st.button("Finalize and Proceed to Visualizations"):
                    st.session_state.cleaning_finalized = True
                    st.rerun()
    else:
        st.info("Please upload and process datasets first.")

with tab3:
    st.header("Chat with AI Assistant")
   
    st.markdown("""
    ## Example Questions
    - "What analysis can you perform on this dataset?"
    - "Show me basic statistics for all columns"
    - "Create a correlation heatmap"
    - "Plot the distribution of a specific column"
    - "What is the relationship between two columns?"
    """)

    # Display chat history
    for exchange in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(exchange[0])
        with st.chat_message("assistant"):
            st.write(exchange[1])

    # Chat input
    if prompt := st.chat_input("Your question"):
        with st.spinner("Thinking..."):
            respond(prompt)

with tab4:
    st.header("Auto Dashboard Generator")

    # Dashboard controls
    dashboard_title = st.text_input("Dashboard Title", placeholder="Enter your dashboard title")
 
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Generate Suggested Dashboard (Auto)"):
            with st.spinner("Generating dashboard..."):
                message, figures = auto_generate_dashboard(st.session_state.dataset_metadata_list)
                st.success(message)

    with col2:
        if st.button("Refresh Column Options"):
            st.session_state.columns = get_columns()
            st.rerun()

    # Dashboard display
    st.subheader("Dashboard")

    # Row 1
    col1, col2 = st.columns(2)

    with col1:
        if st.session_state.dashboard_plots[0]:
            st.plotly_chart(st.session_state.dashboard_plots[0], use_container_width=True)
            if st.button("Remove Plot 1"):
                remove_plot(0)
                st.rerun()

    with col2:
        if st.session_state.dashboard_plots[1]:
            st.plotly_chart(st.session_state.dashboard_plots[1], use_container_width=True)
            if st.button("Remove Plot 2"):
                remove_plot(1)
                st.rerun()

    # Row 2
    col3, col4 = st.columns(2)

    with col3:
        if st.session_state.dashboard_plots[2]:
            st.plotly_chart(st.session_state.dashboard_plots[2], use_container_width=True)
            if st.button("Remove Plot 3"):
                remove_plot(2)
                st.rerun()

    with col4:
        if st.session_state.dashboard_plots[3]:
            st.plotly_chart(st.session_state.dashboard_plots[3], use_container_width=True)
            if st.button("Remove Plot 4"):
                remove_plot(3)
                st.rerun()

    # Custom plot generator
    st.subheader("Custom Plot Generator")

    # Column selection
    col1, col2, col3 = st.columns(3)

    with col1:
        x_axis = st.selectbox("X-axis Column", options=st.session_state.columns)

    with col2:
        y_axis = st.selectbox("Y-axis Column", options=st.session_state.columns)

    with col3:
        facet = st.selectbox("Facet (optional)", options=["None"] + st.session_state.columns)
 
    if st.button("Generate Custom Visualizations"):
        with st.spinner("Generating custom visualizations..."):
            custom_plots = generate_custom_plots_with_llm(st.session_state.dataset_metadata_list, x_axis, y_axis, facet)
            # Store plots in session state
            for i, plot in enumerate(custom_plots):
                if plot:
                    st.session_state.custom_plots_to_save[i] = plot

            # Display custom plots with add buttons
            for i, plot in enumerate(custom_plots):
                if plot:
                    st.plotly_chart(plot, use_container_width=True)
                    st.button(
                        f"Add Plot {i+1} to Dashboard",
                        key=f"add_plot_{i}",
                        on_click=save_plot_to_dashboard,
                        args=(i,)
                    )

with tab5:
    st.header("Generate Analysis Report")

    st.markdown("""
    Generate a PDF report containing:
    - Dashboard visualizations
    - Chat conversation history
    """)

    report_title = st.text_input("Report Title (Optional)", "Data Analysis Report")
   
    if st.button("Generate PDF Report"):
        with st.spinner("Generating report..."):
            pdf_data = generate_enhanced_pdf_report()
            if pdf_data:
                # Create download button for PDF
                b64_pdf = base64.b64encode(pdf_data).decode('utf-8')
                # Create download link
                pdf_download_link = f'<a href="data:application/pdf;base64,{b64_pdf}" download="data_analysis_report.pdf">Download PDF Report</a>'
                st.markdown("### Your report is ready!")
                st.markdown(pdf_download_link, unsafe_allow_html=True)
                # Preview option (simplified)
                with st.expander("Preview Report"):
                    st.warning("PDF preview is not available in Streamlit, please download the report to view it.")
            else:
                st.error("Failed to generate the report. Please try again.")

# Cleanup on app exit
def cleanup():
    try:
        shutil.rmtree(st.session_state.temp_dir)
        print(f"Cleaned up temporary directory: {st.session_state.temp_dir}")
    except Exception as e:
        print(f"Error cleaning up: {e}")

import atexit
atexit.register(cleanup)