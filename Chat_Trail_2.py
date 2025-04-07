import os
import streamlit as st
import pandas as pd
import numpy as np
import uuid
import time
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
# from fpdf import FPDF
import base64
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
 

# Import LangChain and LangGraph components

from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_experimental.utilities import PythonREPL
from langgraph.prebuilt import ToolInvocation, ToolExecutor
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langgraph.graph import StateGraph, END
from reportlab.platypus import PageBreak
from PIL import Image as PILImage

# Set your Groq API key

# os.environ["GROQ_API_KEY"] = " ADD YOUR API KEY HERE "

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

  - `sklearn`

  - `plotly`

All these libraries are already imported for you.

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
        exec_globals.update({"plotly_figures": [], "images_dir": st.session_state.images_dir})
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

# Set up the LLM and tools ( For testing purposes use the model names mentioned in the comments)
llm = ChatGroq(model="gemma2-9b-it", temperature=0)

# "deepseek-r1-distill-llama-70b"
# "meta-llama/llama-4-scout-17b-16e-instruct"
# "deepseek-r1-distill-qwen-32b"

 
tools = [complete_python_task]

model = llm.bind_tools(tools)

tool_executor = ToolExecutor(tools)

# Load the prompt template
chat_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("placeholder", "{messages}"),
])

model = chat_template | model

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

 

# def generate_pdf_report():

#     """Generate a PDF report with chat history and dashboard visualizations"""

#     try:

#         # Create PDF object with Unicode support

#         pdf = FPDF()

#         pdf.add_page()

#         pdf.add_font('DejaVu', '', 'DejaVuSansCondensed.ttf', uni=True)

       

#         # Set title

#         pdf.set_font('Arial', 'B', 16)

#         pdf.cell(0, 10, 'Data Analysis Report', 0, 1, 'C')

#         pdf.ln(10)

       

#         # Add timestamp

#         pdf.set_font('Arial', 'I', 10)

#         pdf.cell(0, 5, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'R')

#         pdf.ln(10)

       

#         # Add dashboard plots if available

#         pdf.set_font('Arial', 'B', 14)

#         pdf.cell(0, 10, 'Dashboard Visualizations', 0, 1, 'L')

       

#         plot_count = 0

#         for i, plot in enumerate(st.session_state.dashboard_plots):

#             if plot is not None:

#                 plot_count += 1

#                 # Convert plotly figure to image

#                 img_bytes = io.BytesIO()

#                 plot.write_image(img_bytes, format='png', width=500, height=300)

#                 img_bytes.seek(0)

               

#                 # Create a temporary file for the image

#                 temp_img_path = f"{st.session_state.temp_dir}/plot_{i}.png"

#                 with open(temp_img_path, 'wb') as f:

#                     f.write(img_bytes.getvalue())

               

#                 # Add to PDF

#                 pdf.ln(5)

#                 pdf.cell(0, 5, f'Visualization {i+1}', 0, 1, 'L')

#                 pdf.image(temp_img_path, x=10, w=180)

#                 pdf.ln(5)

       

#         if plot_count == 0:

#             pdf.set_font('Arial', '', 10)

#             pdf.cell(0, 10, 'No visualizations have been added to the dashboard.', 0, 1, 'L')

       

#         # Add chat history

#         pdf.add_page()

#         pdf.set_font('Arial', 'B', 14)

#         pdf.cell(0, 10, 'Analysis Conversation History', 0, 1, 'L')

       

#         if st.session_state.chat_history:

#             pdf.set_font('Arial', '', 10)

#             for i, (user_msg, assistant_msg) in enumerate(st.session_state.chat_history):

#                 # Clean messages of emojis and other problematic characters

#                 user_msg_clean = ''.join(c for c in user_msg if ord(c) < 128)

               

#                 # Simplify assistant message (remove markdown and image references)

#                 assistant_msg_clean = assistant_msg.replace('![Visualization]', '[Visualization included in dashboard]')

#                 assistant_msg_clean = ''.join(c for c in assistant_msg_clean if ord(c) < 128)

               

#                 pdf.ln(5)

#                 pdf.set_font('Arial', 'B', 10)

#                 pdf.cell(0, 5, f'You: ', 0, 1, 'L')

               

#                 pdf.set_font('Arial', '', 10)

#                 pdf.multi_cell(0, 5, user_msg_clean)

               

#                 pdf.ln(3)

#                 pdf.set_font('Arial', 'B', 10)

#                 pdf.cell(0, 5, f'Assistant: ', 0, 1, 'L')

               

#                 pdf.set_font('Arial', '', 10)

#                 pdf.multi_cell(0, 5, assistant_msg_clean[:1000] + ('...' if len(assistant_msg_clean) > 1000 else ''))

               

#                 pdf.ln(5)

#         else:

#             pdf.set_font('Arial', '', 10)

#             pdf.cell(0, 10, 'No conversation history available.', 0, 1, 'L')

           

#         # Save PDF to a bytes buffer

#         pdf_output = io.BytesIO()

#         pdf.output(pdf_output)

#         pdf_output.seek(0)

       

#         return pdf_output.getvalue()

       

#     except Exception as e:

#         import traceback

#         print(f"Error generating PDF report: {str(e)}")

#         print(traceback.format_exc())

#         return None

 

# import io

# from reportlab.lib.pagesizes import letter

# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image

# from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# from reportlab.lib.units import inch

# import plotly.io as pio

# from PIL import Image as PILImage

# import numpy as np

# import base64

# from datetime import datetime

 

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

    """Generate an enhanced PDF report with chat history first and dashboard screenshot"""

    try:

        # Create a buffer for the PDF

        buffer = io.BytesIO()
        
        # Create the PDF document with adjusted page size if needed

        doc = SimpleDocTemplate(buffer, pagesize=letter,
                               leftMargin=36, rightMargin=36,
                               topMargin=36, bottomMargin=36)

        styles = getSampleStyleSheet()

        # Add custom styles

        styles.add(ParagraphStyle(name='ReportTitle',

                                  parent=styles['Heading1'],

                                  alignment=1))  # 1 is centered


        # Create the document content

        elements = []

        # Add title

        elements.append(Paragraph('Data Analysis Report', styles['ReportTitle']))
        elements.append(Spacer(1, 0.25*inch))


        # Add timestamp

        timestamp = Paragraph(f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',

                              styles['Italic'])

        elements.append(timestamp)
        elements.append(Spacer(1, 0.5*inch))

       
        # Add conversation history FIRST

        elements.append(Paragraph('Analysis Conversation History', styles['Heading2']))
        elements.append(Spacer(1, 0.1*inch))

       
        if st.session_state.chat_history:

            for i, (user_msg, assistant_msg) in enumerate(st.session_state.chat_history):

                elements.append(Paragraph(f'<b>You:</b>', styles['Normal']))
                elements.append(Paragraph(user_msg, styles['Normal']))
                elements.append(Spacer(1, 0.1*inch))
                elements.append(Paragraph(f'<b>Assistant:</b>', styles['Normal']))

                # Simplify assistant message (remove markdown and image references)

                simplified_msg = assistant_msg.replace('![Visualization]', '[Visualization included in dashboard]')
                simplified_msg = simplified_msg[:1000] + ('...' if len(simplified_msg) > 1000 else '')
                elements.append(Paragraph(simplified_msg, styles['Normal']))
                elements.append(Spacer(1, 0.2*inch))

        else:
            elements.append(Paragraph('No conversation history available.', styles['Normal']))

       

        # Force a page break before the dashboard

        elements.append(PageBreak())

        # Add dashboard as a single screenshot (on a new page)

        elements.append(Paragraph('Dashboard Overview', styles['Heading2']))
        elements.append(Spacer(1, 0.1*inch))

       
        # Capture the dashboard as a single image

        dashboard_img_path = capture_dashboard_screenshot()

       

        if dashboard_img_path:

            # Calculate available width (accounting for margins)
            
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
            elements.append(Paragraph('Dashboard Visualizations (Individual)', styles['Heading3']))

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

                    # Add to PDF with appropriate scaling

                    img = Image(temp_img_path, width=5*inch, height=3*inch)
                    elements.append(Paragraph(f'Visualization {i+1}', styles['Heading4']))
                    elements.append(Spacer(1, 0.1*inch))
                    elements.append(img)
                    elements.append(Spacer(1, 0.2*inch))          

            if plot_count == 0:
                elements.append(Paragraph('No visualizations have been added to the dashboard.',
                                        styles['Normal']))

       

        # Build the PDF
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

    print(f"Chat with workflow called with {len(dataset_info)} datasets")

    try:

        # Extract chat history for context

        previous_messages = []

        for exchange in history:

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

        # Format the response

        response = ""

        for msg in messages:
            if hasattr(msg, "content"):                
                response += msg.content + "\n\n"

        # Check if there are any visualization images

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

# def add_custom_to_dashboard(fig, index):

#     """Add a custom plot to the dashboard"""

#     if fig is not None:

#         # Find the first empty slot

#         for i in range(len(st.session_state.dashboard_plots)):

#             if st.session_state.dashboard_plots[i] is None:

#                 st.session_state.dashboard_plots[i] = fig

#                 break

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


# Streamlit UI
st.set_page_config(page_title="Data Analysis Assistant", layout="wide")
st.title("Data Analysis Assistant")
st.markdown("Upload your datasets, ask questions, and generate visualizations to gain insights.")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Upload Datasets", "Chat with AI Assistant", "Auto Dashboard Generator", "Generate Report"])

with tab1:

    st.header("Upload Datasets")
    uploaded_files = st.file_uploader("Upload CSV or Excel Files",
                                    accept_multiple_files=True,
                                    type=['csv', 'xlsx', 'xls'])

  

    if uploaded_files and st.button("Process Uploaded Files"):

        with st.spinner("Processing files..."):
            preview, _, columns = process_file_upload(uploaded_files)
            st.markdown(preview)
            st.session_state.columns = columns
            st.rerun()

with tab2:
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

with tab3:
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
 

with tab4:

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