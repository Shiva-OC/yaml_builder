import streamlit as st
from pydantic import BaseModel, Field
from typing import List, Literal, Any, Dict, get_args
import yaml
import subprocess
import time
import os

# Step 1: Define the Pydantic Model with Instructions
class DefaultConfiguration(BaseModel):
    splitter: Literal['token_text_splitter', 'semantic_splitter', 'sentence_text_splitter', 'sentence_text_splitter_LC'] = Field(
        'sentence_text_splitter_LC', description="Select Splitter (Options: token_text_splitter, semantic_splitter, sentence_text_splitter, sentence_text_splitter_LC)"
    )
    embedding: Literal['openai', 'huggingface', 'sky', 'dkubex'] = Field(
        'huggingface', description="Select Embedding (Options: openai, huggingface, sky, dkubex)"
    )
    metadata: List[Literal['default', 'custom']] = Field(
        ['default'], description="Select Metadata Options (Options: default, custom)"
    )
    reader: List[Literal['file', 'directoryreader', 'scrapeddatareader', 'scrapyreader', 'pyreader', 'githubreader', 'sharepointreader']] = Field(
        ['scrapyreader'], description="Select Reader (Options: file, directoryreader, scrapeddatareader, scrapyreader, pyreader, githubreader, sharepointreader)"
    )
    adjacent_chunks: bool = Field(True, description="Select whether adjacent chunks should be processed.")

    sentence_text_splitter: Dict[str, Any] = Field({
        "chunk_size": {"type": "slider", "value": 256, "min": 0, "max": 1024},
        "chunk_overlap": {"type": "slider", "value": 0, "min": 0, "max": 1024}
    }, description="Sentence Text Splitter Parameters (chunk_size: slider, chunk_overlap: slider)")

    sentence_text_splitter_LC: Dict[str, Any] = Field({
        "chunk_size": {"type": "slider", "value": 256, "min": 0, "max": 1024},
        "chunk_overlap": {"type": "slider", "value": 0, "min": 0, "max": 1024}
    }, description="Sentence Text Splitter LC Parameters (chunk_size: slider, chunk_overlap: slider)")

    token_text_splitter: Dict[str, Any] = Field({
        "chunk_size": {"type": "slider", "value": 256, "min": 0, "max": 2048},
        "chunk_overlap": {"type": "slider", "value": 10, "min": 1, "max": 10}
    }, description="Token Text Splitter Parameters (chunk_size: slider, chunk_overlap: slider)")

    huggingface: Dict[str, Any] = Field({
        "model": {"type": "text_input", "value": "BAAI/bge-large-en-v1.5"}
    }, description="Huggingface Parameters (model: text_input)")

    openai: Dict[str, Any] = Field({
        "model": {"type": "text_input", "value": "text-embedding-ada-002"},
        "llmkey": {"type": "text_input", "value": "<llmkey>"}
    }, description="OpenAI Parameters (model: text_input, llmkey: text_input)")

    dkubex: Dict[str, Any] = Field({
        "embedding_key": {"type": "text_input", "value": "<embedding_key>"},
        "embedding_url": {"type": "text_input", "value": "<embedding_url>"},
        "batch_size": {"type": "number_input", "value": 10, "min": 1}
    }, description="Dkubex Parameters (embedding_key: text_input, embedding_url: text_input, batch_size: number_input)")

    sky: Dict[str, Any] = Field({
        "embedding_key": {"type": "text_input", "value": "<embedding_key>"},
        "embedding_url": {"type": "text_input", "value": "<embedding_url>"},
        "batch_size": {"type": "number_input", "value": 10, "min": 1}
    }, description="Sky Parameters (embedding_key: text_input, embedding_url: text_input, batch_size: number_input)")

    mlflow: Dict[str, Any] = Field({
        "experiment": {"type": "text_input", "value": "GI-ingestion"}
    }, description="MLFlow Parameters (experiment: text_input)")

    scrapyreader: Dict[str, Any] = Field({
        "inputs": {"type": "text_input", "value": ""},
        "data_args": {"spiders": {"gi": {"url": "https://company.getinsured.com/state-based-marketplaces/state-based-marketplace-resources/"}}}
    }, description="Scrapy Reader Parameters (inputs: text_input, data_args: dict)")

    @staticmethod
    def get_parameters(selection: str) -> Dict[str, Any]:
        if selection == 'sentence_text_splitter_LC':
            return DefaultConfiguration().sentence_text_splitter_LC
        elif selection == 'sentence_text_splitter':
            return DefaultConfiguration().sentence_text_splitter
        elif selection == 'token_text_splitter':
            return DefaultConfiguration().token_text_splitter
        elif selection == 'huggingface':
            return DefaultConfiguration().huggingface
        elif selection == 'openai':
            return DefaultConfiguration().openai
        elif selection == 'dkubex':
            return DefaultConfiguration().dkubex
        elif selection == 'sky':
            return DefaultConfiguration().sky
        elif selection == 'mlflow':
            return DefaultConfiguration().mlflow
        elif selection == 'scrapyreader':
            return DefaultConfiguration().scrapyreader
        else:
            return {}

# Step 2: Create the Form and Handle Submission
def create_form(model_class):
    form_data = {}

    for field_name, field_info in model_class.__fields__.items():
        field_value = field_info.default
        field_description = field_info.field_info.description

        if get_args(field_info.type_):
            options = get_args(field_info.type_)
            if isinstance(field_value, list):
                selected_options = st.multiselect(field_description, options=options, default=field_value, key=f"{field_name}")
                form_data[field_name] = selected_options
            else:
                selected_index = options.index(field_value) if field_value in options else 0
                selected_option = st.selectbox(field_description, options=options, index=selected_index, key=f"{field_name}")
                form_data[field_name] = selected_option

                if field_name in ["splitter", "embedding"]:
                    extra_params = model_class.get_parameters(selected_option)
                    for sub_field, sub_field_info in extra_params.items():
                        if "type" not in sub_field_info:
                            continue  # Skip if 'type' key is not present
                        input_type = sub_field_info["type"]
                        if input_type == "slider":
                            form_data[sub_field] = st.slider(sub_field.capitalize().replace("_", " "), 
                                                             min_value=sub_field_info["min"], 
                                                             max_value=sub_field_info["max"], 
                                                             value=sub_field_info["value"], 
                                                             key=f"{field_name}_{sub_field}")
                        elif input_type == "text_input":
                            form_data[sub_field] = st.text_input(sub_field.capitalize().replace("_", " "), 
                                                                 value=sub_field_info["value"], 
                                                                 key=f"{field_name}_{sub_field}")
                        elif input_type == "number_input":
                            form_data[sub_field] = st.number_input(sub_field.capitalize().replace("_", " "), 
                                                                   min_value=sub_field_info.get("min", None), 
                                                                   value=sub_field_info["value"], 
                                                                   key=f"{field_name}_{sub_field}")

        elif isinstance(field_value, dict) and field_name not in ["splitter", "embedding"]:
            for sub_field, sub_field_info in field_value.items():
                if "type" not in sub_field_info:
                    continue  # Skip if 'type' key is not present
                input_type = sub_field_info["type"]
                if input_type == "slider":
                    form_data[sub_field] = st.slider(sub_field.capitalize().replace("_", " "), 
                                                     min_value=sub_field_info["min"], 
                                                     max_value=sub_field_info["max"], 
                                                     value=sub_field_info["value"], 
                                                     key=f"{field_name}_{sub_field}")
                elif input_type == "text_input":
                    form_data[sub_field] = st.text_input(sub_field.capitalize().replace("_", " "), 
                                                         value=sub_field_info["value"], 
                                                         key=f"{field_name}_{sub_field}")
                elif input_type == "number_input":
                    form_data[sub_field] = st.number_input(sub_field.capitalize().replace("_", " "), 
                                                           min_value=sub_field_info.get("min", None), 
                                                           value=sub_field_info["value"], 
                                                           key=f"{field_name}_{sub_field}")
        else:
            form_data[field_name] = st.checkbox(field_description, value=field_value, key=f"{field_name}")
    
    return form_data


def run_command(command, placeholder):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output = ""
    
    for line in iter(process.stdout.readline, ''):
        output += line
        st.session_state['output'] = output.replace('\n', '<br>')
        placeholder.markdown(
            f"<div style='background-color: #1a1c24; color: #ffffff; font-family: monospace; white-space: pre-wrap; height: 300px; width: 100%; overflow-y: auto; padding: 10px; border-radius: 10px;'>{st.session_state['output']}</div><div id='end-of-output'></div>",
            unsafe_allow_html=True
        )
        scroll_js = """
        <script>
        var element = document.getElementById("end-of-output");
        element.scrollIntoView({behavior: "smooth"});
        </script>
        """
        st.markdown(scroll_js, unsafe_allow_html=True)
        time.sleep(0.2)  # Slightly increase delay for smoother output
    
    process.stdout.close()
    process.wait()
    return output


# Main Streamlit Interface
st.title("Interactive Configuration Form")

# Ask for YAML save path, file name, and dataset name
st.write("### Save Options")
yaml_directory = st.text_input("Enter the directory to save the YAML file:")
yaml_filename = st.text_input("Enter the name of the YAML file to save (without extension):", "config")
dataset_name = st.text_input("Enter the dataset name:", "demo")

# Define the YAML path outside the submission block to avoid NameError
yaml_path = os.path.join(yaml_directory, f"{yaml_filename}.yaml")

# Create the form and collect data
with st.form("configuration_form"):
    st.write("### Configuration Options")
    form_data = create_form(DefaultConfiguration)
    
    submitted = st.form_submit_button("View Configuration")

if submitted:
    # Generate and display YAML content
    st.write("### Generated YAML Configuration")
    yaml_content = yaml.dump(form_data, sort_keys=False)
    st.code(yaml_content, language="yaml")

    # Save YAML to a file in the specified path
    try:
        with open(yaml_path, "w") as file:
            yaml.dump(form_data, file, sort_keys=False)
        st.success(f"Configuration saved to {yaml_path}")
    except Exception as e:
        st.error(f"Failed to save configuration: {e}")

# Terminal Output Section
st.write("### Terminal Output")
command = f"d3x dataset ingest -d {dataset_name} -c {yaml_path}"

if st.button("Run Command"):
    st.write(f"Running: `{command}`")
    terminal_placeholder = st.empty()
    output = run_command(command, terminal_placeholder)
    st.session_state['outputs'] = output
    st.success("Command execution completed.")

# Include styles and scripts for smooth scrolling and consistent UI
st.markdown(
    """
    <style>
    .stCodeBlock {
        max-height: 300px;
        overflow: auto;
        padding: 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)
