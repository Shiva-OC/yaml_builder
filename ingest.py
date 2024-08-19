import streamlit as st
from pydantic import BaseModel, Field
from typing import List, Literal, Any, Dict, get_args
import yaml
import os
import subprocess
import time

# Step 1: Define the Pydantic Model with Instructions
class DefaultConfiguration(BaseModel):
    splitter: Literal['sentence_text_splitter_LC', 'token_text_splitter', 'sentence_text_splitter', 'semantic_splitter'] = Field(
        'sentence_text_splitter_LC', description="Select Splitter (Options: sentence_text_splitter_LC, token_text_splitter, sentence_text_splitter, semantic_splitter)"
    )
    embedding: Literal['huggingface', 'openai', 'sky', 'dkubex'] = Field(
        'huggingface', description="Select Embedding (Options: huggingface, openai, sky, dkubex)"
    )
    metadata: List[str] = Field(
        [], description="Select Metadata Options (Options: default, custom)"
    )
    reader: List[str] = Field(
        [], description="Select Reader (Options: file, directoryreader, scrapeddatareader, scrapyreader, sharepointreader, githubreader)"
    )
    adjacent_chunks: bool = Field(True, description="Select whether adjacent chunks should be processed.")
    
    # Parameters for selected splitters
    sentence_text_splitter_LC: Dict[str, Any] = Field({
        "chunk_size": 256,
        "chunk_overlap": 0
    }, description="Sentence Text Splitter LC Parameters")

    sentence_text_splitter: Dict[str, Any] = Field({
        "chunk_size": 256,
        "chunk_overlap": 0
    }, description="Sentence Text Splitter Parameters")

    token_text_splitter: Dict[str, Any] = Field({
        "chunk_size": 256,
        "chunk_overlap": 0
    }, description="Token Text Splitter Parameters")

    semantic_splitter: Dict[str, Any] = Field({
        "buffer_size": 1,
        "breakpoint_percentile_threshold": 95
    }, description="Semantic Splitter Parameters")

    # Embedding-specific configurations
    huggingface: Dict[str, Any] = Field({
        "model": "BAAI/bge-large-en-v1.5"
    }, description="Huggingface Parameters")

    openai: Dict[str, Any] = Field({
        "model": "text-embedding-ada-002",
        "llmkey": "sk-proj-nsl-HiNr**********************"
    }, description="OpenAI Parameters")

    dkubex: Dict[str, Any] = Field({
        "embedding_key": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
        "embedding_url": "https://dkubex-url",
        "batch_size": 10
    }, description="Dkubex Parameters")

    sky: Dict[str, Any] = Field({
        "embedding_key": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
        "embedding_url": "http://35.171.182.10:30003/v1/",
        "batch_size": 10
    }, description="Sky Parameters")

    custom_metadata: Dict[str, Any] = Field({
        "adjacent_chunks": False,
        "extractor_path": "/path/to/extractor.py"
    }, description="Custom Metadata Configuration")

    file_reader: Dict[str, Any] = Field({
        "inputs": {
            "loader_args": {
                "input_dir": "/home/data/input",
                "recursive": True,
                "exclude_hidden": True,
                "raise_on_error": True
            }
        }
    }, description="File Reader Parameters")

    directory_reader: Dict[str, Any] = Field({
        "inputs": {
            "loader_args": {
                "input_dir": "/home/ocdlgit/",
                "recursive": True,
                "exclude_hidden": True,
                "raise_on_error": True
            }
        }
    }, description="Directory Reader Parameters")

    scrapeddata_reader: Dict[str, Any] = Field({
        "inputs": {
            "loader_args": {
                "input_dir": "/home/ocdlgit/raw/raw_external",
                "recursive": True,
                "exclude_hidden": True,
                "raise_on_error": True
            }
        }
    }, description="Scraped Data Reader Parameters")

    scrapyreader: Dict[str, Any] = Field({
        "inputs": {
            "loader_args": {
                "spiders": {"demo": {"url": "https://company.getinsured.com/..."}}
            }
        }
    }, description="Scrapy Reader Parameters")

    sharepointreader: Dict[str, Any] = Field({
        "inputs": {
            "loader_args": {
                "client_id": "",
                "client_secret": "",
                "tenant_id": "",
                "sharepoint_site_name": "dkubexfmrag",
                "doc_source": "demo",
                "state_category": "NV",
                "designation_category": "Consumer",
                "topic_category": "General",
                "sharepoint_folder_path": "test",
                "recursive": True
            }
        }
    }, description="Demo SharePoint Reader Parameters")

    githubreader: Dict[str, Any] = Field({
        "inputs": {
            "loader_args": {
                "input_dir": "",
                "include_file_ext": "",
                "github_token": "",
                "owner": "",
                "repo": "",
                "verbose": False,
                "concurrent_requests": 5,
                "timeout": 30
            },
            "recursive": True,
            "exclude_hidden": True,
            "raise_on_error": True
        },
        "data_args": {
            "doc_source": "demo"
        }
    }, description="GitHub Reader Parameters")

    mlflow: Dict[str, Any] = Field({
        "experiment": "Demo-Ingestion"
    }, description="MLFlow Parameters")

    @staticmethod
    def get_parameters(selection: str) -> Dict[str, Any]:
        config = DefaultConfiguration()
        if selection == 'sentence_text_splitter_LC':
            return config.sentence_text_splitter_LC
        elif selection == 'sentence_text_splitter':
            return config.sentence_text_splitter
        elif selection == 'token_text_splitter':
            return config.token_text_splitter
        elif selection == 'semantic_splitter':
            return config.semantic_splitter
        elif selection == 'huggingface':
            return config.huggingface
        elif selection == 'openai':
            return config.openai
        elif selection == 'dkubex':
            return config.dkubex
        elif selection == 'sky':
            return config.sky
        elif selection == 'custom':
            return config.custom_metadata
        elif selection == 'file':
            return config.file_reader
        elif selection == 'directoryreader':
            return config.directory_reader
        elif selection == 'scrapeddatareader':
            return config.scrapeddata_reader
        elif selection == 'scrapyreader':
            return config.scrapyreader
        elif selection == 'sharepointreader':
            return config.sharepointreader
        elif selection == 'githubreader':
            return config.githubreader
        elif selection == 'mlflow':
            return config.mlflow
        else:
            return {}

# Step 2: Create the Form
def create_form(model_class):
    form_data = {}

    # Fields for YAML details
    yaml_directory = st.text_input("Enter the directory to save the YAML file:", "/home/oc/auto_home/streamlit_form_generator")
    yaml_file_name = st.text_input("Enter the name of the YAML file to save (without extension):", "demo_config")
    dataset_name = st.text_input("Enter the dataset name:", "demodata")

    # Iterate over the fields in the model class
    for field_name, field_info in model_class.__fields__.items():
        field_value = field_info.default
        field_description = field_info.field_info.description

        # Splitter and Embedding Selectboxes
        if field_name in ['splitter', 'embedding']:
            selected_option = st.selectbox(field_description, options=get_args(field_info.type_), index=get_args(field_info.type_).index(field_value))
            form_data[field_name] = selected_option

            selected_params = model_class.get_parameters(selected_option)
            form_data[selected_option] = {}  # Create section for selected option
            for key, param in selected_params.items():
                if isinstance(param, bool):
                    form_data[selected_option][key] = st.checkbox(f"{field_name.capitalize()} - {key}", value=param)
                elif isinstance(param, int):
                    form_data[selected_option][key] = st.number_input(f"{field_name.capitalize()} - {key}", value=param)
                elif isinstance(param, str):
                    form_data[selected_option][key] = st.text_input(f"{field_name.capitalize()} - {key}", value=param)

        # Metadata Multiselect
        elif field_name == 'metadata':
            selected_metadata = st.multiselect(field_description, options=['default', 'custom'], default='default')
            form_data[field_name] = selected_metadata

            for meta in selected_metadata:
                metadata_params = model_class.get_parameters(meta)
                form_data[meta] = {}
                for key, param in metadata_params.items():
                    if isinstance(param, bool):
                        form_data[meta][key] = st.checkbox(f"{meta.capitalize()} - {key}", value=param)
                    elif isinstance(param, int):
                        form_data[meta][key] = st.number_input(f"{meta.capitalize()} - {key}", value=param)
                    elif isinstance(param, str):
                        form_data[meta][key] = st.text_input(f"{meta.capitalize()} - {key}", value=param)

        # Reader Multiselect
        elif field_name == 'reader':
            selected_readers = st.multiselect(field_description, options=['file', 'directoryreader', 'scrapeddatareader', 'scrapyreader', 'sharepointreader','githubreader'], default='file')
            form_data[field_name] = selected_readers

            for reader in selected_readers:
                reader_params = model_class.get_parameters(reader)
                form_data[reader] = {"inputs": {"loader_args": {}}}  # Create nested dictionary for inputs and loader_args
                for key, param in reader_params['inputs']['loader_args'].items():
                    unique_key = f"{reader}_{key}"
                    if isinstance(param, bool):
                        form_data[reader]["inputs"]["loader_args"][key] = st.checkbox(f"{reader.capitalize()} - {key}", value=param, key=unique_key)
                    elif isinstance(param, int):
                        form_data[reader]["inputs"]["loader_args"][key] = st.number_input(f"{reader.capitalize()} - {key}", value=param, key=unique_key)
                    elif isinstance(param, str):
                        form_data[reader]["inputs"]["loader_args"][key] = st.text_input(f"{reader.capitalize()} - {key}", value=param, key=unique_key)

        # Boolean field
        elif isinstance(field_value, bool):
            form_data[field_name] = st.checkbox(field_description, value=field_value)

    return form_data, yaml_directory, yaml_file_name, dataset_name

# Step 3: Handle YAML Generation and Execution
def generate_yaml(form_data, yaml_directory, yaml_file_name, dataset_name):
    yaml_path = os.path.join(yaml_directory, f"{yaml_file_name}.yaml")
    # form_data['dataset_name'] = dataset_name  # Add dataset name to YAML
    with open(yaml_path, 'w') as yaml_file:
        yaml.dump(form_data, yaml_file, sort_keys=False)
    return yaml_path

def execute_yaml_command(yaml_path,dataset_name):
    command = f"d3x dataset ingets -d {dataset_name} --config {yaml_path}"
    result = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in result.stdout:
        st.text(line.decode().strip())
    result.wait()
    return command


def run_app():
    st.title("YAML Configuration and Execution App")

    if "state" not in st.session_state:
        st.session_state.state = "input"

    if st.session_state.state == "input":
        form_data, yaml_directory, yaml_file_name, dataset_name = create_form(DefaultConfiguration)

        if st.button("Generate and View Configuration"):
            yaml_path = generate_yaml(form_data, yaml_directory, yaml_file_name, dataset_name)
            st.session_state.yaml_path = yaml_path
            st.session_state.dataset_name = dataset_name  # Initialize dataset_name in session state
            st.session_state.state = "view_config"
            st.rerun()  # Force rerun to apply state change

    elif st.session_state.state == "view_config":
        with open(st.session_state.yaml_path, 'r') as yaml_file:
            yaml_content = yaml_file.read()
        st.code(yaml_content, language='yaml', line_numbers=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Edit Configuration"):
                st.session_state.state = "input"
                st.rerun()  # Force rerun to apply state change

        with col2:
            if st.button("Run Configuration"):
                st.session_state.state = "running_command"
                st.rerun()  # Force rerun to apply state change

    elif st.session_state.state == "running_command":
        command = execute_yaml_command(st.session_state.yaml_path, st.session_state.dataset_name)
        st.text(f"Running command: {command}")
        st.session_state.state = "view_config"
        st.rerun()  # Force rerun to return to view_config state after running the command


run_app()
