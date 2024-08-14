import streamlit as st
from pydantic import BaseModel, Field
from typing import List, Literal, Any, Dict, get_args
import yaml
import subprocess
import time
import os

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
        [], description="Select Reader (Options: file, directoryreader, scrapeddatareader, scrapyreader, sharepointreader,githubreader)"
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
        "llmkey": "skAQNTCqZrXSQYT3BlbkFJUMwYprxRrYRfH_IQEeALnS0qK9VbfLB5rJR_6eFKnx28JP1HH0aqb0vRUA"
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


def create_form(model_class):
    form_data = {}

    # Fields for YAML details
    yaml_directory = st.text_input("Enter the directory to save the YAML file:", "/home/data")
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
                    form_data[meta][key] = st.text_input(f"Metadata - {key}", value=param)

        # Reader Multiselect with Nested Inputs
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

        # Adjacent Chunks Checkbox
        elif field_name == 'adjacent_chunks':
            form_data[field_name] = st.checkbox(field_description, value=field_value)

        # MLFlow Parameters
        elif field_name == 'mlflow':
            mlflow_params = model_class.get_parameters('mlflow')
            form_data[field_name] = {key: st.text_input(f"MLFlow - {key}", value=param_value) for key, param_value in mlflow_params.items()}
    
    # Collect YAML path details
    yaml_path = os.path.join(yaml_directory, f"{yaml_file_name}.yaml")
    return form_data, yaml_path, dataset_name

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

# Create the form and collect data
with st.form("configuration_form"):
    st.write("### Configuration Options")
    form_data, yaml_path, dataset_name = create_form(DefaultConfiguration)
    
    submitted = st.form_submit_button("Generate YAML and View Configuration")

if submitted:
    # Generate and display YAML content
    st.write("### Generated YAML Configuration")
    yaml_data = {
        "splitter": form_data['splitter'],
        "embedding": form_data['embedding'],
        "metadata": form_data['metadata'],
        "reader": form_data['reader'],
        "adjacent_chunks": form_data['adjacent_chunks'],
        form_data['splitter']: form_data.get(form_data['splitter'], {}),
        form_data['embedding']: form_data.get(form_data['embedding'], {}),
        "mlflow": form_data['mlflow'],
    }

    # Adding subsections for readers
    for reader in form_data['reader']:
        yaml_data[reader] = form_data.get(reader, {})

    # Display the YAML content
    yaml_content = yaml.dump(yaml_data, sort_keys=False, default_flow_style=False)
    st.code(yaml_content, language="yaml")

    # Save YAML to a file in the specified path
    try:
        with open(yaml_path, "w") as yaml_file:
            yaml.dump(yaml_data, yaml_file, sort_keys=False, default_flow_style=False)
        st.success(f"YAML configuration saved to: {yaml_path}")
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





#---------------------------------------------------------------------------------------------------------------------------------------


# import streamlit as st
# from pydantic import BaseModel, Field
# from typing import List, Literal, Any, Dict, get_args
# import yaml
# import subprocess
# import time
# import os

# # Step 1: Define the Pydantic Model with Instructions
# class DefaultConfiguration(BaseModel):
#     splitter: Literal['token_text_splitter', 'semantic_splitter', 'sentence_text_splitter', 'sentence_text_splitter_LC'] = Field(
#         'sentence_text_splitter_LC', description="Select Splitter (Options: token_text_splitter, semantic_splitter, sentence_text_splitter, sentence_text_splitter_LC)"
#     )
#     embedding: Literal['openai', 'huggingface', 'sky', 'dkubex'] = Field(
#         'huggingface', description="Select Embedding (Options: openai, huggingface, sky, dkubex)"
#     )
#     metadata: List[Literal['default', 'custom']] = Field(
#         ['default'], description="Select Metadata Options (Options: default, custom)"
#     )
#     reader: List[Literal['file', 'directoryreader', 'scrapeddatareader', 'scrapyreader', 'pyreader', 'githubreader', 'sharepointreader']] = Field(
#         ['scrapyreader'], description="Select Reader (Options: file, directoryreader, scrapeddatareader, scrapyreader, pyreader, githubreader, sharepointreader)"
#     )
#     adjacent_chunks: bool = Field(True, description="Select whether adjacent chunks should be processed.")

#     sentence_text_splitter: Dict[str, Any] = Field({
#         "chunk_size": {"type": "slider", "value": 256, "min": 0, "max": 1024},
#         "chunk_overlap": {"type": "slider", "value": 0, "min": 0, "max": 1024}
#     }, description="Sentence Text Splitter Parameters (chunk_size: slider, chunk_overlap: slider)")

#     sentence_text_splitter_LC: Dict[str, Any] = Field({
#         "chunk_size": {"type": "slider", "value": 256, "min": 0, "max": 1024},
#         "chunk_overlap": {"type": "slider", "value": 0, "min": 0, "max": 1024}
#     }, description="Sentence Text Splitter LC Parameters (chunk_size: slider, chunk_overlap: slider)")

#     token_text_splitter: Dict[str, Any] = Field({
#         "chunk_size": {"type": "slider", "value": 256, "min": 0, "max": 2048},
#         "chunk_overlap": {"type": "slider", "value": 10, "min": 1, "max": 10}
#     }, description="Token Text Splitter Parameters (chunk_size: slider, chunk_overlap: slider)")

#     huggingface: Dict[str, Any] = Field({
#         "model": {"type": "text_input", "value": "BAAI/bge-large-en-v1.5"}
#     }, description="Huggingface Parameters (model: text_input)")

#     openai: Dict[str, Any] = Field({
#         "model": {"type": "text_input", "value": "text-embedding-ada-002"},
#         "llmkey": {"type": "text_input", "value": "<llmkey>"}
#     }, description="OpenAI Parameters (model: text_input, llmkey: text_input)")

#     dkubex: Dict[str, Any] = Field({
#         "embedding_key": {"type": "text_input", "value": "<embedding_key>"},
#         "embedding_url": {"type": "text_input", "value": "<embedding_url>"},
#         "batch_size": {"type": "number_input", "value": 10, "min": 1}
#     }, description="Dkubex Parameters (embedding_key: text_input, embedding_url: text_input, batch_size: number_input)")

#     sky: Dict[str, Any] = Field({
#         "embedding_key": {"type": "text_input", "value": "<embedding_key>"},
#         "embedding_url": {"type": "text_input", "value": "<embedding_url>"},
#         "batch_size": {"type": "number_input", "value": 10, "min": 1}
#     }, description="Sky Parameters (embedding_key: text_input, embedding_url: text_input, batch_size: number_input)")

#     mlflow: Dict[str, Any] = Field({
#         "experiment": {"type": "text_input", "value": "GI-ingestion"}
#     }, description="MLFlow Parameters (experiment: text_input)")

#     scrapyreader: Dict[str, Any] = Field({
#         "inputs": {"type": "text_input", "value": ""},
#         "data_args": {"spiders": {"gi": {"url": "https://company.getinsured.com/state-based-marketplaces/state-based-marketplace-resources/"}}}
#     }, description="Scrapy Reader Parameters (inputs: text_input, data_args: dict)")

#     @staticmethod
#     def get_parameters(selection: str) -> Dict[str, Any]:
#         if selection == 'sentence_text_splitter_LC':
#             return DefaultConfiguration().sentence_text_splitter_LC
#         elif selection == 'sentence_text_splitter':
#             return DefaultConfiguration().sentence_text_splitter
#         elif selection == 'token_text_splitter':
#             return DefaultConfiguration().token_text_splitter
#         elif selection == 'huggingface':
#             return DefaultConfiguration().huggingface
#         elif selection == 'openai':
#             return DefaultConfiguration().openai
#         elif selection == 'dkubex':
#             return DefaultConfiguration().dkubex
#         elif selection == 'sky':
#             return DefaultConfiguration().sky
#         elif selection == 'mlflow':
#             return DefaultConfiguration().mlflow
#         elif selection == 'scrapyreader':
#             return DefaultConfiguration().scrapyreader
#         else:
#             return {}

# # Step 2: Create the Form and Handle Submission
# def create_form(model_class):
#     form_data = {}

#     for field_name, field_info in model_class.__fields__.items():
#         field_value = field_info.default
#         field_description = field_info.field_info.description

#         if get_args(field_info.type_):
#             options = get_args(field_info.type_)
#             if isinstance(field_value, list):
#                 selected_options = st.multiselect(field_description, options=options, default=field_value, key=f"{field_name}")
#                 form_data[field_name] = selected_options
#             else:
#                 selected_index = options.index(field_value) if field_value in options else 0
#                 selected_option = st.selectbox(field_description, options=options, index=selected_index, key=f"{field_name}")
#                 form_data[field_name] = selected_option

#                 if field_name in ["splitter", "embedding"]:
#                     extra_params = model_class.get_parameters(selected_option)
#                     for sub_field, sub_field_info in extra_params.items():
#                         if "type" not in sub_field_info:
#                             continue  # Skip if 'type' key is not present
#                         input_type = sub_field_info["type"]
#                         if input_type == "slider":
#                             form_data[sub_field] = st.slider(sub_field.capitalize().replace("_", " "), 
#                                                              min_value=sub_field_info["min"], 
#                                                              max_value=sub_field_info["max"], 
#                                                              value=sub_field_info["value"], 
#                                                              key=f"{field_name}_{sub_field}")
#                         elif input_type == "text_input":
#                             form_data[sub_field] = st.text_input(sub_field.capitalize().replace("_", " "), 
#                                                                  value=sub_field_info["value"], 
#                                                                  key=f"{field_name}_{sub_field}")
#                         elif input_type == "number_input":
#                             form_data[sub_field] = st.number_input(sub_field.capitalize().replace("_", " "), 
#                                                                    min_value=sub_field_info.get("min", None), 
#                                                                    value=sub_field_info["value"], 
#                                                                    key=f"{field_name}_{sub_field}")

#         elif isinstance(field_value, dict) and field_name not in ["splitter", "embedding"]:
#             for sub_field, sub_field_info in field_value.items():
#                 if "type" not in sub_field_info:
#                     continue  # Skip if 'type' key is not present
#                 input_type = sub_field_info["type"]
#                 if input_type == "slider":
#                     form_data[sub_field] = st.slider(sub_field.capitalize().replace("_", " "), 
#                                                      min_value=sub_field_info["min"], 
#                                                      max_value=sub_field_info["max"], 
#                                                      value=sub_field_info["value"], 
#                                                      key=f"{field_name}_{sub_field}")
#                 elif input_type == "text_input":
#                     form_data[sub_field] = st.text_input(sub_field.capitalize().replace("_", " "), 
#                                                          value=sub_field_info["value"], 
#                                                          key=f"{field_name}_{sub_field}")
#                 elif input_type == "number_input":
#                     form_data[sub_field] = st.number_input(sub_field.capitalize().replace("_", " "), 
#                                                            min_value=sub_field_info.get("min", None), 
#                                                            value=sub_field_info["value"], 
#                                                            key=f"{field_name}_{sub_field}")
#         else:
#             form_data[field_name] = st.checkbox(field_description, value=field_value, key=f"{field_name}")
    
#     return form_data


# def run_command(command, placeholder):
#     process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
#     output = ""
    
#     for line in iter(process.stdout.readline, ''):
#         output += line
#         st.session_state['output'] = output.replace('\n', '<br>')
#         placeholder.markdown(
#             f"<div style='background-color: #1a1c24; color: #ffffff; font-family: monospace; white-space: pre-wrap; height: 300px; width: 100%; overflow-y: auto; padding: 10px; border-radius: 10px;'>{st.session_state['output']}</div><div id='end-of-output'></div>",
#             unsafe_allow_html=True
#         )
#         scroll_js = """
#         <script>
#         var element = document.getElementById("end-of-output");
#         element.scrollIntoView({behavior: "smooth"});
#         </script>
#         """
#         st.markdown(scroll_js, unsafe_allow_html=True)
#         time.sleep(0.2)  # Slightly increase delay for smoother output
    
#     process.stdout.close()
#     process.wait()
#     return output


# # Main Streamlit Interface
# st.title("Interactive Configuration Form")

# # Ask for YAML save path, file name, and dataset name
# st.write("### Save Options")
# yaml_directory = st.text_input("Enter the directory to save the YAML file:")
# yaml_filename = st.text_input("Enter the name of the YAML file to save (without extension):", "config")
# dataset_name = st.text_input("Enter the dataset name:", "demo")

# # Define the YAML path outside the submission block to avoid NameError
# yaml_path = os.path.join(yaml_directory, f"{yaml_filename}.yaml")

# # Create the form and collect data
# with st.form("configuration_form"):
#     st.write("### Configuration Options")
#     form_data = create_form(DefaultConfiguration)
    
#     submitted = st.form_submit_button("View Configuration")

# if submitted:
#     # Generate and display YAML content
#     st.write("### Generated YAML Configuration")
#     yaml_content = yaml.dump(form_data, sort_keys=False)
#     st.code(yaml_content, language="yaml")

#     # Save YAML to a file in the specified path
#     try:
#         with open(yaml_path, "w") as file:
#             yaml.dump(form_data, file, sort_keys=False)
#         st.success(f"Configuration saved to {yaml_path}")
#     except Exception as e:
#         st.error(f"Failed to save configuration: {e}")

# # Terminal Output Section
# st.write("### Terminal Output")
# command = f"d3x dataset ingest -d {dataset_name} -c {yaml_path}"

# if st.button("Run Command"):
#     st.write(f"Running: `{command}`")
#     terminal_placeholder = st.empty()
#     output = run_command(command, terminal_placeholder)
#     st.session_state['outputs'] = output
#     st.success("Command execution completed.")

# # Include styles and scripts for smooth scrolling and consistent UI
# st.markdown(
#     """
#     <style>
#     .stCodeBlock {
#         max-height: 300px;
#         overflow: auto;
#         padding: 0;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )
