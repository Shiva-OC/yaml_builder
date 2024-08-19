import streamlit as st
from pydantic import BaseModel, Field
from typing import List, Literal, Any, Dict
import yaml
import os
import subprocess

class DefaultConfiguration(BaseModel):
    dataset: str = Field("demodata", description="Enter the dataset name")
    generate_ground_truth: bool = Field(True, description="Generate ground truth?")
    questions_generator: Dict[str, Any] = Field({
        "prompt": "default",
        "num_questions_per_chunk": 1,
        "max_chunks": 100,
        "llm": "openai",
        "llmkey": "sk-xyz",
        "llm_url": "",
        "max_tokens": 2048
    }, description="Questions Generator Parameters")
    ground_truth: str = Field("/home/ground-truth_simple.csv", description="Path to ground truth")
    rag_configuration: str = Field("/home/simple_rag.yaml", description="RAG Configuration File Path")
    vectorstore: str = Field("weaviate_vectorstore", description="Vector Database")
    evaluator: List[str] = Field([
        "semantic_similarity_evaluator",
        "correctness_evaluator",
        "cross_vector_similarity_evaluator",
        "answer_relevancy_evaluator",
        "retrieval_evaluator"
    ], description="List of evaluators")
    semantic_similarity_evaluator: Dict[str, Any] = Field({
        "prompt": "default",
        "llm": "openai",
        "llmkey": "sk-xyz",
        "llm_url": "",
        "max_tokens": 2048
    }, description="Semantic Similarity Evaluator Parameters")
    correctness_evaluator: Dict[str, Any] = Field({
        "prompt": "default",
        "llm": "openai",
        "llmkey": "sk-xyz",
        "llm_url": "",
        "max_tokens": 2048
    }, description="Correctness Evaluator Parameters")
    cross_vector_similarity_evaluator: Dict[str, Any] = Field({
        "prompt": "default",
        "llm": "openai",
        "llmkey": "sk-xyz",
        "llm_url": "",
        "max_tokens": 2048
    }, description="Cross Vector Similarity Evaluator Parameters")
    answer_relevancy_evaluator: Dict[str, Any] = Field({
        "prompt": "default",
        "llm": "openai",
        "llmkey": "sk-xyz",
        "llm_url": "",
        "max_tokens": 2048
    }, description="Answer Relevancy Evaluator Parameters")
    retrieval_evaluator: Dict[str, Any] = Field({
        "weaviate_vectorstore": {
            "kind": "weaviate",
            "vectorstore_provider": "dkubex",
            "textkey": "paperchunks",
            "embedding_class": "HuggingFaceEmbedding",
            "embedding_model": "BAAI/bge-large-en-v1.5",
            "llmkey": "",
            "similarity_top_k": 3
        },
    }, description="Retrieval Evaluator Parameters")

    metrics: List[str] = Field([
        "mrr",
        "hit_rate"
    ],description="evaluation Metrics")

    context_relevancy_evaluator: Dict[str, Any] = Field({
        "llm": {
            "provider": "openai",
            "key": ""
        },
        "prompt": "default"
    }, description="Context Relevancy Evaluator Parameters")
    faithfullness_evaluator: Dict[str, Any] = Field({
        "llm": {
            "provider": "openai",
            "key": ""
        },
        "prompt": "default"
    }, description="Faithfulness Evaluator Parameters")
    tracking: Dict[str, str] = Field({
        "experiment": "dkubexfm-rag-evaluate"
    }, description="Tracking Parameters")

    platform: Literal['sky', 'ray'] = Field("ray", description="Select the platform")

    @staticmethod
    def get_parameters(selection: str) -> Dict[str, Any]:
        config = DefaultConfiguration()
        if selection == 'semantic_similarity_evaluator':
            return config.semantic_similarity_evaluator
        elif selection == 'correctness_evaluator':
            return config.correctness_evaluator
        elif selection == 'cross_vector_similarity_evaluator':
            return config.cross_vector_similarity_evaluator
        elif selection == 'answer_relevancy_evaluator':
            return config.answer_relevancy_evaluator
        elif selection == 'retrieval_evaluator':
            return config.retrieval_evaluator
        elif selection == 'context_relevancy_evaluator':
            return config.context_relevancy_evaluator
        elif selection == 'faithfullness_evaluator':
            return config.faithfullness_evaluator
        else:
            return {}







def create_form(model_class):
    form_data = {}
    llm_providers = ['huggingface', 'openai', 'sky', 'dkubex']

    yaml_directory = st.text_input("Enter the directory to save the YAML file:", "/home/oc/auto_home/streamlit_form_generator")
    yaml_file_name = st.text_input("Enter the name of the YAML file to save (without extension):", "demo_config")

    for field_name, field_info in model_class.__fields__.items():
        field_value = field_info.default
        field_description = field_info.field_info.description

        # Initialize the dictionary for nested fields
        if isinstance(field_value, dict):
            form_data[field_name] = {}

        if isinstance(field_value, str):
            form_data[field_name] = st.text_input(field_description, value=field_value, key=f"{field_name}")
        elif isinstance(field_value, bool):
            form_data[field_name] = st.checkbox(field_description, value=field_value, key=f"{field_name}")
        elif isinstance(field_value, list):
            selected_evaluators = st.multiselect(field_description, options=field_value, default=field_value, key=f"{field_name}")
            form_data[field_name] = selected_evaluators
            for idx, evaluator in enumerate(selected_evaluators):
                evaluator_params = model_class.get_parameters(evaluator)
                form_data[evaluator] = {}
                for key, param in evaluator_params.items():
                    unique_key = f"{evaluator}-{key}-{idx}"  # Ensure unique keys
                    if isinstance(param, bool):
                        form_data[evaluator][key] = st.checkbox(f"{evaluator.capitalize()} - {key}", value=param, key=unique_key)
                    elif isinstance(param, int):
                        form_data[evaluator][key] = st.number_input(f"{evaluator.capitalize()} - {key}", value=param, key=unique_key)
                    elif isinstance(param, str):
                        if key == 'llm' or key == 'provider':
                            form_data[evaluator][key] = st.selectbox(f"{evaluator.capitalize()} - {key}", options=llm_providers, index=llm_providers.index(param), key=unique_key)
                        elif key == 'metrics':
                            form_data[evaluator][key] = st.multiselect(f"{evaluator.capitalize()} - {key}", options=['mrr', 'hit_rate'], default=param, key=unique_key)
                        else:
                            form_data[evaluator][key] = st.text_input(f"{evaluator.capitalize()} - {key}", value=param, key=unique_key)
                    elif isinstance(param, list) and key == 'metrics':
                        form_data[evaluator][key] = st.multiselect(f"{evaluator.capitalize()} - {key}", options=['mrr', 'hit_rate'], default=param, key=unique_key)

        elif isinstance(field_value, dict):
            for key, param in field_value.items():
                unique_key = f"{field_name}-{key}"  # Ensure unique keys
                if isinstance(param, bool):
                    form_data[field_name][key] = st.checkbox(f"{field_name.capitalize()} - {key}", value=param, key=unique_key)
                elif isinstance(param, int):
                    form_data[field_name][key] = st.number_input(f"{field_name.capitalize()} - {key}", value=param, key=unique_key)
                elif isinstance(param, str):
                    if key == 'llm' or key == 'provider':
                        form_data[field_name][key] = st.selectbox(f"{field_name.capitalize()} - {key}", options=llm_providers, index=llm_providers.index(param), key=unique_key)
                    else:
                        form_data[field_name][key] = st.text_input(f"{field_name.capitalize()} - {key}", value=param, key=unique_key)
                elif isinstance(param, list) and key == 'metrics':
                    form_data[field_name][key] = st.multiselect(f"{field_name.capitalize()} - {key}", options=['mrr', 'hit_rate'], default=param, key=unique_key)

    return form_data, yaml_directory, yaml_file_name



def generate_yaml(form_data, yaml_directory, yaml_file_name):


        # Remove empty dictionaries for metrics like mrr and hit_rate
    if 'metrics' in form_data:
        form_data = {key: value for key, value in form_data.items() if value not in [{}, []]}
    
    # Ensure that empty metric fields like mrr and hit_rate are not included
    for metric in ['mrr', 'hit_rate']:
        if metric in form_data:
            del form_data[metric]

    yaml_path = os.path.join(yaml_directory, f"{yaml_file_name}.yaml")
    with open(yaml_path, 'w') as yaml_file:
        yaml.dump(form_data, yaml_file, sort_keys=False)
    return yaml_path

def execute_yaml_command(yaml_path, dataset_name):
    command = f"d3x dataset ingest -d {dataset_name} --config {yaml_path}"
    result = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in result.stdout:
        st.text(line.decode().strip())
    result.wait()
    return command


def run_app():
    st.title("YAML Configuration and Execution App")

    if "state" not in st.session_state:
        st.session_state.state = "input"
        st.session_state.dataset_name = "demodata"  # Initialize dataset_name in session state

    if st.session_state.state == "input":
        form_data, yaml_directory, yaml_file_name = create_form(DefaultConfiguration)
        st.session_state.dataset_name = st.text_input("Enter the dataset name:", value=st.session_state.dataset_name)

        if st.button("Generate and View Configuration"):
            yaml_path = generate_yaml(form_data, yaml_directory, yaml_file_name)
            st.session_state.yaml_path = yaml_path
            st.session_state.state = "view_config"
            st.rerun()

    elif st.session_state.state == "view_config":
        with open(st.session_state.yaml_path, 'r') as yaml_file:
            yaml_content = yaml_file.read()
        st.code(yaml_content, language='yaml', line_numbers=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Edit Configuration"):
                st.session_state.state = "input"
                st.rerun()
        with col2:
            if st.button("Run Command"):
                st.session_state.state = "run_command"
                st.rerun()

    elif st.session_state.state == "run_command":
        if st.session_state.dataset_name:
            command = execute_yaml_command(st.session_state.yaml_path, st.session_state.dataset_name)
            st.session_state.command = command
            st.session_state.state = "command_complete"
            st.rerun()
        else:
            st.error("Dataset name is required to run the command.")

    elif st.session_state.state == "command_complete":
        st.text(f"Command executed: {st.session_state.command}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Edit Configuration"):
                st.session_state.state = "input"
                st.rerun()
        with col2:
            if st.button("Run Command Again"):
                st.session_state.state = "run_command"
                st.rerun()

if __name__ == "__main__":
    run_app()
