# Create the .streamlit directory if it doesn't exist
mkdir -p .streamlit/

# Create or overwrite the config.toml file with server settings
echo "[server]" > .streamlit/config.toml
echo "baseUrlPath = \"$INGRESS_PREFIX\"" >> .streamlit/config.toml

# Append the theme settings to the config.toml file
echo "[theme]" >> .streamlit/config.toml
echo "base = \"dark\"" >> .streamlit/config.toml

# Run the Streamlit application
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
