# FinancialSummarizer
This repository contains the code for Finanical Summarizer Problem Statement.


Steps to run: 
1. Create a new Conda Environment
conda create -p finenv python=3.9 -y

2. Activate the Environment
conda activate finenv/

3. Install the packages in Environmnet
pip install -r  requirements.txt

4. Create .env file which would contain the keys
OPENAI_API_KEY 
OPENAI_API_TYPE 
OPENAI_API_BASE 
OPENAI_API_VERSION 
vector_store_address 
vector_store_password 
STORAGEACCOUNTURL
STORAGEACCOUNTKEY

5. To run the application
streamlit run main.py