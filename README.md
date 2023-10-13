# FinancialSummarizer
This repository contains the code for Finanical Summarizer Problem Statement.

##Objective:
To conduct a comprehensive analysis to determine whether investing in a particular private bond in India is safe or not for an individual investor, leveraging the capabilities of Generative AI. ​

Analysis should be performed considering various inputs, such as:​

    Credit ratings from agencies like CRISIL​

    Financial statements, including income statements, balance sheets, and cash flow statements​

    Yearly and quarterly financial results​

    Townhall meeting summaries​

    Press releases by the company​

    Other relevant factors like market conditions and regulatory compliance​

The aim is to provide an AI driven solution that delivers a comprehensive summary to the investor to assist in the decision-making process

##Steps to run: 
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