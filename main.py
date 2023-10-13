#Intrgrate code with OpenAI API
import os
from langchain.llms import OpenAI
import streamlit as st
import os
import langchain
import pypdf
import unstructured 
import utils
from langchain.document_loaders import MergedDataLoader
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import customscore
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
import matplotlib.pyplot as plt
from langchain.tools.yahoo_finance_news import YahooFinanceNewsTool

# import pandas lib as pd
import pandas as pd


load_dotenv()


st.title("Financial Summarizer")

# input_text = st.text_input("Enter the Bond Name here")
input_text=st.selectbox(

    'Select the bond name',

    ('Shriram Finance','L&T Fin.Holdings','Coforge'))

indicators_defaults = {

        "Credit Ratings":20,
        "Debt to Equity":20,
        "Interest Coverage": 10,
        "Liquidity Ratio": 10,
        "Profit Margin": 10,
        "Revenue Growth": 10,
        "Management Quality": 10,
        "Legal Compliance": 10,
   
        
        }

company_tickers={
    'Shriram Finance':'SHRIRAMFIN.NS',
    'L&T Fin.Holdings':'L&TFH.NS',
    'Coforge':'COFORGE.NS'


}
risk_score_calculator = customscore.RiskScore()
method_mapping = {

        "Credit Ratings":risk_score_calculator.credit_ratings,
        "Debt to Equity":risk_score_calculator.debt_to_equity,
        "Interest Coverage": risk_score_calculator.interest_coverage,
        "Liquidity Ratio": risk_score_calculator.liquidity_ratio,
        "Profit Margin": risk_score_calculator.profit_margin,
        "Revenue Growth": risk_score_calculator.revenue_growth,
        "Management Quality": risk_score_calculator.management_quality,
        "Legal Compliance": risk_score_calculator.legal_compliance,
   
        
        }
selected_weights = {}
st.sidebar.subheader("Risk Assessment Parameters")
# Iterate over indicators and their default values
for indicator, default_value in indicators_defaults.items():
    st.sidebar.subheader(indicator)
    # Create a slider for each indicator
    selected_weights[indicator] = st.sidebar.slider(
        f"Weights {indicator} (0-100)",
        min_value=0,
        max_value=100,
        value=default_value
    )

if st.button('Submit'):
    tab1, tab2 , tab3 = st.tabs(["Summary", "Charts", "List of Bonds"])
    with tab1:
        st.subheader("Issuer Details")
        import yfinance as yf
        company = yf.Ticker(company_tickers[input_text])
        dict_info =  company.info
    
        col1, col2 = st.columns(2)

        # Display data in a structured format
        with col1:
            st.markdown(f"**Sector:** {dict_info['sector']}")
            st.markdown(f"**Dividend Yield:** {round(dict_info['dividendYield']*100,2)} %")
            st.markdown(f"**Market Cap:** {round(dict_info['marketCap'] / 1e9, 2)} Billion")

            
        with col2:
            pe_ratio = dict_info.get('trailingPE')
            st.markdown(f"**PE Ratio:** {pe_ratio if pe_ratio is not None else 'N/A'}")
            st.markdown(f"**Current Price:** {dict_info['currentPrice']}")
            st.markdown(f"**Previous Close:** {dict_info['previousClose']}")
    with tab2:   
        st.subheader("Data Visualizations")
        df = utils.profit_loss_azure(input_text) 
        st.line_chart(df['Net profit'])
        st.line_chart(df['Sales'])
        # fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 9))


        # axs[0].plot(df['Net profit'])
        # axs[1].plot(df['Sales'])


        # axs[0].set_xlabel('Report Date')
        # axs[0].set_ylabel('Net Profit')
        # axs[0].set_title('Net Profit')

        # axs[1].set_xlabel('Report Date')
        # axs[1].set_ylabel('Sales')
        # axs[1].set_title('Sales')
        # st.pyplot(fig)

    with tab3:
        df = utils.get_bonds(input_text)
        st.table(df)
    
    total_weights = sum(selected_weights.values())
    # st.write(dict["trailingPE"])
    if total_weights == 100: 
        st.subheader("Bond Safety Analysis")
        assessment_results = []
        table_data=[]
        table_placeholder = st.empty()
        # Iterate over the sub-categories, calling the corresponding method for each
        for sub_category, weight in selected_weights.items():
            method = method_mapping[sub_category]
            score, explanation = method(input_text,weight)
            row = {
                "Category": sub_category,
                "Weight": weight,
                "Score": score,
                "Explanation": explanation,
            }
            table_data.append(row)
            df = pd.DataFrame(table_data)
            df.index = df.index + 1

            # Update the table in the app with the new DataFrame
            table_placeholder.table(df)

        overall_risk_score = risk_score_calculator.calculate_overall_risk_score()
        st.write(f"Overall Risk Score (1 - High Risk, 10 - Low Risk): {overall_risk_score}")
        if(overall_risk_score>=7):
              st.write(f"It is safe to invest in the bond")
        else:
             st.write(f"It is not safe to invest in the bond")  
        
       
      
    else:
        st.write("The total weights must sum up to 100. Please adjust the weights accordingly")
        


