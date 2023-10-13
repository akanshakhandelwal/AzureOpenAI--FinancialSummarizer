import langchain
import os
import pandas as pd
def load_document(file):
    
    print(file)
    name, extension = os.path.splitext(file)

    
    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension=='.xlsx':
        from langchain.document_loaders import UnstructuredExcelLoader
        print(f'Loading {file}')
        loader=UnstructuredExcelLoader(file,mode="elements")
    elif extension=='.csv':
        from langchain.document_loaders.csv_loader import CSVLoader
        print(f'Loading {file}')
        loader=CSVLoader(file)
    else:
        print('Document format is not supported!')
        return None
    #data = loader.load()
    return loader

def chunk_data(data, chunk_size=500):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    return chunks

def print_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    print(f'Total Tokens: {total_tokens}')
    print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}')


def insert_or_fetch_embeddings(index_name,chunks):
    import pinecone
    from langchain.vectorstores import Pinecone
    from langchain.embeddings.openai import OpenAIEmbeddings
    
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
    
    pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment=os.getenv('PINECONE_ENVIRONMENT'))
    
    
    if index_name in pinecone.list_indexes():
        print(f'Index {index_name} already exists. Loading embeddings ... ', end='')
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
        print('Ok')
    else:
        print(f'Creating index {index_name} and embeddings ...', end='')
        pinecone.create_index(index_name, dimension=1536, metric='cosine')
        vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
        print('Ok')
        
    return vector_store

def delete_pinecone_index(index_name='all'):
    import pinecone
    pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment=os.getenv('PINECONE_ENVIRONMENT'))
    
    
    if index_name == 'all':
        indexes = pinecone.list_indexes()
        print('Deleting all indexes ... ')
        for index in indexes:
            pinecone.delete_index(index)
        print('Ok')
    else:
        print(f'Deleting index {index_name} ...', end='')
        pinecone.delete_index(index_name)
        print('Ok')

def ask_and_get_answer(vector_store, q):
    from langchain.chains import ConversationalRetrievalChain,RetrievalQA,create_qa_with_sources_chain,StuffDocumentsChain
    from langchain.chat_models import ChatOpenAI
    from langchain.chat_models.azure_openai import AzureChatOpenAI
    from langchain.prompts import (PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
    prompt_template = """You must only return a numeric score and an explanation for each risk parameter for the company in the question.Return a json with two keys: 'score', containing a numeric value between 0 to 10, and 'explanation', containing a detailed analysis. Please do not add any code, single or double quotes in the explanation.
    {context}
    Question: {question}
    Answer: {{"score": numeric_value, "source":"source documents","explanation": "brief_explanation"}}"""


    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}

    llm = AzureChatOpenAI(deployment_name = 'gpt35',model_name='gpt-35-turbo', temperature=0.2,openai_api_key=os.getenv('OPENAI_API_KEY'))

    #retriever = vector_store.as_retriever()

    #chain = ConversationalRetrievalChain.from_llm(llm=llm,retriever=retriever,combine_docs_chain_kwargs={'prompt': qa_prompt})
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3},include_metadata=True, metadata_key = 'source')

    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,return_source_documents=True,chain_type_kwargs=chain_type_kwargs)
    
    
    answer = chain(q)
    return answer

def ask_and_get_answer_yfinance(q):
    from langchain.agents import initialize_agent, AgentType
    from langchain.chat_models.azure_openai import AzureChatOpenAI
    import yfinance as yf
    from langchain.tools.yahoo_finance_news import YahooFinanceNewsTool
    from langchain.prompts import (PromptTemplate,
                                   
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
    prompt_template = """You must only return a answer for the company in the question.Return Only the explanation and not the thoughts.
    {context}
    Question: {question}
    Answer: """


    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}

    llm = AzureChatOpenAI(deployment_name = 'gpt35',model_name='gpt-35-turbo', temperature=0.2,openai_api_key=os.getenv('OPENAI_API_KEY'))

    tools = [YahooFinanceNewsTool()]

    agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)
    
    
    answer = agent_chain.run(q)
    return answer

def profit_loss_azure(bondname):
    from azure.storage.blob import BlobServiceClient
    from langchain.agents import create_pandas_dataframe_agent
    from langchain.chat_models.azure_openai import AzureChatOpenAI
    STORAGEACCOUNTURL= os.getenv("STORAGEACCOUNTURL")
    STORAGEACCOUNTKEY= os.getenv("STORAGEACCOUNTKEY")
    CONTAINERNAME= "azureml"
    LOCALFILENAME="/Users/akansha/Downloads/"+bondname+"_Profit_Loss.xlsx"
    BLOBNAME= bondname+"/"+bondname+".xlsx"
    #download from blob

    blob_service_client_instance = BlobServiceClient(account_url=STORAGEACCOUNTURL, credential=STORAGEACCOUNTKEY)
    blob_client_instance = blob_service_client_instance.get_blob_client(CONTAINERNAME, BLOBNAME, snapshot=None)
    with open(LOCALFILENAME, "wb") as my_blob:
        blob_data = blob_client_instance.download_blob()
        blob_data.readinto(my_blob)
     
    headers =['Report Date','Mar-14','Mar-15','Mar-16','Mar-17','Mar-18','Mar-19','Mar-20','Mar-21','Mar-22','Mar-23']

    # Read the Excel file and skip rows before row 15
    df = pd.read_excel(LOCALFILENAME, sheet_name='Data Sheet',skiprows=15,names=headers)

    df=df.iloc[:15,:]
    df.dropna(inplace=True) 
    df = df.set_index('Report Date')
   
    df=df.T
    df.reset_index(inplace=True)
    df.set_index('index',inplace=True)
    df['Net profit'] = df['Net profit'].astype(float)
    df['Sales'] = df['Sales'].astype(float)
    print(df)
    return(df.round(0))

def get_bonds(bondname):
    from azure.storage.blob import BlobServiceClient
    from langchain.agents import create_pandas_dataframe_agent
    from langchain.chat_models.azure_openai import AzureChatOpenAI
    STORAGEACCOUNTURL= os.getenv("STORAGEACCOUNTURL")
    STORAGEACCOUNTKEY= os.getenv("STORAGEACCOUNTKEY")
    CONTAINERNAME= "azureml"
    LOCALFILENAME="/Users/akansha/Downloads/bonds.csv"
    BLOBNAME= "Bonds.csv"
    #download from blob

    blob_service_client_instance = BlobServiceClient(account_url=STORAGEACCOUNTURL, credential=STORAGEACCOUNTKEY)
    blob_client_instance = blob_service_client_instance.get_blob_client(CONTAINERNAME, BLOBNAME, snapshot=None)
    with open(LOCALFILENAME, "wb") as my_blob:
        blob_data = blob_client_instance.download_blob()
        blob_data.readinto(my_blob)
     
    if(bondname == "L&T Fin.Holdings" ):
        bondname = "L & T FINANCE LIMITED"
    # Read the Excel file and skip rows before row 15
    df = pd.read_csv(LOCALFILENAME)
    df_active = df[df['Instrument Status'] == 'Active']
    df_selected = df_active[df_active['Name of Issuer'].str.contains(bondname, case=False)]

    # Select the columns relevant to the user's request
    selected_columns = ['ISIN', 'Security Description', 'Face Value(in Rs.)',
                        'Issue Size(in Rs.)', 'Date of Allotment', 'Date of Redemption/Conversion',
                        'Coupon Rate (%)', 'Coupon Type', 'Frequency of Interest Payment']

# Create the final DataFrame with the selected columns
    df_selected = df_selected[selected_columns]
    return df_selected


def find_pdf_and_excel_files(directory):
    # Initialize lists to store the file paths
    list_files = []


    # Use os.walk to traverse through the directory and its subfolders
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            # Check if the file is a PDF or Excel file
            if file.lower().endswith(('.pdf', '.xlsx', '.xls')):
                    list_files.append(file_path)

    return list_files
  

def create_azure_index():
   
    from langchain.document_loaders import TextLoader
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.document_loaders import MergedDataLoader
    vector_store = get_azure_vector_store()
    loaders = []
    import os
    from pathlib import Path
    # assign directory

    directory_path = r"/Users/akansha/Downloads/Private Bonds/Coforge"
    files= find_pdf_and_excel_files(directory_path)
    for file in files:
                loader = load_document(file)
                loaders.append(loader)
    print(loaders)       
    merged= MergedDataLoader(loaders)
    documents = merged.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    vector_store.add_documents(documents=docs)

def get_azure_vector_store():
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.vectorstores.azuresearch import AzureSearch
    
    model: str = "text-embedding-ada-002"
    embeddings: OpenAIEmbeddings = OpenAIEmbeddings(deployment='textembedding', model=model)
    print(embeddings)
    index_name: str = "finsummary"
    vector_store: AzureSearch = AzureSearch(
        azure_search_endpoint=os.getenv('vector_store_address'),
        azure_search_key=os.getenv('vector_store_password'),
        index_name=index_name,
        embedding_function=embeddings.embed_query,
    )
    return vector_store




