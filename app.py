import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools import Tool
from langchain.schema import Document

import os
from dotenv import load_dotenv

load_dotenv()
groq_api = os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_API_KEY")

# ✅ Step 1: Initialize Wikipedia & Arxiv Tools
api_wrap = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wrap)

ar_wrap = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
ar = ArxivQueryRun(api_wrapper=ar_wrap)
search = DuckDuckGoSearchRun(name="Search")

# ✅ Step 2: Load and Process Documents for FAISS
loader = WebBaseLoader("https://docs.smith.langchain.com/")
docs = loader.load()

# ✅ Step 3: Ensure Each Document Has 'page_content'
formatted_docs = [
    Document(page_content=doc.page_content if hasattr(doc, "page_content") else doc.text, metadata=doc.metadata)
    for doc in docs
]

# ✅ Step 4: Split Documents into Chunks and Verify Structure
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(docs)

# ✅ Step 5: Store in FAISS and Create Custom Retriever Tool
vd = FAISS.from_documents(split_docs, OpenAIEmbeddings())
retriever = vd.as_retriever()

# ✅ Step 6: Wrap the Retriever in a Proper Tool Format
def retriever_tool_func(query):
    """Retrieve relevant documents based on the query."""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

retriever_tool = Tool(
    name="Langsmith Search",
    func=retriever_tool_func,
    description="Search for information about Langsmith from indexed documents."
)

# ✅ Step 7: Set Up Streamlit Chat UI
st.title("LangChain - Chat with Search")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I am a chatbot who can search the web. How can I help you today?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # ✅ Step 8: Initialize LLM and Agent
    llm = ChatGroq(groq_api_key=groq_api, model_name="Llama3-8b-8192", streaming=True)
    
    tools = [ar, wiki, retriever_tool]  # ✅ Ensure retriever tool is properly included
    search_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        
        # ✅ Step 9: Pass Only User Query Instead of Chat History
        response = search_agent.run(prompt, callbacks=[st_cb])  
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
