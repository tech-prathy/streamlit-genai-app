# app.py
import streamlit as st
import os
from dotenv import load_dotenv

from pdf_loader import load_pdf
from rag_pipeline import create_vector_store, retrieve_docs
from prompts import get_routing_prompt, build_structured_prompt
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# -------------------------------------------------------------
# Core Configurations & Initialization
# -------------------------------------------------------------
load_dotenv()

st.set_page_config(page_title="Smart Support Copilot", layout="wide", page_icon="🛠️")

# Enforce uniform styling to clean up margins and enhance scannability
st.markdown("""
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    stMarkdown {line-height: 1.6;}
    </style>
""", unsafe_allow_html=True)

openai_api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
openai_base_url = os.environ.get("AZURE_OPENAI_BASE_URL", "")
api_version = os.environ.get("AZURE_API_VERSION", "2024-12-01-preview")
azure_deployment_name = os.environ.get('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4o-mini')

# Initialize LangChain Azure OpenAI Client
client = AzureChatOpenAI(
    api_key=openai_api_key,
    azure_endpoint=openai_base_url,
    api_version=api_version,
    azure_deployment=azure_deployment_name,
    temperature=0.3
)

# Initialize global UI communication storage states
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
    
# -------------------------------------------------------------
# Interface Sidebar Layout
# -------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Control Panel")
    st.markdown("---")
    
    st.subheader("📂 Knowledge Grounding")
    uploaded_file = st.file_uploader("Upload Knowledge base or Manuals (PDF)", type="pdf")
    
    if uploaded_file:
        with st.spinner("Indexing document segments into FAISS matrix..."):
            raw_text = load_pdf(uploaded_file)
            if raw_text:
                st.session_state.vector_store = create_vector_store(raw_text)
                st.success("✅ Knowledge base built successfully!")
            else:
                st.error("Could not parse usable text from the provided PDF file.")
                
    st.markdown("---")
    
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# -------------------------------------------------------------
# Primary Application Stage UI
# -------------------------------------------------------------
st.title("📱 Smart Support Copilot")
st.caption("Context-Aware Enterprise Agent for Customer Service Operations.")

# Maintain chronological conversational progression by rendering early
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        if chat["role"] == "assistant":
            st.markdown(f"**[{chat['type']}]**")
        st.markdown(chat["content"])

# Collect user prompt input using native chat input box
if user_query := st.chat_input("Ask a troubleshooting step, comparison matrix request, or item specification..."):
    
    # 1. Immediately push and project user input on interface stage
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query, "type": "USER"})
    
    # 2. Extract context structures across preceding discussion 
    chat_history_text = ""
    for chat in st.session_state.chat_history[:-1]:
        prefix = "Customer" if chat["role"] == "user" else "Agent"
        chat_history_text += f"{prefix}: {chat['content']}\n"
        
    # 3. Execution Phase: Compute Query Classification Strategy Layer
    routing_prompt = get_routing_prompt(user_query, chat_history_text)
    
    try:
        # Since routing_prompt from prompts.py is typically a formatted string or ChatPromptTemplate object, 
        # we invoke the client directly with a HumanMessage containing that payload.
        response = client.invoke([HumanMessage(content=routing_prompt)])
        query_type = response.content.strip().upper()
        
        # Fallback safeguard validation block
        if query_type not in ["TROUBLESHOOTING", "COMPARISON", "GENERAL"]:
            query_type = "GENERAL"
    except Exception as e:
        query_type = "GENERAL"
        
    # 4. Phase 2: RAG Pipeline Grounded Documents Lookup
    retrieved_text = ""
    docs = []
    if st.session_state.vector_store:
        docs = retrieve_docs(st.session_state.vector_store, user_query, k=3)
        retrieved_text = "\n\n".join([f"[Source Chunk {i+1}]: {doc.page_content}" for i, doc in enumerate(docs)])
        
    # 5. Build Final Custom Targeted Route Instruction Prompt Matrix
    final_prompt = build_structured_prompt(query_type, retrieved_text, chat_history_text, user_query)
    
    # 6. Stream Out Custom Structured Outputs UI Box
    with st.chat_message("assistant"):
        # UI badges displaying classification status and grounding confidence hint
        badge_cols = st.columns([0.25, 0.75])
        with badge_cols[0]:
            st.info(f"🎯 Route: **{query_type}**")
        with badge_cols[1]:
            if retrieved_text:
                st.success("🔒 Response grounded via provided knowledge base documents.")
            else:
                st.warning("⚠️ Grounding text unavailable. Relying on default knowledge data bases.")
                
        placeholder = st.empty()
        full_response = ""
        
        try:
            # LangChain uses .stream() for handling response streaming.
            # We convert our raw final_prompt text into a list containing a HumanMessage.
            stream = client.stream([HumanMessage(content=final_prompt)])
            
            for chunk in stream:
                # Content in LangChain streaming arrives in chunk.content
                delta = chunk.content or ""
                full_response += delta
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
            
        except Exception as e:
            full_response = f"An API connection exception occurred processing the response sequence: {str(e)}"
            placeholder.markdown(full_response)

    # Cache response inside the reactive interface system variables state map
    st.session_state.chat_history.append({
        "role": "assistant", 
        "content": full_response,
        "type": query_type
    })