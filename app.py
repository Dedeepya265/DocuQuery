import streamlit as st
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from dotenv import load_dotenv
import os

# Load environment variables (if needed)
load_dotenv()

# ✅ PDF text extraction using pdfplumber
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            with pdfplumber.open(pdf) as pdf_reader:
                for page in pdf_reader.pages:
                    content = page.extract_text()
                    if content:
                        text += content
        except Exception as e:
            st.error(f"❌ Failed to read {pdf.name}: {e}")
    return text

# ✅ Split long text into smaller chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    return text_splitter.split_text(text)

# ✅ Generate embeddings and store in FAISS
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_texts(text_chunks, embedding=embeddings)

# ✅ Set up the LLM-powered conversational chain
def get_conversational_chain(vector_store):
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        truncation=True  # prevents input overflow errors
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )

# ✅ Handle each user question
def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        speaker = "👤 Human" if i % 2 == 0 else "🤖 Bot"
        st.write(f"{speaker}: **{message.content}**")

# ✅ Streamlit app layout and flow
def main():
    st.set_page_config(page_title="DocuQuery: AI-Powered PDF Knowledge Assistant")
    st.header("📄 DocuQuery: AI-Powered PDF Knowledge Assistant")

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Input box for user questions
    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")
    if user_question and st.session_state.conversation:
        handle_user_input(user_question)

    # Sidebar for uploading PDFs
    with st.sidebar:
        st.title("Settings")
        st.subheader("Upload your Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Process Button",
            accept_multiple_files=True,
            type=["pdf"]
        )
        if st.button("Process") and pdf_docs:
            with st.spinner("Processing..."):
                st.write("📄 Extracting text from PDFs...")
                raw_text = get_pdf_text(pdf_docs)

                st.write("✂️ Splitting text into chunks...")
                text_chunks = get_text_chunks(raw_text)

                st.write("🔍 Creating vector store...")
                vector_store = get_vector_store(text_chunks)

                st.write("🧠 Initializing AI model...")
                st.session_state.conversation = get_conversational_chain(vector_store)

                st.success("✅ Ready! Ask questions in the box above.")

# ✅ Entry point
if __name__ == "__main__":
    main()
