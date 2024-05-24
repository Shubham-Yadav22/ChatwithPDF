import streamlit as st
import pickle
import os
from PyPDF2 import PdfReader  
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from openai import OpenAI

# Sidebar contents
with st.sidebar:
    st.title('LLM Chat App For PDF Documents 💡')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [llama 3](https://huggingface.co/docs/transformers/main/en/model_doc/llama3) LLM model
    - [LM Studio](https://lmstudio.ai/)
    ''')


def main():
    st.header("Chat with your PDF 💡")

    # Initialize session state variables
    if 'pdf_text' not in st.session_state:
        st.session_state['pdf_text'] = ""
    if 'vector_store' not in st.session_state:
        st.session_state['vector_store'] = None
    if 'conversation' not in st.session_state:
        st.session_state['conversation'] = []

    # Upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStores = pickle.load(f)
        else:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            VectorStores = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStores, f)

        st.session_state['pdf_text'] = text
        st.session_state['vector_store'] = VectorStores

    if st.session_state['vector_store']:
        # Create a form for query input
        with st.form(key='query_form', clear_on_submit=True):
            query = st.text_input("Ask questions about your PDF file:")
            submit_button = st.form_submit_button(label='Submit')

        if submit_button and query:
            # Perform similarity search
            docs = st.session_state['vector_store'].similarity_search(query=query)

            # Point to the local server
            client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

            completion = client.chat.completions.create(
                model="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
                messages=[
                    {"role": "system", "content": "Answer in brief but make it understandable."},
                    {"role": "user", "content": query}
                ],
                temperature=0.7,
            )

            response = completion.choices[0].message.content

            # Store the query and response in the conversation history
            st.session_state['conversation'].append({"role": "user", "content": query})
            st.session_state['conversation'].append({"role": "assistant", "content": response})

        # Display the conversation history
        for message in st.session_state['conversation']:
            if message['role'] == 'user':
                st.write(f"**You:** {message['content']}")
            else:
                st.write(f"**Assistant:** {message['content']}")


if __name__ == "__main__":
    main()
