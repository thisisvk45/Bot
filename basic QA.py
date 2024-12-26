import streamlit as st
from transformers import pipeline
import torch

@st.cache_resource
def load_model():
    """
    Load the FLAN-T5-Large model using Hugging Face's pipeline.
    Automatically assigns the model to a GPU if available.
    """
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        device=device
    )

# Load the Q&A model
qa_model = load_model()

# Streamlit App Title and Description
st.title("Legal Question Chatbot")
st.write("Chat with the legal expert bot. Ask your questions and get detailed answers. Provide relevant context if needed.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input fields for the user
question = st.text_input("Enter your question:", key="current_question")

# Generate Answer on Submit
if st.button("Submit") and question:
    with st.spinner("Thinking..."):
        try:
            prompt = (
                f"You are a legal expert specializing in Indian law. Provide a detailed answer to this legal question: {question}"
            )
            result = qa_model(prompt, max_length=300, temperature=0.7)
            answer = result[0]['generated_text']
            st.session_state.chat_history.append((question, answer))
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Display Chat History
if st.session_state.chat_history:
    st.write("### Chat History")
    for i, (q, a) in enumerate(st.session_state.chat_history):
        st.write(f"**Q{i+1}:** {q}")
        st.write(f"**A{i+1}:** {a}")
        st.write("---")

# Clear Chat History Button
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
