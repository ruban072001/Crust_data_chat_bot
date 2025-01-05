import os
from dotenv import load_dotenv
import streamlit as st
from langchain.tools import Tool
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_huggingface import HuggingFaceEmbeddings
from langchain import hub
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables
load_dotenv()

# Initialize Streamlit app
st.title("AI Chatbot - Ask Anything!")
st.subheader("An AI assistant to answer questions and fetch information.")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [("system", "You are a helpful assistant.")]

# OpenAI API Configuration
llm = ChatOpenAI(model='gpt-3.5-turbo')
embed = OpenAIEmbeddings(model="text-embedding-3-small")
hug = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Database path
db_path = os.path.join(r"C:\Users\KIRUBA\Desktop\openai api's\interview chatbot", 'db', 'chroma_db_1')

# Initialize Chroma
db = Chroma(embedding_function=hug, persist_directory=db_path)

# Retriever setup
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={'k': 5, 'score_threshold': 0.1}
)

# Prompts for history-aware retrieval
template_his = (
    "Given a chat history and the latest user question, "
    "formulate a standalone question that can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed."
)
prompt_his = ChatPromptTemplate.from_messages(
    [("system", template_his), MessagesPlaceholder("chat_history"), ("human", "{input}")]
)
history_retrieve = create_history_aware_retriever(llm, retriever, prompt_his)

# QA Chain
template_qa = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say 'I don't know.' "
    "Keep the answer concise and within three sentences.\n\n"
    "{context}"
)
prompt_qa = ChatPromptTemplate.from_messages(
    [("system", template_qa), MessagesPlaceholder("chat_history"), ("human", "{input}")]
)
documents = create_stuff_documents_chain(llm, prompt_qa)
rag_chain = create_retrieval_chain(retriever=history_retrieve, combine_docs_chain=documents)

# Tools
def get_info(query):
    from wikipedia import summary
    try:
        return summary(query, sentences=2)
    except Exception:
        return "Unable to fetch information from Wikipedia."

tools = [
    Tool(
        name="Answer Question",
        func=lambda input, **kwargs: rag_chain.invoke(
            {"input": input, "chat_history": kwargs.get("chat_history", [])}
        ),
        description="Find relevant answers for any context."
    ),
    Tool(
        name="Wikipedia",
        func=get_info,
        description="Fetch more information about specific topics."
    ),
]

# Agent setup
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor.from_agent_and_tools(agent, tools, verbose=True, handle_parsing_errors=True)

# Chat interface
user_input = st.chat_input("Ask your question...")

if user_input:
    # Append user input to chat history
    st.session_state.chat_history.append(("human", user_input))

    try:
        # Get response from agent
        result = executor.invoke({"input": user_input, "chat_history": st.session_state.chat_history})
        answer = result.get("output", "Sorry, I couldn't understand your question.")
    except Exception as e:
        answer = f"An error occurred: {str(e)}"

    # Append AI's response to chat history
    st.session_state.chat_history.append(("ai", answer))

# Display chat history
for role, message in st.session_state.chat_history:
    if role == "human":
        st.chat_message("user").write(message)
    else:
        st.chat_message("assistant").write(message)
