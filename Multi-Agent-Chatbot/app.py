import streamlit as st
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict
from typing import Annotated, List, Union
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
from dotenv import load_dotenv
import os
load_dotenv()

# Tool imports
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults

# Tool setup
api_arxiv = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=500)
arxiv = ArxivQueryRun(api_wrapper=api_arxiv, description="Query arxiv papers")
api_wikipedia = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=500)
wikipedia = WikipediaQueryRun(api_wrapper=api_wikipedia)
tavily = TavilySearchResults()

tools = [arxiv, wikipedia, tavily]

# LLM setup
from langchain_groq import ChatGroq
llm = ChatGroq(model="qwen-qwq-32b")
llm_with_tools = llm.bind_tools(tools=tools)

# Response generation prompt
response_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant. Provide concise, well-formatted answers based on the context.
     
     Guidelines:
     1. Respond in complete sentences
     2. Never show internal thinking or reasoning
     3. Never output raw JSON or tool data
     4. Format lists with bullet points when appropriate (Date & Source if available)
     5. Always respond with just the final answer"""),
    ("human", """Question: {question}
     
     Context: {context}"""),
])

# Define LangGraph state
class State(TypedDict):
    messages: Annotated[List[Union[HumanMessage, AIMessage, ToolMessage]], add_messages]

# Define LangGraph nodes
def tool_calling_llm(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def generate_response(state: State):
    # Extract conversation components
    user_msg = next(m for m in state["messages"] if isinstance(m, HumanMessage))
    tool_results = [m.content for m in state["messages"] if isinstance(m, ToolMessage)]
    
    # Format context for LLM
    context = "\n".join([f"- {res}" for res in tool_results]) if tool_results else "No relevant information found"
    
    # Generate final response
    response = llm.invoke(response_prompt.format(
        question=user_msg.content,
        context=context
    ))
    
    # Return ONLY the final response message
    return {"messages": [AIMessage(content=response.content)]}

# Build the graph
builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools))
builder.add_node("generate_response", generate_response)

# Connect nodes
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools", "generate_response")
builder.add_edge("generate_response", END)

graph = builder.compile()

# Streamlit UI
st.set_page_config(page_title="AI Assistant", layout="centered")
st.title("Multi-Agent AI Chatbot")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Ask me anything...")

if user_input:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.spinner("Thinking..."):
        try:
            # Process through graph
            result = graph.invoke({"messages": [HumanMessage(content=user_input)]})
            
            # Extract final response (last AI message without tool calls)
            final_response = next(
                m.content for m in reversed(result["messages"]) 
                if isinstance(m, AIMessage) and not m.tool_calls
            )
            
            # Clean any residual thinking tags
            clean_response = final_response.split("</think>")[-1]
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": clean_response})
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(clean_response)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": "Sorry, I encountered an error processing your request."
            })