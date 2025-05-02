#!/usr/bin/env python3
"""
Streamlit web interface for Deep Recall Memories.

This app provides a web UI for testing and interacting with Deep Recall memory features.
"""

import os
import sys
import time
import json
import requests
from datetime import datetime
import streamlit as st
import pandas as pd
import altair as alt
from typing import Dict, List, Any

# Configuration
API_URL = os.environ.get("API_URL", "http://localhost:8404")

# Page config
st.set_page_config(
    page_title="Deep Recall Memories",
    page_icon="🧠",
    layout="wide",
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_id" not in st.session_state:
    st.session_state.user_id = f"streamlit_user_{int(time.time())}"
if "show_debug" not in st.session_state:
    st.session_state.show_debug = True  # Default to showing debug info
if "memory_test_active" not in st.session_state:
    st.session_state.memory_test_active = False
if "test_messages" not in st.session_state:
    st.session_state.test_messages = []

# Title and description
st.title("Deep Recall Memories")

# Sidebar
with st.sidebar:
    st.session_state.user_id = st.text_input("User ID", value=st.session_state.user_id)

    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

    st.session_state.show_debug = st.checkbox("Show Debug Info", value=st.session_state.show_debug)
    
    # Memory Test Feature
    st.subheader("Memory Test")
    st.write("Test if the system remembers information you've provided earlier.")
    
    if st.button("Start Memory Test"):
        st.session_state.memory_test_active = True
        st.session_state.test_messages = []
        st.rerun()
    
    if st.session_state.memory_test_active and st.button("End Memory Test"):
        st.session_state.memory_test_active = False
        st.rerun()
    
    st.header("About")
    st.markdown(
        """
        **Deep Recall** is an open source framework for adding *contextual memory* 
        about individual users to enrich conversations with open-source LLMs. 
        
        Core Components:
        
        - **FAISS vector database** for efficient similarity search of text embeddings

        - **Sentence transformers** (all-MiniLM-L6-v2 model) to generate text embeddings

        - **PostgreSQL with the pgvector extension** for metadata storage and retrieval
        """
    )
    
# Check if API is available
def check_api():
    try:
        response = requests.get(f"{API_URL}/")
        return response.status_code == 200
    except Exception as e:
        st.error(f"Error checking API: {str(e)}")
        return False

if not check_api():
    st.error(
        f"Cannot connect to API server at {API_URL}. "
        "Please make sure the server is running."
    )
    st.stop()

# Memory Test UI
if st.session_state.memory_test_active:
    st.subheader("Memory Test Mode")
    
    # Two columns layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("1. Provide information")
        information = st.text_area("Enter information you want the system to remember:", height=100)
        if st.button("Store Information"):
            if information:
                # Store the information as a memory
                try:
                    response = requests.post(
                        f"{API_URL}/memories",
                        params={
                            "user_id": st.session_state.user_id,
                            "text": information,
                            "importance": "HIGH"
                        }
                    )
                    if response.status_code == 200:
                        st.success("Information stored successfully!")
                        st.session_state.test_messages.append({
                            "role": "user", 
                            "content": information,
                            "timestamp": datetime.now().isoformat()
                        })
                    else:
                        st.error(f"Failed to store information: {response.text}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter some information to store.")
    
    with col2:
        st.write("2. Test if the system remembers")
        query = st.text_area("Ask about the information you provided:", height=100)
        if st.button("Test Memory"):
            if query:
                try:
                    # Retrieve memories related to the query
                    response = requests.get(
                        f"{API_URL}/memories",
                        params={
                            "user_id": st.session_state.user_id,
                            "query": query,
                            "limit": 5,
                            "threshold": 0.6
                        }
                    )
                    
                    if response.status_code == 200:
                        memories = response.json()
                        
                        if memories:
                            st.success("The system remembers!")
                            
                            # Display the retrieved memories
                            st.subheader("Retrieved Memories")
                            for i, memory in enumerate(memories):
                                with st.expander(f"Memory {i+1} (Similarity: {memory.get('similarity', 'N/A'):.2f})"):
                                    st.write(memory.get("text", ""))
                                    st.write(f"Created: {memory.get('created_at', '')}")
                        else:
                            st.error("The system doesn't remember this information.")
                    else:
                        st.error(f"Error retrieving memories: {response.text}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter a query to test memory.")
    
    # Display test history
    if st.session_state.test_messages:
        st.subheader("Test History")
        for msg in st.session_state.test_messages:
            st.write(f"**{msg['timestamp']}**: {msg['content']}")
else:
    # Regular chat UI
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Show memories in debug mode
            if st.session_state.show_debug and message["role"] == "assistant" and "memories" in message:
                if message["memories"]:
                    st.markdown("##### Relevant Memories")
                    memory_data = [{
                        "Text": mem["text"][:100] + "..." if len(mem["text"]) > 100 else mem["text"],
                        "Created": datetime.fromisoformat(mem["created_at"].replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M"),
                        "Relevance": f"{mem['similarity']:.2f}" if mem.get('similarity') else "N/A"
                    } for mem in message["memories"]]
                    
                    st.dataframe(memory_data, use_container_width=True)
                    
                    # Relevance visualization
                    if any('similarity' in mem for mem in message["memories"]):
                        chart_data = pd.DataFrame([{
                            "Memory": f"Memory {i+1}", 
                            "Relevance": float(mem.get("similarity", 0))
                        } for i, mem in enumerate(message["memories"]) if "similarity" in mem])
                        
                        if not chart_data.empty:
                            chart = alt.Chart(chart_data).mark_bar().encode(
                                x='Memory',
                                y='Relevance',
                                color=alt.Color('Relevance', scale=alt.Scale(scheme='viridis'))
                            ).properties(height=200)
                            
                            st.altair_chart(chart, use_container_width=True)
                else:
                    st.markdown("*No relevant memories found*")

    # Chat input
    if prompt := st.chat_input("What's on your mind?"):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
            
        # Display assistant thinking
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("⏳ Thinking...")
            
            # Send to API
            try:
                response = requests.post(
                    f"{API_URL}/chat",
                    json={
                        "user_id": st.session_state.user_id,
                        "message": prompt
                    }
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    
                    # Debug the response data
                    if st.session_state.show_debug:
                        st.markdown("##### API Response Data")
                        st.json(response_data)
                    
                    # Check if response field exists
                    if "response" in response_data:
                        # Display the response
                        resp_text = response_data["response"]
                        message_placeholder.markdown(resp_text)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": resp_text,
                            "memories": response_data.get("memories", [])
                        })
                    else:
                        message_placeholder.error("Error: Response field missing from API response")
                    
                    # Show debug info if enabled
                    if st.session_state.show_debug and response_data.get("memories"):
                        st.markdown("##### Relevant Memories")
                        memory_data = [{
                            "Text": mem["text"][:100] + "..." if len(mem["text"]) > 100 else mem["text"],
                            "Created": datetime.fromisoformat(mem["created_at"].replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M"),
                            "Relevance": f"{mem['similarity']:.2f}" if mem.get('similarity') else "N/A"
                        } for mem in response_data["memories"]]
                        
                        st.dataframe(memory_data, use_container_width=True)
                        
                        # Relevance visualization
                        if any('similarity' in mem for mem in response_data["memories"]):
                            chart_data = pd.DataFrame([{
                                "Memory": f"Memory {i+1}", 
                                "Relevance": float(mem.get("similarity", 0))
                            } for i, mem in enumerate(response_data["memories"]) if "similarity" in mem])
                            
                            if not chart_data.empty:
                                chart = alt.Chart(chart_data).mark_bar().encode(
                                    x='Memory',
                                    y='Relevance',
                                    color=alt.Color('Relevance', scale=alt.Scale(scheme='viridis'))
                                ).properties(height=200)
                                
                                st.altair_chart(chart, use_container_width=True)
                else:
                    message_placeholder.error(f"Error: {response.status_code} - {response.text}")
                    
                    # Show raw response for debugging
                    if st.session_state.show_debug:
                        st.markdown("##### Raw Response")
                        st.text(response.text)
            except Exception as e:
                message_placeholder.error(f"Error communicating with server: {str(e)}")
                
                # Show exception details for debugging
                if st.session_state.show_debug:
                    st.markdown("##### Exception Details")
                    st.exception(e) 