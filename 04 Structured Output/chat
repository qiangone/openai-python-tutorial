import streamlit as st
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict
import os

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.nuwaapi.com/v1"
)

# Initialize ChromaDB embedding function
# This is a key part of using ChromaDB with an OpenAI model
embedding_function = OpenAIEmbeddingFunction(api_key=client.api_key, model_name="text-embedding-ada-002")

# Initialize ChromaDB connection
@st.cache_resource
def init_db():
    """Initialize ChromaDB database connection and get the collection.

    Returns:
        ChromaDB collection object
    """

    CHROMA_PATH = r"chroma_db"

    # Connect to the persistent ChromaDB client with a specific path
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # Get or create the 'growing_vegetables' collection with the specified embedding function
    collection = db.get_or_create_collection("growing_vegetables")
    
    # Note: ChromaDB's get_or_create_collection handles the case where the collection
    # doesn't exist, so we don't need a try-except block here.
    return collection

def get_embedding(text, model="text-embedding-ada-002"):
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding

def get_context(query: str, collection, num_results: int = 5) -> List[Dict]:
    """Search the ChromaDB collection for relevant context and return structured data.

    Args:
        query: User's question
        collection: ChromaDB collection object
        num_results: Number of results to return

    Returns:
        List[Dict]: A list of dictionaries, each representing a document chunk
    """
    # Perform a vector search on the ChromaDB collection
    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=num_results
    )
    context_list = []
    
    # Ensure results are not empty
    if not results or 'documents' not in results or not results['documents'][0]:
        return []

    # Iterate through search results to format the output
    for i in range(len(results['documents'][0])):
        text_content = results['documents'][0][i]
        metadata = results['metadatas'][0][i]
        
        # Using .get() for safer access to metadata keys
        filename = metadata.get("source", "Unknown file")
        page_numbers = metadata.get("page_label", [])
        title = metadata.get("title", "Untitled section")

        # Build source citation
        source_parts = [filename]
        if page_numbers:
            source_parts.append(f"p. {', '.join(map(str, page_numbers))}")

        context_list.append({
            "text": text_content,
            "source": " - ".join(source_parts),
            "title": title
        })

    return context_list


def get_chat_response(messages: List[Dict], context: str) -> str:
    """Get streaming response from OpenAI API.

    Args:
        messages: Chat history
        context: Retrieved context as a single string
    """
    system_prompt = f"""You are a helpful assistant that answers questions based on the provided context.
    Use only the information from the context to answer questions. If you're unsure or the context
    doesn't contain the relevant information, say so.
    
    Context:
    {context}
    """
    # Prepare messages with system prompt
    messages_with_context = [{"role": "system", "content": system_prompt}] + messages

    # Create the streaming response
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages_with_context,
        temperature=0.7,
        stream=True,
    )

    # Use Streamlit's built-in streaming capability
    response = st.write_stream(stream)
    return response


# Initialize Streamlit app
st.title("📚 Document Q&A")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize database connection
collection = init_db()

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the document"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get relevant context
    with st.status("Searching document...", expanded=True) as status:
        # Get structured context data
        context_data = get_context(prompt, collection)
        
        # Prepare a single string for the LLM
        context_for_llm = "\n\n".join(
            f"Source: {item['source']}\nTitle: {item['title']}\nContent: {item['text']}" 
            for item in context_data
        )

        st.markdown(
            """
            <style>
            .search-result {
                margin: 10px 0;
                padding: 10px;
                border-radius: 4px;
                background-color: #f0f2f6;
            }
            .search-result summary {
                cursor: pointer;
                color: #0f52ba;
                font-weight: 500;
            }
            .search-result summary:hover {
                color: #1e90ff;
            }
            .metadata {
                font-size: 0.9em;
                color: #666;
                font-style: italic;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.write("Found relevant sections:")
        # Display each context chunk using the structured data
        for chunk in context_data:
            st.markdown(
                f"""
                <div class="search-result">
                    <details>
                        <summary>Source: {chunk['source']}</summary>
                        <div class="metadata">Title: {chunk['title']}</div>
                        <div style="margin-top: 8px;">{chunk['text']}</div>
                    </details>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Display assistant response
    with st.chat_message("assistant"):
        response = get_chat_response(st.session_state.messages, context_for_llm)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

