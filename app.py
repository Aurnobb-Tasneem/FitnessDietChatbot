import streamlit as st
import requests
import json
import urllib.parse
import time

# Streamlit page configuration
st.set_page_config(page_title="Workout & Diet Chatbot", layout="wide")

# Initialize session state for messages per model
if "messages" not in st.session_state:
    st.session_state.messages = {
        "google/gemini-1.5-flash": [],
        "mistralai/Mixtral-8x7B-Instruct-v0.1": [],
        "meta-llama/Llama-3.1-70B-Instruct": []
    }

# Streamlit UI
st.title("Workout & Diet Chatbot")
st.markdown("Interact with each model separately for workout, diet, or gym recommendations!")

# Create three columns
col1, col2, col3 = st.columns(3)

# Model names with display names for UI
models = [
    {"id": "google/gemini-1.5-flash", "display": "Gemini"},
    {"id": "mistralai/Mixtral-8x7B-Instruct-v0.1", "display": "Mistral"},
    {"id": "meta-llama/Llama-3.1-70B-Instruct", "display": "Llama"}
]


def stream_response(query, target_model):
    """
    Streams responses from the FastAPI backend for the target model.
    """
    encoded_query = urllib.parse.quote(query)
    encoded_model = urllib.parse.quote(target_model)
    url = f"http://localhost:8000/chat?query={encoded_query}&model={encoded_model}"

    try:
        with requests.get(url, stream=True, headers={"Accept": "text/event-stream"}) as response:
            response.raise_for_status()
            full_response = ""
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    line = line.strip()
                    if line.startswith("data:"):
                        data = line[5:].strip()
                        try:
                            event_data = json.loads(data)
                            event = event_data.get("event")
                            if event == "response_chunk":
                                model = event_data["model"]
                                chunk = event_data["chunk"]
                                if model == target_model or model == "system":
                                    full_response = chunk  # Use the full response sent by server
                                    yield model, full_response
                            elif event == "done":
                                break
                        except json.JSONDecodeError as e:
                            yield "error", f"Failed to parse SSE data: {data}"
    except requests.RequestException as e:
        yield "error", f"Error connecting to server: {str(e)}"


# Render each model's conversation form
for col, model in zip([col1, col2, col3], models):
    with col:
        st.subheader(model["display"])
        chat_container = st.container()

        # Display chat history for this model
        with chat_container:
            for message in st.session_state.messages[model["id"]]:
                with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "👤"):
                    if message["role"] == "assistant":
                        st.markdown(f"**{message['display_name']}**: {message['content']}")
                    else:
                        st.markdown(message['content'])

        # Input box for this model
        user_input = st.chat_input(f"Ask {model['display']}...", key=f"input_{model['id']}")

        if user_input:
            # Add user message to this model's history
            st.session_state.messages[model["id"]].append({"role": "user", "content": user_input})
            with chat_container:
                with st.chat_message("user", avatar="👤"):
                    st.markdown(user_input)

                # Stream responses for this model
                placeholder = st.empty()
                current_response = {"model": None, "content": "", "display_name": model["display"]}
                with placeholder:
                    with st.chat_message("assistant", avatar="🤖"):
                        response_container = st.markdown("")

                for model_name, content in stream_response(user_input, model["id"]):
                    if model_name == "error":
                        st.session_state.messages[model["id"]].append({
                            "role": "assistant",
                            "model": "Error",
                            "display_name": "Error",
                            "content": content
                        })
                        with placeholder:
                            with st.chat_message("assistant", avatar="🤖"):
                                st.markdown(f"**Error**: {content}")
                    else:
                        current_response["model"] = model_name
                        current_response["content"] = content
                        with placeholder:
                            with st.chat_message("assistant", avatar="🤖"):
                                response_container.markdown(f"**{model['display']}**: {content}")
                        time.sleep(0.05)  # Smooth streaming effect

                # Save the final response as a single message
                if current_response["model"]:
                    st.session_state.messages[model["id"]].append({
                        "role": "assistant",
                        "model": current_response["model"],
                        "display_name": model["display"],
                        "content": current_response["content"]
                    })