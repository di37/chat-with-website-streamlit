from utils import *

# app config
st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("Chat with websites")

# sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

if website_url is None or website_url == "":
    st.info("Please enter a website URL")

else:
    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)

    for message in st.session_state.chat_history:
        with st.chat_message("AI" if isinstance(message, AIMessage) else "Human"):
            st.write(message.content)

    # user input
    if user_query := st.chat_input("Type your message here..."):
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        with st.chat_message("Human"):
            st.write(user_query)

        with st.chat_message("AI"):
            response_placeholder = st.empty()
            full_response = ""
            for chunk in get_response(user_query):
                if chunk.get("answer"):
                    full_response += chunk.get("answer")
                    response_placeholder.write(full_response)
            logger.info("Response successfully generated.")
        
        st.session_state.chat_history.append(AIMessage(content=full_response))