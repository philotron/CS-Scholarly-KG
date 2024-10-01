import streamlit as st
import requests
import datetime
import re
from google.cloud import firestore

# Path to your service account key file
service_account_path = './firestore-key.json'

# Create credentials from the service account file

# Use the credentials to create a Firestore client
db = firestore.Client.from_service_account_json(service_account_path)

# connect GPU: ssh -L 6000:localhost:5000 ubuntu@0.0.0.0
# connect database
# rasa server launch: rasa run --enable-api --cors "*"
# Define the URL of your Rasa chatbot server


def custom_css_style():
    """
    Injects custom CSS styles into a Streamlit application to highlight important definitions and topics.

    This function defines a CSS style block that adjusts the font size, weight, and color of elements with
    the classes `definition-style` and `topic-style`. The `definition-style` class elements are given a 
    specific color (#ff6347) for emphasis, while both classes increase the font size to 16px and set the font 
    weight to bold.
    """

    css = """
    <style>
    .definition-style {
        font-size:16px !important;
        font-weight:bold !important;
        color:#ff6347; 
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def handle_definition(text):
    """
    Parses a given text into an introduction, a highlighted definition, and the remaining part.

    This function takes a text string that includes a special marker ("📌") to indicate the start of a
    definition. It splits the text into three parts: the introduction, the definition (starting with the 
    marker), and the remaining part after the definition.
    """

    parsed_def = text.split("📌")
    intro = parsed_def[0]
    rest = parsed_def[1]
    main_sentence = rest.split("\n")
    definition = "📌" + main_sentence[0]
    end = main_sentence[1]
    return intro, definition, end


def detect_button_click(button):
    """
    Handles the click event for a button in a Streamlit application.

    This function updates the Streamlit session state to mark which button was clicked, clears the list of
    available buttons, and then triggers a rerun of the Streamlit script to reflect these changes.
    """

    st.session_state["button_clicked"] = button     
    st.session_state["available_buttons"] = []
    st.rerun()     

def paper_list_select_box(response_paper_list):
    """
    Filters and processes a list of papers from a Rasa response to populate a Streamlit select box.

    This function takes a response containing a list of papers, processes it to extract the relevant
    papers, and appends them to the `paper_list` in the Streamlit session state. It assumes that the 
    list of papers starts after the fourth line of the response (comes from action_show_paper_list_for_comparison of rasa).
    """

    paper_list = []
    parsed = response_paper_list.strip("").split("\n")
    paper_list = parsed[4:]
    for paper in paper_list:
        paper = paper.strip("- ")
        st.session_state["paper_list"].append(paper)


def paper_selection_submittion():
    """
    Combines selected papers into a single string and updates the session state in a Streamlit application.

    This function retrieves two selected papers from the Streamlit session state, combines them into a single 
    comma-separated string, and updates the session state with this combined string.
    """

    paper_1 = st.session_state.get("selected_paper_1")
    paper_2 = st.session_state.get("selected_paper_2")
    selected_papers = paper_1 + ", " + paper_2
    st.session_state["selected_papers"] = selected_papers


def handle_paper_ids(selected_papers):
    """
    Extracts numeric IDs from selected paper titles and formats them into a single string.

    This function takes a string of selected paper titles, splits them by commas, extracts numeric IDs from 
    each title using regular expressions, and combines these IDs into a formatted string.
    """

    selected_papers = selected_papers.split(",")
    ids = []
    for paper in selected_papers:
        num = re.findall(r'\d+', paper)
        ids.append(num[0])
    paper_ids = f"paper {ids[0]} and paper {ids[1]}"
    return paper_ids


def scroll_to_bottom():
    """
    Scrolls the web page to the bottom in a Streamlit application.

    This function injects a JavaScript snippet into a Streamlit app to automatically scroll the web page 
    to the bottom. It uses the `st.markdown` method with the `unsafe_allow_html` parameter set to `True` 
    to allow the execution of the JavaScript code.
    """

    js = "window.scrollTo(0, document.body.scrollHeight);"
    st.markdown(f"<script>{js}</script>", unsafe_allow_html=True)


if __name__ == "__main__":
    rasa_server_url = 'http://localhost:5005/webhooks/rest/webhook'

    # UI pade configuration
    st.set_page_config(
    page_title="LISSA",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto",
)

    st.header("LISSA: Language Interface for Scholarly Search Assistance")

    # Highlight of important definitions
    custom_css_style()


    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": """Hey! I am LISSA, a conversational interface for exploring publications in Natural Language Processing (NLP)! Please let me know how you want to start. 
                \n If would like to get an overview, just ask 'What are the main topics in NLP?'.
                \n You can also tell me what you're working on, like 'I need to generate automatic summaries of financial reports.', and I'll try to suggest related NLP topics to explore.""",
            }
        ]

        try:
            # Attempt to create a new document reference in the Firestore 'logs' collection
            doc_ref = db.collection('logs').document()

            st.session_state["firestore_id"] = doc_ref.id

        except Exception as e:
            print("Couldn't connect to Firestore: ", e)

    else: 
        if st.session_state["firestore_id"] is not None:
            try:
                # Retrieve the document reference from the Firestore 'logs' collection using the stored document ID
                doc_ref = db.collection('logs').document(st.session_state["firestore_id"])
                doc_ref.set({
                    "chat_history": st.session_state.messages,
                    "timestamp": datetime.datetime.now()
                })   
            except Exception as e:
                print("Couldn't log the data ", e)     


    # create session_state dictionary to save variables
    if "available_buttons" not in st.session_state:
        st.session_state["available_buttons"] = []

    if "button_clicked" not in st.session_state:
        st.session_state["button_clicked"] = None

    if "paper_list" not in st.session_state:
        st.session_state["paper_list"] = []

    if "selected_papers" not in st.session_state:
        st.session_state["selected_papers"] = None

    if "selected_paper_1" not in st.session_state:
        st.session_state["selected_paper_1"] = None

    if "selected_paper_2" not in st.session_state:
        st.session_state["selected_paper_2"] = None


    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            scroll_to_bottom()

    # create buttons if there's available button in the response
    if st.session_state.get("available_buttons", []).__len__() > 0:
        for button in st.session_state.get("available_buttons", []):
            print(button["payload"])
            st.button(button["title"], on_click=detect_button_click, args=[button])

            scroll_to_bottom()


    st.session_state["available_buttons"] = []

    # create reset button
    example_question = """↩️ Reset"""
    example_button = st.button(example_question, key=example_question)

    # React to user input
    if (
        prompt := st.chat_input("Type your message here") 
        or st.session_state.get("selected_papers") is not None
        or st.session_state.get("button_clicked", None) is not None 
        or example_button
    ):
        if st.session_state.get("selected_papers") is not None:
            pre_prompt = st.session_state.get("selected_papers")
            prompt = handle_paper_ids(pre_prompt)
            st.session_state["selected_paper_1"] = None
            st.session_state["selected_paper_2"] = None
            st.session_state["selected_papers"] = None
            st.session_state["paper_list"] = []

        elif st.session_state.get("button_clicked", None) is not None:
            prompt = st.session_state.get("button_clicked", {}).get("payload")
            st.session_state["button_clicked"] = None

        elif example_button:
            prompt = example_question   

        else:
            st.session_state["selected_paper_1"] = None
            st.session_state["selected_paper_2"] = None
            st.session_state["selected_papers"] = None
            st.session_state["paper_list"] = []    

    # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
    # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        payload = {"sender": "user", "message": prompt}

        response = requests.post(rasa_server_url, json=payload)

        if response.status_code == 200:
            # Process successful responses from Rasa
            response_data = response.json()

            for message in response_data:
                # Append each message (text or buttons) to the chat history
                if 'text' in message:
                    with st.chat_message("assistant"):
                        print(message['text'])

                        # highlight the defenition part of the response using custom css style
                        if "📌" in message['text']:
                            text = message['text']
                            intro, definition, end= handle_definition(text)
                            st.markdown(intro)
                            st.markdown(f'<p class="definition-style">{definition}</p>', unsafe_allow_html=True)
                            st.markdown(end)
                        else:
                            st.markdown(message['text'])

                    st.session_state.messages.append({"role": "assistant", "content": message['text']})

                # generate select box for paper list
                if "📎" in st.session_state.messages[-1]["content"]:
                    paper_list_select_box(message['text'])
                    with st.form("paper_selection"):
                        col1, col2 = st.columns(2)
                        with col1:
                            selected_paper_1 = st.selectbox(label="First paper", 
                                                            options=st.session_state["paper_list"], 
                                                            key="selected_paper_1")
                        with col2:
                            selected_paper_2 = st.selectbox(label="Second paper", 
                                                            options=st.session_state["paper_list"], 
                                                            key="selected_paper_2")
                        submitted = st.form_submit_button("Submit", on_click=paper_selection_submittion)

                # check if buttons in the response        
                if 'buttons' in message:
                    st.session_state["available_buttons"] = message['buttons']
                    st.rerun()
        else:
            st.error("Failed to send message to the chatbot.")
