import streamlit as st
import GA as genetic

# Set page title and background color
st.set_page_config(page_title="Hate Speech Detector", page_icon=":guardsman:", layout="wide", initial_sidebar_state="expanded")

#st.title("Hate Speech Detector", anchor=None, unsafe_allow_html=False,)

# Set dark theme
st.markdown("""
    <style>
    body {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

# Add a navigation bar
st.container()
menu = ["Home", "Contact Us", "Help", "About"]
choice = st.sidebar.selectbox("Select a page", menu)

# Define page content based on user's menu selection
if choice == "Home":
    # Add a title and some instructions
    st.title("Hate Speech Detector")
    st.write("Enter a sentence below to check for hate speech:")
    text_input = st.text_input("Input text here")

    # Add a button to initiate detection
    if st.button("Detect"):
        if text_input != '':
            if 1 == evaluate(text_input):
                st.write("This is a form of hate speech.")
            else:
                st.write("This is not a form of hate speech.")
        st.write("Detection results will appear here")
        
elif choice == "Contact Us":
    # Add contact information
    st.title("Contact Us")
    st.write("Email(ISHIKA BHOLA): f20201821@hyderabad.bits-pilani.ac.in")
    st.write("Email(RIYA SINGH): f20202048@hyderabad.bits-pilani.ac.in")
    st.write("Email(YASHI KHANDELWAL): f20202450@hyderabad.bits-pilani.ac.in")
    st.write("Email(JALI VIGNESHWAR REDDY): f20190447@hyderabad.bits-pilani.ac.in")
    st.write("Phone(ISHIKA BHOLA): 7905992891")
    
elif choice == "Help":
    # Add some help information
    st.title("Help")
    st.write("If you need help using the Hate Speech Detector, please email us at . f20201821@hyderabad.bits-pilani.ac.in")
    
elif choice == "About":
    # Add some information about the app
    st.title("About")
    st.write("The Hate Speech Detector uses Artificial Intelligence search algorithm models such as Genetic Algorithm and Particle Swarm Optimisation to classify text into hateful or non-hateful speech ")
    st.write("Version 1.0")
