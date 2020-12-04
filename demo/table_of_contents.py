import streamlit as st


class ToC():

    def __init__(self):
        self._items = []
        self._placeholder = None
    
    def header(self, text):
        self._markdown(text, "h2")

    def subheader(self, text):
        self._markdown(text, "h3", " " * 2)

    def placeholder(self, sidebar=False):
        self._placeholder = st.sidebar.empty() if sidebar else st.empty()

    def generate(self):
        if self._placeholder:
            self._placeholder.markdown("\n".join(self._items), unsafe_allow_html=True)
        return "" # Gambiarra 🤘
        
    def _markdown(self, text, level, space=""):
        key = "".join(filter(str.isalnum, text)).lower()

        st.markdown(f"<{level} id='{key}'>{text}</{level}>", unsafe_allow_html=True)
        self._items.append(f"{space}* <a href='#{key}'>{text}</a>")

