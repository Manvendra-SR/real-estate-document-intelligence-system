import streamlit as st
import requests

st.title("📄 Dynamic PDF Query Assistant")

# File Upload Section
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file:
    if st.button("Process & Index PDF"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        with st.spinner("Indexing..."):
            res = requests.post("http://localhost:8000/upload", files=files)
            if res.status_code == 200:
                st.success(res.json()["message"])
            else:
                st.error("Upload failed")

# Query Section
query = st.text_input("Ask a question about the document:")
if query:
    with st.spinner("Searching..."):
        response = requests.post("http://localhost:8000/search", json={"query": query})
        if response.status_code == 200:
            data = response.json()
            
            # Display time taken
            st.caption(f"⚡ Search completed in {data['latency_seconds']:.4f} seconds")
            
            for item in data['results']:
                with st.expander(f"Page {item['page']} (Relevance: {item['score']:.2f})"):
                    st.write(item['text'])
        else:
            st.info("Please upload and index a PDF first.")