import streamlit as st
import requests

base_url = "http://localhost:8000"
# FastAPI endpoint URLs
SEARCH_TITLE_URL = f"{base_url}/search/title/"
SEARCH_DIRECTOR_URL = f"{base_url}/search/director/"
SEARCH_DESCRIPTION_URL = f"{base_url}/search/description/"


# Streamlit App
def main():
    st.title("Movie Search App")

    # User input
    search_text = st.text_input("Enter search query:")

    # Radio buttons for search type
    search_type = st.radio("Search by:", ["Title", "Director", "Description"])

    # Search button
    if st.button("Search"):
        print(search_type)
        if search_type:
            # Send request based on the selected search type
            if search_type == "Title":
                results = send_request(SEARCH_TITLE_URL + search_text)
            elif search_type == "Director":
                results = send_request(SEARCH_DIRECTOR_URL + search_text)
            elif search_type == "Description":
                results = send_request(SEARCH_DESCRIPTION_URL + search_text)
            else:
                st.error("Invalid search type selected.")
                return

            # Display results
            display_results(results)


# Function to send request to FastAPI
def send_request(url):
    try:
        response = requests.get(url, timeout=180)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error: {e}")
        return None


# Function to display results
def display_results(results):
    if results:
        st.subheader("Search Results:")
        for result in results:
            st.write(f"**Title:** {result['title']}")
            st.write(f"**Director:** {result['director']}")
            st.write(f"**Description:** {result['description']}")
            st.write(f"**Similarity Score:** {result['similarity_score']:.2%}")
            st.progress(result["similarity_score"])
            st.write("---")
    else:
        st.warning("No results found.")


if __name__ == "__main__":
    main()
