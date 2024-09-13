import streamlit as st
import tempfile
import os
import google.generativeai as genai
from PyPDF2 import PdfReader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Replace this with your actual Gemini API key
GEMINI_API_KEY = 'AIzaSyDk8Z5MDp9KzYT2SFCsGIUYvjceAc7KAMI'
genai.configure(api_key=GEMINI_API_KEY)
# Initialize the model
model = genai.GenerativeModel('gemini-1.5-flash')
chat = model.start_chat(history=[])

def pdf_loader(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def csv_loader(tmp_file_path):
    return pd.read_csv(tmp_file_path)

def main():
    st.set_page_config(page_title="FAQ Chatbot with File Analysis", layout="wide")
    st.header("💬 Chatbot with File Analysis")

    if "conversation" not in st.session_state:
        st.session_state.conversation = chat
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "files_text" not in st.session_state:
        st.session_state.files_text = ""
    if "file_data" not in st.session_state:
        st.session_state.file_data = None
    if "plots" not in st.session_state:
        st.session_state.plots = []
    if "plot_details" not in st.session_state:
        st.session_state.plot_details = []

    with st.sidebar:
        st.title("Settings")
        st.markdown('---')
        st.subheader('Upload Your Files')
        uploaded_files = st.file_uploader("Upload your files (PDF/CSV)", type=['pdf', 'csv'], accept_multiple_files=True)

        process = st.button("Process")

    if process:
        st.session_state.files_text, st.session_state.file_data = get_files_data(uploaded_files)
        st.write("Files processed.")

    if st.session_state.files_text:
        st.subheader("File Content")
        st.text_area("File Content", st.session_state.files_text, height=200)

    if st.session_state.file_data is not None:
        st.subheader("Data Summary")
        st.write(st.session_state.file_data.head())  # Display first few rows of the data

        # Dynamic Column Selection for CSV files
        columns = st.session_state.file_data.columns

        plot_type = st.selectbox("Select Plot Type:", ["None", "Line Plot", "Scatter Plot", "Heatmap", "Pie Chart","Bar Chart", "Histogram", "Radar Chart", "Bubble Chart", "Box Plot", "Pair Plot"])

        if plot_type != "None":
            if plot_type in ["Line Plot", "Scatter Plot", "Bar Chart", "Bubble Chart"]:
                x_axis = st.selectbox("Select X-axis:", columns, key="x_axis")
                y_axis = st.selectbox("Select Y-axis:", columns, key="y_axis")

            if plot_type == "Line Plot":
                if x_axis and y_axis:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(st.session_state.file_data[x_axis], st.session_state.file_data[y_axis])
                    ax.set_title("Line Plot")
                    ax.set_xlabel(x_axis)
                    ax.set_ylabel(y_axis)
                    st.session_state.plots.append(fig)
                    st.session_state.plot_details.append(f"Line Plot of {y_axis} vs {x_axis}")
                    st.pyplot(fig)

            elif plot_type == "Scatter Plot":
                if x_axis and y_axis:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.scatterplot(data=st.session_state.file_data, x=x_axis, y=y_axis, ax=ax)
                    ax.set_title("Scatter Plot")
                    ax.set_xlabel(x_axis)
                    ax.set_ylabel(y_axis)
                    st.session_state.plots.append(fig)
                    st.session_state.plot_details.append(f"Scatter Plot of {y_axis} vs {x_axis}")
                    st.pyplot(fig)

            elif plot_type == "Bar Chart":
                if x_axis and y_axis:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    st.session_state.file_data.groupby(x_axis)[y_axis].sum().plot(kind='bar', ax=ax)
                    ax.set_title("Bar Chart")
                    ax.set_xlabel(x_axis)
                    ax.set_ylabel(y_axis)
                    st.session_state.plots.append(fig)
                    st.session_state.plot_details.append(f"Bar Chart of {y_axis} by {x_axis}")
                    st.pyplot(fig)

            elif plot_type == "Bubble Chart":
                size = st.selectbox("Select Bubble Size:", columns, key="bubble_size")
                if x_axis and y_axis and size:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.scatterplot(data=st.session_state.file_data, x=x_axis, y=y_axis, size=size, legend=False, sizes=(20, 2000), ax=ax)
                    ax.set_title("Bubble Chart")
                    ax.set_xlabel(x_axis)
                    ax.set_ylabel(y_axis)
                    st.session_state.plots.append(fig)
                    st.session_state.plot_details.append(f"Bubble Chart of {y_axis} vs {x_axis} with size {size}")
                    st.pyplot(fig)

            elif plot_type == "Heatmap":
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(st.session_state.file_data.corr(), annot=True, cmap="coolwarm", linewidths=.5, ax=ax)
                ax.set_title("Heatmap")
                st.session_state.plots.append(fig)
                st.session_state.plot_details.append("Heatmap of Correlation Matrix")
                st.pyplot(fig)

            elif plot_type == "Histogram":
                column = st.selectbox("Select Column:", columns, key="hist_column")
                if column:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(st.session_state.file_data[column], kde=True, ax=ax)
                    ax.set_title("Histogram")
                    ax.set_xlabel(column)
                    st.session_state.plots.append(fig)
                    st.session_state.plot_details.append(f"Histogram of {column}")
                    st.pyplot(fig)

            elif plot_type == "Radar Chart":
                categories = st.multiselect("Select Categories:", columns, key="radar_categories")
                if len(categories) > 1:
                    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(polar=True))
                    values = st.session_state.file_data[categories].mean().values.flatten().tolist()
                    values += values[:1]
                    categories += categories[:1]

                    angles = [n / float(len(categories)) * 2 * 3.141592653589793 for n in range(len(categories))]
                    angles += angles[:1]

                    ax.set_theta_offset(3.141592653589793 / 2)
                    ax.set_theta_direction(-1)

                    plt.xticks(angles[:-1], categories)

                    ax.plot(angles, values, linewidth=1, linestyle='solid')
                    ax.fill(angles, values, 'b', alpha=0.1)
                    ax.set_title("Radar Chart")
                    st.session_state.plots.append(fig)
                    st.session_state.plot_details.append(f"Radar Chart of {', '.join(categories[:-1])}")
                    st.pyplot(fig)
                else:
                    st.warning("Please select at least two categories for the radar chart.")

            elif plot_type == "Pie Chart":
                column = st.selectbox("Select Column for Pie Chart:", columns, key="pie_column")
                if column:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    data_counts = st.session_state.file_data[column].value_counts()
                    # Create pie chart with labels and percentages
                    wedges, texts, autotexts = ax.pie(
                        data_counts,
                        labels=data_counts.index,
                        autopct='%1.1f%%',
                        startangle=90,
                        wedgeprops=dict(width=0.3)  # Optionally set width to make it a donut chart
                    )
                    ax.set_title("Pie Chart")

                    # Customize label and percentage colors
                    for text in texts:
                        text.set_fontsize(10)
                    for autotext in autotexts:
                        autotext.set_fontsize(10)
                        autotext.set_color('red')
                    ax.legend(wedges, data_counts.index, title="Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
                    st.pyplot(fig)


            elif plot_type == "Box Plot":
                column = st.selectbox("Select Column for Box Plot:", columns, key="box_column")
                if column:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.boxplot(data=st.session_state.file_data, y=column, ax=ax)
                    ax.set_title("Box Plot")
                    ax.set_ylabel(column)
                    st.session_state.plots.append(fig)
                    st.session_state.plot_details.append(f"Box Plot of {column}")
                    st.pyplot(fig)

            elif plot_type == "Pair Plot":
                fig = sns.pairplot(st.session_state.file_data)
                st.session_state.plots.append(fig)
                st.session_state.plot_details.append("Pair Plot of the DataFrame")
                st.pyplot(fig)

    user_question = st.text_input("Ask a question about the PDF or plotted graphs:", key="user_question_input")
    if st.button("Send", key="send_button") and user_question:
        # Combine file text and plot details for context
        combined_context = st.session_state.files_text + "\n\n" + "\n".join(st.session_state.plot_details)
        response = query_gemini_api(user_question, combined_context)
        st.session_state.chat_history.append({"user": user_question, "bot": response})
        st.session_state.user_question = ""
        st.subheader("Response from AI")
        st.write(response)

    display_chat_history(st.session_state.chat_history)

def get_files_data(uploaded_files):
    text = ""
    file_data = None
    for uploaded_file in uploaded_files:
        split_tup = os.path.splitext(uploaded_file.name)
        file_extension = split_tup[1]
        if file_extension == ".pdf":
            text += pdf_loader(uploaded_file)
        elif file_extension == ".csv":
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
                file_data = csv_loader(tmp_file_path)
    return text, file_data

def query_gemini_api(question, context):
    # Combining user question with file content and plot details as context
    prompt = f"The following context is extracted from an uploaded file and plotted graphs:\n\n{context}\n\nUsing this context, answer the question:\n\n{question}"
    try:
        response = st.session_state.conversation.send_message(prompt)
        return response.text
    except Exception as e:
        st.error(f"Failed to get response from Gemini API: {e}")
        return "Error: Failed to retrieve a valid response from the API."

def display_chat_history(chat_history):
    st.subheader("Chat History")
    for chat in chat_history:
        if "user" in chat:
            st.markdown(f"You: {chat['user']}")
        if "bot" in chat:
            st.markdown(f"Bot: {chat['bot']}")

if __name__ == "__main__":
    main()
