This project is a smart chatbot that can read web pages or PDF files and answer questions about them. It works by combining two powerful ideas:

Retrieval – It finds the most important parts of the text that relate to your question.

Generation – It uses an AI model to give you a helpful answer using that information.

You can run this chatbot  using local AI models like LLaMA 3 through a tool called Ollama.

It also has a nice web interface built with Streamlit so you can chat with.

Installation & Setup
To run this chatbot, you need to install Ollama for local AI model LLaMA 3, and install a few Python libraries for text processing, embedding, and the Streamlit interface.

 1. Install Ollama and LLaMA 3
Ollama lets you run AI models on your own machine.
Download and install Ollama from https://ollama.com
In your terminal, run the following command to download the LLaMA 3 model:
ollama pull llama3

2. Install Python Libraries
Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate 
Then install the required libraries:

streamlit
openai
python-dotenv
requests
beautifulsoup4
pdfplumber
faiss-cpu
tiktoken
PyMuPDF
Install it with:

pip install -r requirements.txt
