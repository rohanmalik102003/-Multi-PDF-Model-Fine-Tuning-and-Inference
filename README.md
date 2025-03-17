Developed at ISRO (U.R. Rao Satellite Centre, Bangalore)

This project enables fine-tuning of Llama-3.2-3B-Instruct on multiple PDF documents. It extracts text using OCR (Tesseract) when necessary and allows users to train a model based on PDF content for question-answering and summarization.

📌 Features
✅ Fine-tunes Llama-3.2-3B-Instruct on multiple PDFs
✅ Uses OCR (Tesseract) to extract text from scanned PDFs
✅ Allows users to ask multiple questions post-training
✅ Customizable training parameters (learning rate, max steps, etc.)
✅ Streamlit UI for easy interaction
✅ Runs entirely offline after setup

🛠 Setup Instructions (Linux)
1️⃣ Clone the Repository
git clone https://github.com/rohanmalik102003/-Multi-PDF-Model-Fine-Tuning-and-Inference.git
cd multi-pdf-finetuning

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Download the Model (Llama-3.2-3B-Instruct)
git clone https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-bnb-4bit models/Llama-3.2-3B-Instruct-bnb-4bit
📌 Ensure the model is inside models/ before running the application.

🚀 Usage
Start the Streamlit Application
streamlit run PDF.py

🔹 Open the browser at http://localhost:8501
🔹 Upload multiple PDF files for training
🔹 Fine-tune the model using the UI
🔹 Ask multiple questions based on the trained model
