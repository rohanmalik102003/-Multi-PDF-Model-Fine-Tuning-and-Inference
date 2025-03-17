import streamlit as st
import fitz
import pytesseract
from PIL import Image
import os
import tempfile
import logging
from datasets import Dataset
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer
from unsloth import FastLanguageModel
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

st.title("Multi-PDF Model Fine-Tuning and Inference")

if "trained" not in st.session_state:
    st.session_state.trained = False
if "model" not in st.session_state:
    st.session_state.model = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None
if "answers" not in st.session_state:
    st.session_state.answers = {}

st.sidebar.subheader("Inference Settings")
st.session_state.temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value=1.0, value=0.9, step=0.05)
st.session_state.top_p = st.sidebar.slider("Top P", min_value=0.1, max_value=1.0, value=1.0, step=0.05)
st.session_state.max_new_tokens = st.sidebar.slider("Max New Tokens", min_value=10, max_value=1024, value=50, step=10)

if not st.session_state.trained:
    st.subheader("Upload PDFs and Set Training Parameters")
    uploaded_pdfs = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

    if uploaded_pdfs:
        learning_rate = st.slider("Learning Rate", min_value=1e-6, max_value=1e-2, value=1e-5, step=1e-6)
        max_steps = st.slider("Max Steps", min_value=10, max_value=100, value=40)

        if st.button("Start Training"):
            st.write("Processing the uploaded PDFs and starting training...")

            pdf_paths = []
            for uploaded_pdf in uploaded_pdfs:
                temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                pdf_paths.append(temp_pdf.name)
                with open(temp_pdf.name, "wb") as f:
                    f.write(uploaded_pdf.getbuffer())

            max_seq_length = 512
            output_dir = os.path.join(tempfile.gettempdir(), "pdf_model_outputs")
            os.makedirs(output_dir, exist_ok=True)
            use_ocr = True

            logging.info("Loading the language model and tokenizer...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name="unsloth/Llama-3.2-3B-Instruct",
                max_seq_length=max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )
            tokenizer.padding_side = 'left'

            logging.info("Setting up the LoRA fine-tuning configuration...")
            model = FastLanguageModel.get_peft_model(
                model,
                r=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_alpha=16,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing=False,
                random_state=3407,
                use_rslora=False,
            )

            def extract_text_from_pdf(pdf_path):
                doc = fitz.open(pdf_path)
                text_chunks = []
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text("text")
                    if text.strip():
                        text_chunks.append({"page": page_num + 1, "content": text.strip()})
                    elif use_ocr:
                        pix = page.get_pixmap()
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        ocr_text = pytesseract.image_to_string(img)
                        if ocr_text.strip():
                            text_chunks.append({"page": page_num + 1, "content": ocr_text.strip()})
                return text_chunks

            def generate_instruction_response_pairs(chunks, pdf_name):
                dataset = []
                for chunk in chunks:
                    page = chunk["page"]
                    content = chunk["content"]
                    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
                    for para in paragraphs:
                        instruction = f"Summarize the content of page {page} in {pdf_name}."
                        response = f"The content is: {para}"
                        dataset.append({"instruction": instruction, "response": response})
                        instruction2 = f"What is the key information on page {page} in {pdf_name}?"
                        response2 = f"The key information is: {para}"
                        dataset.append({"instruction": instruction2, "response": response2})
                return dataset

            full_dataset = []
            for pdf_path in pdf_paths:
                pdf_name = os.path.basename(pdf_path)
                text_chunks = extract_text_from_pdf(pdf_path)
                full_dataset.extend(generate_instruction_response_pairs(text_chunks, pdf_name))

            if not full_dataset:
                st.error("No valid instruction-response pairs were generated. Check the input PDFs.")
            else:
                st.write(f"Generated {len(full_dataset)} instruction-response pairs.")
                def formatting_prompts_func(examples):
                    instructions = [example["instruction"] for example in examples]
                    responses = [example["response"] for example in examples]
                    texts = [f"Instruction:\n{instr}\nResponse:\n{resp}" for instr, resp in zip(instructions, responses)]
                    return {"text": texts}

                formatted_dataset = formatting_prompts_func(full_dataset)
                hf_dataset = Dataset.from_dict(formatted_dataset)

                def preprocess_data(example_batch):
                    tokenized_batch = tokenizer(
                        example_batch["text"],
                        max_length=max_seq_length,
                        padding="max_length",
                        truncation=True,
                    )
                    tokenized_batch["labels"] = tokenized_batch["input_ids"].copy()
                    return tokenized_batch

                tokenized_dataset = hf_dataset.map(preprocess_data, batched=True, batch_size=8)

                trainer = SFTTrainer(
                    model=model,
                    tokenizer=tokenizer,
                    train_dataset=tokenized_dataset,
                    dataset_text_field="text",
                    max_seq_length=max_seq_length,
                    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
                    dataset_num_proc=1,
                    packing=False,
                    args=TrainingArguments(
                        output_dir=output_dir,
                        per_device_train_batch_size=1,
                        gradient_accumulation_steps=8,
                        warmup_steps=10,
                        max_steps=max_steps,
                        learning_rate=learning_rate,
                        fp16=True if torch.cuda.is_available() else False,
                        logging_steps=1,
                        optim="adamw_8bit",
                        weight_decay=0.01,
                        lr_scheduler_type="linear",
                        seed=3407,
                        report_to="none",
                    ),
                )

                st.write("Training in progress...")
                trainer.train()

                st.session_state.trained = True
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
                st.success("Training complete! You can now ask multiple questions.")

if st.session_state.trained:
    st.subheader("Ask Questions")
    user_questions = st.text_area("Enter multiple questions (one per line):")

    if st.button("Submit Questions") and user_questions:
        FastLanguageModel.for_inference(st.session_state.model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.session_state.model.to(device)

        questions = user_questions.split("\n")
        st.session_state.answers = {}

        for question in questions:
            inputs = st.session_state.tokenizer(f"Instruction:\n{question}\nResponse:", return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            with torch.no_grad():
                output = st.session_state.model.generate(**inputs, max_new_tokens=st.session_state.max_new_tokens, temperature=st.session_state.temperature, top_p=st.session_state.top_p)
            st.session_state.answers[question] = st.session_state.tokenizer.decode(output[0], skip_special_tokens=True)

        for question, answer in st.session_state.answers.items():
            with st.expander(f"Question: {question}"):
                st.write(answer)
