# Intelligent OCR, Data Storage, and AI-Powered Chatbot System

## Description
This project focuses on building a comprehensive pipeline for text and image extraction, processing, summarization, and AI-powered querying. The system employs OCR techniques to extract text and images from input files, processes the extracted data using NLP methods, and stores the cleaned data in both SQL and vector databases for efficient retrieval. Furthermore, an AI chatbot powered by a large language model (LLM) is developed to answer queries based on the processed data. The system includes both a backend API for querying and a user-friendly frontend interface.

## Features

### Text and Image Extraction
- Utilizes OCR techniques such as **Tesseract**, **EasyOCR**, and **OpenCV** for extracting text and images.
- Compares the performance of various OCR methods.

### Text Processing
- Cleans and preprocesses extracted text using NLP techniques.
- Annotates named *entities* and *part-of-speech* (POS) tags.
- Performs feature extrapolation for enriched data understanding.

### Image Summarization
- Summarizes each image using LLMs.
- Compares different summarization techniques and models.

### Data Storage
- Stores processed data in **PostgreSql** database.
- Utilizes **Milvus** (Vector DB) for embedding-based similarity search.

### Hybrid Search
- Implements hybrid search combining dense and sparse vector representations in Milvus.
- Enables both *semantic* and *keyword-based* retrieval for optimal query handling.

### FastAPI Backend
- Provides an API endpoint to retrieve image summaries based on file names.

### AI-Powered Chatbot
- Uses a Large Language Model (LLM) for answering questions across all data points.
- Integrated with Chainlit for a user-friendly frontend interface.

## File Structure
- `data.py`
Contains utility functions for handling data-related tasks such as preprocessing text, embedding generation, and interacting with databases.

- `extractor.py`
Contains utility functions for OCR techniques and extracting text and images from files using libraries like Tesseract, EasyOCR, and others.

- `query_processor.py`
Implements database-related operations, including SQL and vector database queries for hybrid search and semantic retrieval.

- `fastapi.py`
Hosts the FastAPI application that provides an endpoint to retrieve image summaries based on file names.

- `app.py`
Contains Chainlit codes to create a frontend interface for the AI chatbot, enabling interaction with the processed data.

- `main.py`
Serves as the main script to perform the complete pipeline of OCR, data processing, summarization, storage, and query handling.

- `overall.ipynb`
A Jupyter Notebook that consolidates all tasks, providing an end-to-end implementation of OCR, summarization, storage, and chatbot interaction.