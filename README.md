# 🩺 Medical Adaptive RAG Chatbot

A medical question-answering chatbot built using **Adaptive RAG**, **ChromaDB**, **Tavily**, **Gemini**, and **Gradio**.

The system first searches the local medical knowledge base. If the retrieved information has low similarity, it automatically falls back to web search using Tavily.

## 📸 Application

<p align="center">
  <img
   src="https://drive.google.com/uc?export=view&id=1ZiSzti3W_lEjD3AuOVBQj07-j8171WlY">
</p>

## 🏗️ Architecture

```text
User Question
      ↓
   Gradio
      ↓
Adaptive RAG Pipeline
      ↓
  ChromaDB
      ↓
Similarity Check
   /         \
High        Low
 ↓           ↓
Local      Tavily
Knowledge   Web Search
   \         /
    \       /
      Gemini
        ↓
Answer + Sources

