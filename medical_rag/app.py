import gradio as gr
import sys
import os


# =========================================================
# ADD SRC TO PYTHON PATH
# =========================================================

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "src"
    )
)


# =========================================================
# IMPORT ADAPTIVE RAG PIPELINE
# =========================================================

from main import adaptive_rag_pipeline


# =========================================================
# CHAT FUNCTION
# =========================================================

def respond(message, history):

    if not message.strip():
        return history

    result = adaptive_rag_pipeline(
        user_question=message,
        threshold=0.70
    )

    answer = result["answer"]
    citations = result["citations"]

    # --------------------------------------------
    # FORMAT CITATIONS
    # --------------------------------------------

    citation_text = "\n\n### Sources\n"

    for i, citation in enumerate(citations, start=1):

        if citation["type"] == "web":

            citation_text += (
                f"{i}. "
                f"[{citation['title']}]"
                f"({citation['url']})\n"
            )

        elif citation["type"] == "local":

            citation_text += (
                f"{i}. "
                f"{citation['title']} "
                f"(similarity: "
                f"{citation['similarity']:.2f})\n"
            )

    final_response = (
        answer +
        citation_text
    )

    history = history or []

    history.append({
        "role": "user",
        "content": message
    })

    history.append({
        "role": "assistant",
        "content": final_response
    })

    return history

# =========================================================
# GRADIO UI
# =========================================================

with gr.Blocks(
    title="Medical Adaptive RAG Chatbot"
) as demo:

    gr.Markdown(
        """
        # 🩺 Medical Adaptive RAG Chatbot

        Ask questions about your medical knowledge base.

        The system first searches the local knowledge base.
        If the confidence is low, it automatically searches
        the web using Tavily.
        """
    )


    # -----------------------------------------------------
    # CHATBOT
    # -----------------------------------------------------

    chatbot = gr.Chatbot(
        height=550,
        show_label=False
    )


    # -----------------------------------------------------
    # MESSAGE INPUT
    # -----------------------------------------------------

    with gr.Row():

        message = gr.Textbox(
            placeholder="Type your medical question...",
            show_label=False,
            scale=5
        )

        submit = gr.Button(
            "Send",
            variant="primary",
            scale=1
        )


    # -----------------------------------------------------
    # CLEAR BUTTON
    # -----------------------------------------------------

    clear = gr.Button(
        "Clear Chat"
    )


    # =====================================================
    # SEND BUTTON
    # =====================================================

    submit.click(
        respond,
        inputs=[
            message,
            chatbot
        ],
        outputs=[
            chatbot
        ]
    ).then(
        lambda: "",
        outputs=message
    )


    # =====================================================
    # ENTER KEY
    # =====================================================

    message.submit(
        respond,
        inputs=[
            message,
            chatbot
        ],
        outputs=[
            chatbot
        ]
    ).then(
        lambda: "",
        outputs=message
    )


    # =====================================================
    # CLEAR CHAT
    # =====================================================

    clear.click(
        lambda: [],
        outputs=chatbot
    )


# =========================================================
# LAUNCH
# =========================================================

if __name__ == "__main__":

    demo.launch()