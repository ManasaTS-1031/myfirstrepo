import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

key = os.getenv("GEMINI_KEY")

client = genai.Client(api_key=key)


def llm_chat(context, user_query):

    system_prompt = f"""
You are a medical information assistant.

Answer the user's question using ONLY the information provided
in the medical context.


### Medical Context

{context}

### User Question

{user_query}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=system_prompt,
        config=types.GenerateContentConfig(
            temperature=0.0
        )
    )

    return response.text.strip()


if __name__ == "__main__":

    context = """
    Cough is a symptom and protective reflex rather than a single disease.
    It may occur with a runny nose, sore throat, fever, wheezing,
    chest discomfort, mucus production, or shortness of breath.
    """

    question = "What are the symptoms associated with cough?"

    answer = llm_chat(context, question)

    print("\n--- ANSWER ---\n")
    print(answer)