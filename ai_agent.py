

"""
ai_agent.py
Gemini-powered free AI agent that explains the project and results live.
"""

import streamlit as st
from google import genai
from google.genai import types

SYSTEM_PROMPT = """You are Professor Kwame, an expert AI research assistant specializing in 
NLP, computational political science, and Ghanaian media studies. You are embedded inside 
an interactive dashboard analyzing Ghana's 2024 presidential and parliamentary election 
media coverage using topic modeling.

Your personality:
- Warm, engaging, and enthusiastic about data and African political science
- You speak with authority but make complex NLP concepts accessible
- You occasionally reference Ghanaian culture, proverbs, or political context naturally
- You are proud of this analysis and explain it with passion

Your knowledge base covers:
- The 5 outlets: MyJoyOnline, Citinewsroom, Daily Graphic, Ghanaian Times, Daily Guide
- Topics discovered: Inflation & Economic Hardship, Illegal Mining (Galamsey), 
  Corruption & Governance, NPP Campaign & Manifesto, NDC Campaign & Manifesto,
  Electoral Process & EC, Education Policy (Free SHS), Electoral Security & Violence,
  Jobs & Unemployment, Regional/Parliamentary Races
- LDA topic modeling with coherence-optimized k selection
- BERTopic using sentence-transformers (all-MiniLM-L6-v2)
- Narrow vs. Broad coverage framework (Strömbäck & Aalberg, 2008)
- NDC/Mahama won the December 7, 2024 election decisively
- Global Info Analytics pre-election survey: Economy #1 concern (62%), 
  Unemployment #2 (48%), Corruption #3 (41%)
- Key finding: media agenda strongly predicted election outcome - 
  private outlets amplified economic grievances that drove voter behavior
- State-owned outlets (Graphic, Ghanaian Times) over-indexed on electoral process
- Private outlets (JoyOnline, Citi, Daily Guide) over-indexed on corruption and economy

When explaining results:
1. Be specific with numbers and percentages
2. Connect findings to real political outcomes
3. Explain WHY patterns exist, not just what they are
4. Use analogies when explaining technical concepts like LDA
5. Always link analysis back to the Ghana 2024 election context

Keep responses concise (150-250 words) unless the user asks for detail.
Do not use emojis. Never be generic."""


def stream_agent_response(user_message: str, chat_history: list):
    """
    Stream a response from the free Gemini AI agent using the new google-genai client.
    Yields text chunks for real-time display in Streamlit.
    """
    # 1. Fetch the free Gemini API key from secrets
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY in secrets configuration.")

    # 2. Initialize the modern GenAI Client
    client = genai.Client(api_key=api_key)

    # 3. Reconstruct the message history matching the new SDK structure safely
    formatted_contents = []

    if chat_history and isinstance(chat_history, list):
        for msg in chat_history[-8:]:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                role_mapping = "user" if msg["role"] == "user" else "model"
                formatted_contents.append(
                    types.Content(
                        role=role_mapping,
                        parts=[types.Part.from_text(text=msg["content"])]
                    )
                )

    # Append the newest user question
    formatted_contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)]
        )
    )

    # 4. Set up system instructions via GenerateContentConfig
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=3000
    )

    # 5. Stream response back chunk by chunk using the official method
    # 5. Stream response back chunk by chunk using the official method
    try:
        response = client.models.generate_content_stream(
            model='gemini-2.5-flash',  # <--- Upgraded active model!
            contents=formatted_contents,
            config=config
        )
        for chunk in response:
            if chunk and hasattr(chunk, 'text') and chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"API Error: Please check your configuration backend. Details: {str(e)}"


def get_contextual_prompt(section: str) -> str:
    """Return a pre-built prompt based on the current dashboard section."""
    prompts = {
        "overview": (
            "Give me a compelling 3-sentence overview of what this Ghana election "
            "topic modeling project is about and why it matters."
        ),
        "data": (
            "Explain how the data was collected from the 5 Ghanaian media outlets "
            "and what makes this corpus analytically interesting."
        ),
        "preprocessing": (
            "Walk me through the NLP preprocessing pipeline used in this project "
            "and explain why each step matters for topic modeling."
        ),
        "lda": (
            "Explain how LDA topic modeling works using a simple analogy, "
            "then tell me what the coherence score optimization revealed about "
            "Ghana's election media coverage."
        ),
        "topics": (
            "Which topics dominated Ghana's 2024 election coverage and what does "
            "that tell us about the media's agenda-setting role?"
        ),
        "outlets": (
            "Compare and contrast how state-owned outlets (Daily Graphic, Ghanaian Times) "
            "covered the election versus private digital outlets (MyJoyOnline, Citinewsroom)."
        ),
        "narrow_broad": (
            "Explain the narrow vs. broad coverage framework and what the results "
            "reveal about journalism quality in Ghana's 2024 election."
        ),
        "party": (
            "How did media framing of the NPP and NDC differ, and how does this "
            "connect to the actual election outcome where Mahama won decisively?"
        ),
        "conclusion": (
            "Give me the three most important takeaways from this entire analysis "
            "and their implications for Ghanaian democracy and media studies."
        ),
    }
    return prompts.get(section, "What would you like to know about this analysis?")


SUGGESTED_QUESTIONS = [
    "Why did Mahama win and what did the media have to do with it?",
    "How does LDA decide what a 'topic' is?",
    "Was galamsey the most important issue to voters?",
    "Are state-owned media outlets biased toward the government?",
    "What's the difference between LDA and BERTopic?",
    "How did the economy dominate media coverage?",
    "Which outlet had the most balanced coverage?",
    "What is narrow vs broad coverage and why does it matter?",
    "How were topics labeled after the model ran?",
    "What was the single biggest finding of this research?",
]