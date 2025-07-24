import os, base64, json
from azure.openai import AzureOpenAI           # pip install azure-openai>=1.4.0

# --- 0.  Client --------------------------------------------------------------
client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_KEY"],
    api_version="2024-02-15-preview",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
)

DEPLOYMENT = "gpt-4o-vision"                   # your model deployment name

# --- 1.  Helper: convert base64 → data URI -----------------------------------
def to_data_uri(b64_str: str, mime="image/png") -> str:
    return f"data:{mime};base64,{b64_str}"

# --- 2.  Vision-NER call -----------------------------------------------------
SYSTEM_PROMPT = """
You are an NER assistant. 
Return a JSON object with keys person, org, location, product.
Each value is an array of UNIQUE strings (exact surface form).
If none, return an empty array. Do not add extra keys.
"""

def ner_from_image_b64(b64_img: str) -> dict:
    chat_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            # multimodal array: first our instructions…
            {"type": "text",
             "text": "Extract the entities you see/hear in this image. "
                     "If text is present, read it first."},
            # …then the image itself
            {"type": "image_url",
             "image_url": {
                 "url": to_data_uri(b64_img),
                 "detail": "auto"        # low | high | auto
             }}
        ]}
    ]

    rsp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=chat_messages,
        max_tokens=256,
        temperature=0.0,
    )

    return json.loads(rsp.choices[0].message.content)

# --- 3.  Your existing grounding step ---------------------------------------
def ground_entities(entities: dict, db_session):
    """Look up the entities in your DB and enrich with IDs, metadata, …"""
    # unchanged from your text pipeline
    ...

# --- 4.  Usage ---------------------------------------------------------------
def process_image_request(b64_img: str, db_session):
    raw_entities = ner_from_image_b64(b64_img)
    return ground_entities(raw_entities, db_session)
