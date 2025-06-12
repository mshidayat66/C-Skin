import chainlit as cl
from sentence_transformers import SentenceTransformer
import torch
import qdrant_client
from langchain.llms import Ollama
import logging
import time
from langdetect import detect
from typing import Optional, Dict
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
import os

# ----------------------------
# Logging Configuration
# ----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------------------
# Chainlit Data Layer Setup
# ----------------------------
@cl.data_layer
def get_data_layer():
    return SQLAlchemyDataLayer(conninfo=os.environ["DATABASE_URL"])

# ----------------------------
# Model & Client Initialization
# ----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {device}")

model = SentenceTransformer('BAAI/bge-m3', device=device)
client = qdrant_client.QdrantClient(os.environ["QDRANT_URL"])
llama = Ollama(base_url=os.environ["LLM_URL"], model="llama3.1:8b", temperature=0.0, mirostat_tau=4.0, mirostat_eta=0.65)

# ----------------------------
# Function: Language Detection & Translation
# ----------------------------
def detect_and_translate(text):
    try:
        detected_lang = detect(text)
        logger.info(f"Detected language: {detected_lang}")
        if detected_lang == "en":
            return text
        else:
            translation_prompt = f"""
Translate the following sentence from {detected_lang} to English. Do not add explanation, just translate.

Original ({detected_lang}): {text}
English:
"""
            translated = llama.invoke(translation_prompt)
            translated = translated.strip()
            if not translated:
                logger.warning("Translation result is empty. Returning original text.")
                return text
            logger.info(f"Translation result: {translated}")
            return translated
    except Exception as e:
        logger.warning(f"Language detection or translation failed: {e}")
        return text

# ----------------------------
# Function: Qdrant Search
# ----------------------------
def search(query):
    logger.info(f"Searching Qdrant for query: {query}")
    try:
        start_time = time.time()  # Mulai stopwatch

        query_vector = model.encode(query).tolist()
        results = client.search(
            collection_name='Skin Diseases',
            query_vector=query_vector,
            limit=5,
            with_payload=True,
            score_threshold=0.4
        )

        duration = time.time() - start_time  # Hitung durasi
        logger.info(f"Qdrant search took {duration:.2f} seconds and found {len(results)} results")

        final_results = []
        for res in sorted(results, key=lambda x: x.score, reverse=True):
            payload = res.payload
            combined = f"Q: {payload.get('question', '')}\nA: {payload.get('answer', '')}\nSource: {payload.get('source', '')}\nFocus Area: {payload.get('focus_area', '')}"
            final_results.append(combined.strip())
        return final_results
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []
    
# ----------------------------
# Function: Generate Response Using Ollama
# ----------------------------
def generate_response_with_ollama(context, query, tone="professional and friendly"):
    try:
        start_time = time.time()
        context_text = "\n\n".join([f"Doc {i+1}:\n{ctx.strip()}" for i, ctx in enumerate(context)])
        prompt = f"""
Anda adalah chatbot kesehatan bernama C-Skin. Silakan jawab setiap pertanyaan pengguna dengan nada: {tone}.

Petunjuk untuk menjawab pertanyaan terkait penyakit:
- Jawablah hanya menggunakan informasi yang tersedia dalam {context}.
- Jika pertanyaan berkaitan dengan penyakit, berikan informasi yang akurat, ringkas, dan lengkap berdasarkan konteks.
- Jika tidak ditemukan informasi yang relevan dalam konteks, jangan berspekulasi atau mengarang jawaban. Sebagai gantinya, balas dengan: "Ini adalah semua informasi yang saya miliki."
- Hindari penggunaan istilah medis yang rumit atau tidak umum. Gunakan bahasa yang sederhana dan mudah dipahami oleh masyarakat umum.
- Selalu awali jawaban Anda dengan: "Terima kasih telah berkonsultasi dengan C-Skin."
- Selalu akhiri jawaban Anda dengan: "Semoga informasi ini bermanfaat, lekas sembuh, dan terima kasih."
- Jangan gunakan pengetahuan di luar konteks yang diberikan.
- Jangan menyebutkan bahwa Anda terbatas oleh konteks — cukup berikan jawaban sesuai informasi yang tersedia.
- Anda tidak boleh menyimpulkan atau menambahkan fakta yang tidak disebutkan secara eksplisit dalam konteks. Hanya merangkum atau mengungkapkan ulang apa yang memang ada.

Gunakan hanya konteks berikut untuk menjawab pertanyaan pengguna:
{context}

Pertanyaan Pengguna:
{query}

Jawab hanya dalam Bahasa Indonesia.
"""
        result = llama.invoke(prompt, stream=False)
        duration = time.time() - start_time
        logger.info(f"Ollama response generation took {duration:.2f} seconds")
        logger.info(f"Generated response preview: {result[:100]}...")
        return result.strip()
    except Exception as e:
        logger.error(f"Ollama response generation failed: {e}")
        return "Terjadi kesalahan saat menghasilkan jawaban. Silakan coba lagi."

# ----------------------------
# Async Wrappers
# ----------------------------
search_async = cl.make_async(search)
generate_response_async = cl.make_async(generate_response_with_ollama)

# ----------------------------
# OAuth / Login
# ----------------------------
@cl.oauth_callback
def oauth_callback(provider_id: str, token: str, raw_user_data: Dict[str, str], default_user: cl.User) -> Optional[cl.User]:
    try:
        email = raw_user_data.get("email") or f"{raw_user_data.get('login', 'unknown')}@github.com"
        name = raw_user_data.get("name") or raw_user_data.get("login") or "Anonymous"
        logger.info(f"OAuth login success: {email} ({name})")
        return cl.User(identifier=email, metadata={"name": name})
    except Exception as e:
        logger.error(f"OAuth callback error: {e}")
        return None

# ----------------------------
# Chat Handler
# ----------------------------
@cl.on_message
async def main(message: cl.Message):
    try:
        logger.info("Handling user message")

        # Simpan input asli pengguna
        original_query = message.content

        # Terjemahkan untuk pencarian & respon
        query = await cl.make_async(detect_and_translate)(original_query)
        logger.info(f"Translated query: {query}")

        # Ambil atau inisialisasi history dari session
        history = cl.user_session.get("history", [])
        history.append({"role": "user", "text": original_query})

        # Lakukan pencarian berdasarkan hasil terjemahan
        results = await search_async(query)
        if not results:
            await cl.Message(content="Tidak ditemukan informasi yang relevan.", author="C-Skin Chatbot").send()
            return

        # Buat jawaban berdasarkan hasil pencarian
        response = await generate_response_async(results, query)

        # Simpan respon ke history
        history.append({"role": "assistant", "text": response})
        cl.user_session.set("history", history)

        await cl.Message(
            content=response,
            author="C-Skin Chatbot"
        ).send()

    except Exception as e:
        logger.error(f"Unexpected error in main handler: {e}")
        await cl.Message(content="Terjadi kesalahan internal. Silakan coba lagi.", author="C-Skin Chatbot").send()

# ----------------------------
# Resume Chat Handler
# ----------------------------
@cl.on_chat_resume
async def on_chat_resume(thread):
    logger.info("Chat resumed")

    if thread is None or not isinstance(thread, dict):
        logger.warning("Invalid or missing thread. Resetting history.")
        cl.user_session.set("history", [])
        await cl.Message(content="Selamat datang kembali di C-Skin!").send()
        return

    history = cl.user_session.get("history", [])

    # Hanya tampilkan balasan dari chatbot
    for msg in history:
        if msg["role"] == "assistant":
            await cl.Message(content=msg["text"], author="C-Skin Chatbot").send()

        
# ----------------------------
# Feedback Handler
# ----------------------------
#@cl.on_feedback
#async def on_feedback(feedback):
#    message_id = getattr(feedback, "message_id", None)
#    thread_id = getattr(feedback, "thread_id", None)
#    score = getattr(feedback, "score", None)
#    comment = getattr(feedback, "comment", "")

#    print(f"Feedback diterima: message_id={message_id}, thread_id={thread_id}, score={score}, comment={comment}")
   
