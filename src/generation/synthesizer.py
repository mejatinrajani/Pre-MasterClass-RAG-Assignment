import logging
from groq import Groq
from src.config import Config

logger = logging.getLogger(__name__)

class RAGSynthesizer:
    def __init__(self):
        api_key = Config.GROQ_API_KEY
        self.client = Groq(api_key=api_key) if api_key and api_key != "your-groq-api-key-here" else None
        self.model = Config.REASONING_LLM

    def build_prompt(self, query: str, context_chunks: list, chat_history: list) -> str:
        formatted_context = ""
        for i, chunk in enumerate(context_chunks):
            source = chunk.get("source", "Unknown Source")
            content = chunk.get("content", chunk.get("text", "No content provided."))
            formatted_context += f"--- Document {i+1} ({source}) ---\n{content}\n\n"

        # Check if this is the start of a conversation
        is_first_interaction = len(chat_history) <= 1

        prompt = f"""You are the official AI assistant for Bastian Beach Club. 
        
        BEHAVIORAL RULES:
        1. GREETING: If this is the first interaction (is_first_interaction={is_first_interaction}), be polite and welcoming. If not, skip the long greeting and answer the query immediately.
        2. RAG FIRST: For any question, check the PROVIDED CONTEXT first. Use information ONLY from the provided context.
        3. NO HALLUCINATIONS: If the answer is not in the context, say so clearly. Do not offer general information if it isn't in the provided sources.
        4. CITATIONS: Always cite the source (e.g.,).

        PROVIDED CONTEXT:
        {formatted_context if formatted_context else "No relevant context found."}

        USER QUERY: {query}
        """
        return prompt

    def generate_answer(self, query: str, retrieval_result: dict, chat_history: list) -> dict:
        context = retrieval_result.get("context_chunks", [])
        engine_used = retrieval_result.get("engine_used", "None")
        
        prompt = self.build_prompt(query, context, chat_history)
        
        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a professional Bastian Beach Club AI assistant."},
                {"role": "user", "content": prompt}
            ],
            model=self.model,
            temperature=0.1, # Lowered for more precise, less "chatty" responses
        )
        
        return {
            "answer": chat_completion.choices[0].message.content,
            "engine_used": engine_used
        }