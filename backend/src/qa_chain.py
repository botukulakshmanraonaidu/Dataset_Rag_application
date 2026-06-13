import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langchain_core.retrievers import BaseRetriever
from src.vector_store import load_vector_store

load_dotenv()
logger = logging.getLogger(__name__)


def get_qa_chain(retriever: BaseRetriever, settings: dict = None):
    """
    Creates a RetrievalQA chain using the provided retriever (can be FAISS or Hybrid).
    """
    if settings is None:
        settings = {}

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", None)
    
    # Use dynamic settings or fall back to env/defaults
    model_name = settings.get("model_name") or os.getenv("MODEL_NAME", "openai/gpt-4o")
    temperature = settings.get("temperature", 0.0)
    max_tokens = settings.get("max_tokens") or int(os.getenv("MAX_TOKENS", "300"))

    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in .env file!")

    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,        # Explicitly limit tokens to avoid 402 error
        openai_api_key=api_key,
        openai_api_base=base_url,
    )
    
    from langchain_core.prompts import PromptTemplate
    from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
    from langchain_core.documents import Document

    # 1. Configuration
    template = """You are an intelligent assistant designed to answer user queries using a Hybrid Retrieval-Augmented Generation (RAG) system.

🔍 Retrieval Strategy
Analyze the provided context (Vector search + BM25 + Web Fallback).
Combine results intelligently.

🧠 Answering Rules
1. Use ONLY the provided context. Do NOT hallucinate.
2. Your response MUST consist of EXACTLY four (4) lines of text separated by newline characters. Do not output a single paragraph; output exactly 4 separate lines.
3. Keep the output extremely dense and concise to minimize token usage.
4. If using Web Search results, one of the lines MUST state: "Source: Web Search (DuckDuckGo)".
5. If information is missing completely, your response must be exactly:
"I don't have enough information in the provided data.
Please verify your question or try another query.
No local documents matched this search.
Source: System Database."

Context:
{context}

Question: {question}

Helpful Answer (EXACTLY 4 lines):"""
    
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    # 2. Base RAG Chain
    base_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    # 3. Fallback Wrapper
    class HybridFallbackChain:
        def __init__(self, local_chain, web_tool, llm, prompt_template):
            self.local_chain = local_chain
            self.web_tool = web_tool
            self.llm = llm
            self.prompt_template = prompt_template

        def invoke(self, inputs):
            # Try local retrieval first
            result = self.local_chain.invoke(inputs)
            source_docs = result.get("source_documents", [])
            answer = str(result.get("result", ""))
            
            logger.debug(f"LOCAL RAG ANSWER: {answer}")
            for i, doc in enumerate(source_docs):
                logger.debug(f"LOCAL RAG DOC {i+1}: {doc.metadata.get('source', 'Unknown')} - {doc.page_content[:150]}...")
            
            # Fallback if no docs found or if the answer indicates missing info
            normalized_answer = answer.lower().replace("’", "'")
            is_insufficient = (
                not source_docs or 
                "don't specify" in normalized_answer or 
                "don't have enough information" in normalized_answer or
                "do not have enough information" in normalized_answer or
                len(answer.strip()) < 5
            )
            
            if is_insufficient:
                query = inputs.get("query") or inputs.get("text")
                logger.info(f"Local RAG insufficient (Docs: {len(source_docs)}). Falling back to Web Search: {query}")
                
                try:
                    # Attempt Web Search
                    web_context = self.web_tool.run(query)
                    if web_context and len(web_context) > 20:
                        logger.info("Web Search successful. Summarizing results...")
                        formatted_prompt = self.prompt_template.format(context=web_context, question=query)
                        web_response = self.llm.invoke(formatted_prompt).content
                        return {
                            "result": web_response,
                            "source_documents": [Document(page_content=web_context, metadata={"source": "Web Search (DuckDuckGo)"})]
                        }
                    else:
                        logger.warning(f"Web Search returned insufficient data: {web_context}")
                except Exception as e:
                    logger.error(f"Web Search tool failed: {e}")
            
            return result

    ddg_search = DuckDuckGoSearchRun()
    return HybridFallbackChain(base_qa, ddg_search, llm, QA_CHAIN_PROMPT)

if __name__ == "__main__":
    # Example logic
    # vs = load_vector_store()
    # chain = get_qa_chain(vs)
    pass
