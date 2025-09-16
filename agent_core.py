# agent_core.py - DevAgent Backend with Kimi Web Search Integration
import modal
import os
import shutil
import requests
import zipfile
import io
import json
import asyncio
from sqlalchemy import create_engine, text
from pathlib import Path
from pydantic import BaseModel
from fastapi import FastAPI
import time
import re


# --- Configuration and Setup ---
image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
)

app = modal.App(
    name="dev-agent-p",
    image=image,
    secrets=[
        modal.Secret.from_name("tidb-secret"),
        modal.Secret.from_name("kimi-secret"),
    ]
)

repo_volume = modal.Volume.from_name("dev-agent-repo-cache", create_if_missing=True)
model_cache_volume = modal.Volume.from_name("dev-agent-model-cache", create_if_missing=True)


# --- Utility Functions ---
def normalize_question_backend(question: str) -> str:
    """Match frontend normalization exactly"""
    cleaned = re.sub(r'[^\w\s]', '', question.lower())
    return ' '.join(cleaned.split()).strip()


# --- Core Services ---
class LLMService:
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.environ.get("KIMI_API_KEY"),
            base_url="https://api.moonshot.ai/v1"
        )

    def generate_response(self, prompt: str, system_prompt: str, model: str = "moonshot-v1-8k", json_mode: bool = False, max_tokens: int = None):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": 0.3,
            "timeout": 30  # 30 second timeout
        }
        
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        
        try:
            completion = self.client.chat.completions.create(**kwargs)
            return completion.choices[0].message.content
        except Exception as e:
            print(f"LLM API Error: {e}")
            if json_mode:
                return '{"intent": "summary", "error": "timeout"}'
            return "I apologize, but I'm having trouble processing your request right now due to high load. Please try again."


class DatabaseService:
    def __init__(self):
        self.engine = create_engine(os.environ["DATABASE_URL"],
                                  connect_args={"ssl_ca": "/etc/ssl/certs/ca-certificates.crt"},
                                  pool_pre_ping=True,
                                  pool_recycle=300)

    def get_connection(self):
        return self.engine.connect()
    
    def get_repository_by_url(self, git_url: str):
        with self.get_connection() as conn:
            stmt = text("SELECT id, status FROM repositories WHERE git_url = :url")
            return conn.execute(stmt, {"url": git_url}).mappings().first()

    def create_repository(self, git_url: str) -> int:
        with self.get_connection() as conn:
            stmt = text("INSERT INTO repositories (git_url, status) VALUES (:url, 'downloading') ON DUPLICATE KEY UPDATE status='downloading'")
            conn.execute(stmt, {"url": git_url}); conn.commit()
            return conn.execute(text("SELECT id FROM repositories WHERE git_url = :url"), {"url": git_url}).scalar_one()

    def update_ingestion_progress(self, repo_id: int, status: str, processed: int, total: int):
        with self.get_connection() as conn:
            stmt = text("UPDATE repositories SET progress_status = :status, files_processed = :processed, total_files = :total WHERE id = :id")
            conn.execute(stmt, {"status": status, "processed": processed, "total": total, "id": repo_id}); conn.commit()
            
    def set_repository_status(self, repo_id: int, status: str):
        with self.get_connection() as conn:
            stmt = text("UPDATE repositories SET status = :status WHERE id = :id")
            conn.execute(stmt, {"status": status, "id": repo_id}); conn.commit()

    def add_source_file_and_chunks(self, repo_id: int, file_path: str, chunks: list[str], embeddings: list[list[float]]):
        with self.get_connection() as conn:
            with conn.begin():
                stmt_file = text("INSERT INTO source_files (repo_id, file_path) VALUES (:repo_id, :path)")
                conn.execute(stmt_file, {"repo_id": repo_id, "path": file_path})
                file_id = conn.execute(text("SELECT LAST_INSERT_ID()")).scalar_one()
                stmt_chunk = text("INSERT INTO code_chunks (file_id, chunk_text, embedding) VALUES (:file_id, :text, :embedding)")
                chunk_data = [{"file_id": file_id, "text": chunk, "embedding": str(embedding)} for chunk, embedding in zip(chunks, embeddings)]
                if chunk_data: conn.execute(stmt_chunk, chunk_data)

    def vector_search_chunks(self, query_embedding: list[float], repo_id: int, top_k: int = 15):
        with self.get_connection() as conn:
            stmt = text("SELECT c.chunk_text, sf.file_path FROM code_chunks c JOIN source_files sf ON c.file_id = sf.id WHERE sf.repo_id = :repo_id ORDER BY c.embedding <=> :embedding LIMIT :top_k")
            return conn.execute(stmt, {"repo_id": repo_id, "embedding": str(query_embedding), "top_k": top_k}).mappings().all()


# --- AI Agent Functions ---
@app.function(retries=2)
def generate_repository_blueprint(repo_path: str, file_list: list[str]):
    llm_service = LLMService()
    system_prompt_stage1 = "You are an expert software architect. Analyze the provided file structure and generate a concise, high-level overview of the repository's purpose and architecture. Do NOT guess the specific libraries used."
    file_list_str = "\n".join(file_list[:200]) + ("..." if len(file_list) > 200 else "")
    prompt_stage1 = f"File structure:\n\n{file_list_str}"
    overview = llm_service.generate_response(prompt=prompt_stage1, system_prompt=system_prompt_stage1, model="moonshot-v1-32k")
    
    tech_stack_context = ""
    config_files = ['requirements.txt', 'package.json', 'pom.xml', 'build.gradle', 'Gemfile', 'go.mod']
    for config_file_name in config_files:
        found_relative_path = next((f for f in file_list if f.endswith(f"/{config_file_name}") or f == f"/{config_file_name}"), None)
        if found_relative_path:
            full_path_to_read = os.path.join(repo_path, found_relative_path.lstrip('/'))
            try:
                with open(full_path_to_read, 'r', encoding='utf-8') as f: content = f.read()
                tech_stack_context += f"\n\n--- Content of {config_file_name} ---\n{content}"
            except Exception as e:
                print(f"Warning: Could not read config file {full_path_to_read}: {e}")
    
    final_blueprint = overview
    if tech_stack_context:
        system_prompt_stage2 = "You are a tech stack analyst. Based on the following configuration file contents, list the specific frameworks and key libraries used in this project. Be factual."
        tech_stack_analysis = llm_service.generate_response(prompt=tech_stack_context, system_prompt=system_prompt_stage2, model="moonshot-v1-8k")
        final_blueprint += "\n\n### Detected Technology Stack\n" + tech_stack_analysis
    return final_blueprint


@app.function(gpu="l4", volumes={"/root/.cache/huggingface": model_cache_volume}, timeout=120)
def embed_chunks_batch(chunks: list[str]):
    """Optimized embedding with smaller batches"""
    from sentence_transformers import SentenceTransformer
    model_cache_volume.reload()
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
    
    # Process in smaller batches to avoid memory issues
    batch_size = 32
    all_embeddings = []
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False, batch_size=16)
        all_embeddings.extend(batch_embeddings.tolist())
    
    return all_embeddings


@app.function(retries=2, timeout=30)
def run_dispatcher(question: str) -> str:
    """Enhanced question classification including web search detection"""
    question_lower = question.lower()
    
    # Web search indicators
    web_search_keywords = [
        "deploy", "hosting", "aws", "azure", "gcp", "docker", "kubernetes", "ci/cd",
        "latest", "current", "recent", "new version", "updated",
        "best practices", "recommended", "how to", "tutorial",
        "similar projects", "alternatives", "community",
        "documentation", "guide", "examples", "npm install", "pip install"
    ]
    
    if any(keyword in question_lower for keyword in web_search_keywords):
        return "web_search"
    
    # Existing classifications
    security_keywords = ['security', 'vulnerability', 'exploit', 'attack', 'breach', 'secure', 'auth', 'password', 'token', 'injection', 'xss', 'csrf', 'vulnerable']
    refactor_keywords = ['refactor', 'improve', 'optimize', 'clean', 'better', 'performance', 'efficiency', 'maintainability', 'code quality', 'best practice', 'optimization']
    
    if any(keyword in question_lower for keyword in security_keywords):
        return "security"
    elif any(keyword in question_lower for keyword in refactor_keywords):
        return "refactor"
    
    # For ambiguous cases, use small fast model
    llm_service = LLMService()
    system_prompt = '''Classify this question into ONE category:
- "summary" for general questions about code purpose, functionality, or how things work
- "security" for questions about vulnerabilities, security issues, or safety concerns  
- "refactor" for questions about code improvements, optimization, or best practices
- "web_search" for questions needing external/current information like deployment, hosting, or latest practices

Respond with only: summary, security, refactor, or web_search'''
    
    try:
        response = llm_service.generate_response(question, system_prompt, "moonshot-v1-8k", max_tokens=10)
        intent = response.strip().lower()
        return intent if intent in ["summary", "security", "refactor", "web_search"] else "summary"
    except Exception: 
        return "summary"


@app.function(retries=2, timeout=90)
def run_web_search_agent(question: str, context: str = ""):
    """🌐 NEW: Enhanced agent with Kimi web search for current information"""
    from openai import OpenAI
    
    print(f"🌐 Starting web search agent for: {question[:50]}...")
    
    try:
        client = OpenAI(
            base_url="https://api.moonshot.ai/v1",
            api_key=os.environ.get("KIMI_API_KEY")
        )
    except Exception as e:
        print(f"Failed to initialize Kimi client: {e}")
        # Fallback to regular deep thinker
        if context:
            llm_service = LLMService()
            return llm_service.generate_response(question, "You are a helpful programming assistant. Answer based on the provided context.", "moonshot-v1-8k")
        return "I encountered an error accessing current information. Please try asking about specific code in the repository."
    
    tools = [
        {
            "type": "builtin_function",
            "function": {
                "name": "$web_search",
            },
        }
    ]
    
    # Enhanced system prompt
    system_prompt = f"""You are DevAgent, an expert AI programmer with web search capabilities.

CODE CONTEXT: {context[:800] if context else "No specific code context provided - this appears to be a general question."}

INSTRUCTIONS:
1. Use web search to find current, accurate information
2. Combine web search results with code context when relevant
3. Provide practical, actionable answers with specific steps
4. Focus on current best practices and up-to-date information
5. ALWAYS respond in English
6. Keep responses comprehensive but focused (max 500 words)
7. Include links or references when helpful

If the question is about deployment, hosting, or external integrations, prioritize web search results.
If it combines code context with external knowledge, integrate both seamlessly."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    
    max_iterations = 4
    current_iteration = 0
    
    while current_iteration < max_iterations:
        try:
            print(f"🔄 Web search iteration {current_iteration + 1}")
            
            completion = client.chat.completions.create(
                model="kimi-k2-0711-preview",  # Using the same model as your test
                messages=messages,
                tools=tools,
                temperature=0.2
            )
            
            choice = completion.choices[0]
            messages.append(choice.message.model_dump())
            
            if choice.finish_reason != "tool_calls":
                # Final answer received
                print(f"✅ Web search completed with final answer")
                return choice.message.content
            
            # Process tool calls
            print(f"🔧 Processing {len(choice.message.tool_calls)} tool calls")
            for tool_call in choice.message.tool_calls:
                if tool_call.function.name == "$web_search":
                    try:
                        tool_result = json.loads(tool_call.function.arguments)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": "$web_search",
                            "content": json.dumps(tool_result),
                        })
                        print(f"✅ Web search tool executed successfully")
                    except Exception as e:
                        print(f"❌ Web search tool error: {e}")
                        # Continue without this tool result
            
            current_iteration += 1
            
        except Exception as e:
            print(f"💥 Web search agent iteration error: {e}")
            current_iteration += 1
            
            if current_iteration >= max_iterations:
                # Final fallback
                if context:
                    llm_service = LLMService()
                    fallback_prompt = f"Answer this question based on the code context: {question}\n\nContext: {context[:500]}"
                    return llm_service.generate_response(fallback_prompt, "You are a helpful programming assistant.", "moonshot-v1-8k")
                else:
                    return "I encountered difficulties accessing current web information. Please try asking about specific code in the repository or rephrase your question."
    
    # If we've exhausted iterations, return the last assistant message if available
    for msg in reversed(messages):
        if msg.get('role') == 'assistant' and msg.get('content'):
            return msg['content']
    
    return "Unable to complete web search after multiple attempts. Please try asking about specific code in the repository."


@app.function(retries=2, timeout=45)
def run_deep_thinker(question: str, context: str):
    """Optimized deep thinker with shorter responses"""
    llm_service = LLMService()
    system_prompt = """You are DevAgent, an expert AI programmer. Answer the user's question based on the provided code context. 

RULES:
1. Primarily use information from the provided code context
2. Keep answers concise but informative (max 300 words for prefetch)
3. Focus on the most important insights
4. Use bullet points for clarity when listing multiple items
5. If the context doesn't contain direct information, analyze patterns and provide insights based on what you can see

Be helpful and provide valuable insights based on what you can observe in the code."""
    
    # Check if this is a context-aware question
    if "Context:" in question and "Current Question:" in question:
        user_prompt = f"<code_context>\n{context}\n</code_context>\n\n{question}\n\nAnswer based on the code context above. Keep response focused and under 300 words."
    else:
        user_prompt = f"<context>\n{context}\n</context>\n\nQuestion: {question}\n\nAnswer based on the context above. Keep response focused and under 300 words."
    
    return llm_service.generate_response(user_prompt, system_prompt, "moonshot-v1-8k", max_tokens=400)


@app.function(retries=2, timeout=30)
def run_security_guardian(context: str):
    llm_service = LLMService()
    system_prompt = """You are a senior security engineer. Analyze ONLY the provided code for vulnerabilities.

CRITICAL RULES:
1. ONLY analyze the code that is explicitly shown in the context
2. Keep analysis concise (max 250 words)
3. List vulnerabilities with severity ratings based on what you can see
4. If insufficient context, say "Insufficient code context to perform comprehensive security analysis"
5. Do NOT hallucinate potential vulnerabilities"""
    
    user_prompt = f"<code_context>\n{context}\n</code_context>\n\nAnalyze ONLY the code shown above for security vulnerabilities. Keep response under 250 words."
    return llm_service.generate_response(user_prompt, system_prompt, "moonshot-v1-8k", max_tokens=350)


@app.function(retries=2, timeout=30)
def run_refactoring_agent(context: str):
    llm_service = LLMService()
    system_prompt = """You are a principal engineer specializing in code quality. Analyze ONLY the provided code snippets for potential improvements.

CRITICAL RULES:
1. ONLY suggest refactorings based on the code explicitly shown
2. Keep suggestions concise (max 250 words) 
3. Focus on what you can actually see: code structure, naming, patterns
4. Format as bulleted list
5. Be specific about which parts of the shown code you're referring to"""
    
    user_prompt = f"<code_context>\n{context}\n</code_context>\n\nAnalyze ONLY the code shown above and suggest improvements. Keep response under 250 words."
    return llm_service.generate_response(user_prompt, system_prompt, "moonshot-v1-8k", max_tokens=350)


@app.function(retries=2, timeout=20)
def run_general_question_agent(question: str):
    """Handle general technical questions that don't require specific code context"""
    llm_service = LLMService()
    system_prompt = """You are a senior software engineer and technical consultant. Answer technical questions with accurate, helpful information.

GUIDELINES:
1. Provide clear, practical answers (max 300 words)
2. Include key challenges and considerations when relevant
3. Be honest about limitations
4. Structure answer clearly with bullet points when helpful
5. Focus on actionable insights"""
    
    user_prompt = f"Question: {question}\n\nProvide a comprehensive, helpful answer (max 300 words)."
    return llm_service.generate_response(user_prompt, system_prompt, "moonshot-v1-8k", max_tokens=400)


@app.function(retries=2, timeout=30)
def generate_smart_questions_from_blueprint(blueprint: str, repo_id: int):
    """Generate project-specific smart questions based on the repository blueprint"""
    llm_service = LLMService()
    system_prompt = """You are an expert code analyst. Based on the repository blueprint provided, generate 6 smart, specific questions that would be most valuable for understanding this codebase. 

Categories to cover:
- ARCHITECTURE: About overall design and structure
- IMPLEMENTATION: About specific technical implementations  
- DEBUGGING: About potential issues or error handling
- OPTIMIZATION: About performance improvements
- SECURITY: About security considerations
- TESTING: About testing strategies

Return ONLY a valid JSON array with this exact format:
[
  {"category": "ARCHITECTURE", "question": "specific question here"},
  {"category": "IMPLEMENTATION", "question": "specific question here"},
  {"category": "DEBUGGING", "question": "specific question here"},
  {"category": "OPTIMIZATION", "question": "specific question here"},
  {"category": "SECURITY", "question": "specific question here"},
  {"category": "TESTING", "question": "specific question here"}
]"""
    
    user_prompt = f"Repository Blueprint:\n{blueprint}\n\nGenerate 6 smart questions specific to this codebase:"
    
    try:
        response = llm_service.generate_response(user_prompt, system_prompt, "moonshot-v1-8k", json_mode=True, max_tokens=500)
        questions = json.loads(response)
        return questions
    except Exception as e:
        print(f"Error generating smart questions: {e}")
        # Fallback to generic questions
        return [
            {"category": "ARCHITECTURE", "question": "What is the main purpose and architecture of this code?"},
            {"category": "IMPLEMENTATION", "question": "How are the core components implemented?"},
            {"category": "DEBUGGING", "question": "What are potential error states in the main logic?"},
            {"category": "OPTIMIZATION", "question": "How can performance be improved?"},
            {"category": "SECURITY", "question": "Are there any security vulnerabilities?"},
            {"category": "TESTING", "question": "What tests should be written for this code?"}
        ]


@app.function(retries=1, timeout=10)
def generate_followup_questions_fast(original_question: str):
    """Generate follow-up questions with fast keyword-based approach"""
    question_lower = original_question.lower()
    
    if "deploy" in question_lower or "hosting" in question_lower:
        return [
            "What about CI/CD pipeline setup?",
            "How to handle environment variables?",
            "What monitoring should I set up?"
        ]
    elif "purpose" in question_lower or "what" in question_lower:
        return [
            "How is this implemented?",
            "What are the key components?",
            "Are there any limitations?"
        ]
    elif "security" in question_lower or "vulnerability" in question_lower:
        return [
            "How can these issues be fixed?",
            "What are the security best practices?",
            "Are there other potential vulnerabilities?"
        ]
    elif "improve" in question_lower or "optimize" in question_lower:
        return [
            "What would be the impact of these changes?",
            "Are there any trade-offs to consider?",
            "How should I prioritize these improvements?"
        ]
    elif "architecture" in question_lower:
        return [
            "What are the main design patterns used?",
            "How do components interact?",
            "What are potential scalability issues?"
        ]
    else:
        return [
            "Tell me more about this",
            "How does this work?",
            "What should I know next?"
        ]


# --- Ingestion Orchestrator ---
@app.function(volumes={"/tmp/dev-agent": repo_volume}, timeout=1800)
def ingest_repository(repo_url: str) -> dict:
    db_service = DatabaseService()
    clean_repo_url = repo_url.removesuffix('.git')
    repo_id = db_service.create_repository(clean_repo_url)
    repo_path = f"/tmp/dev-agent/{repo_id}"
    blueprint = ""
    
    try:
        main_zip_url = f"{clean_repo_url}/archive/refs/heads/main.zip"
        response = requests.get(main_zip_url, stream=True)
        if response.status_code == 404:
            master_zip_url = f"{clean_repo_url}/archive/refs/heads/master.zip"
            response = requests.get(master_zip_url, stream=True)
        response.raise_for_status()

        if os.path.exists(repo_path): shutil.rmtree(repo_path)
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            top_level_dir = z.namelist()[0]
            z.extractall(repo_path)
        extracted_path = os.path.join(repo_path, top_level_dir)
        
        file_list = [os.path.join(r, f).replace(extracted_path, "") for r, _, fs in os.walk(extracted_path) for f in fs]
        total_files = len(file_list)
        db_service.update_ingestion_progress(repo_id, "Generating blueprint...", 0, total_files)
        
        blueprint_call = generate_repository_blueprint.spawn(extracted_path, file_list)
        
        processed_files = 0
        for file_path in file_list:
            if any(file_path.endswith(ext) for ext in ['.py', '.js', '.ts', '.md', '.txt', '.html', '.css']):
                try:
                    full_path = os.path.join(extracted_path, file_path.lstrip('/'))
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f: content = f.read()
                    chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
                    if chunks:
                        embedding_call = embed_chunks_batch.spawn(chunks)
                        embeddings = embedding_call.get()
                        db_service.add_source_file_and_chunks(repo_id, file_path, chunks, embeddings)
                except Exception as e:
                    print(f"Warning: Could not process file {file_path}: {e}")
            processed_files += 1
            db_service.update_ingestion_progress(repo_id, f"Processing file {processed_files}/{total_files}: {file_path}", processed_files, total_files)
        
        db_service.set_repository_status(repo_id, "completed")
        blueprint = blueprint_call.get()
        
        # Store blueprint in database
        try:
            with db_service.get_connection() as conn:
                stmt = text("UPDATE repositories SET blueprint = :blueprint WHERE id = :id")
                conn.execute(stmt, {"blueprint": blueprint, "id": repo_id})
                conn.commit()
        except Exception as e:
            print(f"Warning: Could not store blueprint: {e}")
        
        return {"repo_id": repo_id, "blueprint": blueprint}
    except Exception as e:
        db_service.set_repository_status(repo_id, "failed")
        return {"repo_id": repo_id, "blueprint": f"Failed to ingest: {e}"}


# --- FastAPI Application ---
web_app = FastAPI()

class IngestRequest(BaseModel):
    repo_url: str

class AskRequest(BaseModel):
    repo_id: int
    question: str

class PrefetchRequest(BaseModel):
    repo_id: int
    questions: list[str]


@web_app.post("/start_ingestion")
def start_ingestion(request: IngestRequest):
    db_service = DatabaseService()
    repo = db_service.get_repository_by_url(request.repo_url.removesuffix('.git'))
    if repo and repo['status'] == 'completed':
        return {"message": "Repository already processed.", "repo_id": repo['id']}
    call = ingest_repository.spawn(request.repo_url)
    return {"message": "Ingestion process started.", "call_id": call.object_id}


@web_app.get("/get_ingestion_status")
def get_ingestion_status(call_id: str):
    from modal.functions import FunctionCall
    function_call = FunctionCall.from_id(call_id)
    try:
        result = function_call.get(timeout=0)
        return {"status": "completed", "repo_id": result.get("repo_id"), "blueprint": result.get("blueprint")}
    except TimeoutError:
        return {"status": "running"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@web_app.post("/ask")
def ask(request: AskRequest):
    """Enhanced /ask endpoint with intelligent routing including web search"""
    db_service = DatabaseService()
    
    print(f"\n🔥 Processing question: {request.question[:100]}...")
    
    # Parallel processing for speed
    intent_call = run_dispatcher.spawn(request.question)
    embedding_call = embed_chunks_batch.spawn([request.question])
    
    # Get context while classification runs
    query_embedding = embedding_call.get()[0]
    context_chunks = db_service.vector_search_chunks(query_embedding, request.repo_id, top_k=8)
    
    # Build context
    context = "\n\n".join([f"--- File: {chunk['file_path']} ---\n{chunk['chunk_text'][:400]}" for chunk in context_chunks[:5]]) if context_chunks else ""
    
    intent_result = intent_call.get()
    print(f"🎯 Question classified as: {intent_result}")
    
    # Enhanced routing with web search
    if intent_result == "web_search":
        print("🌐 Using web search agent")
        answer_call = run_web_search_agent.spawn(request.question, context)
    elif intent_result == "security":
        if context:
            print("🛡️ Using security guardian with context")
            answer_call = run_security_guardian.spawn(context)
        else:
            print("🌐 Security question with no context - using web search")
            answer_call = run_web_search_agent.spawn(request.question, "")
    elif intent_result == "refactor":
        if context:
            print("🔧 Using refactoring agent with context")
            answer_call = run_refactoring_agent.spawn(context)
        else:
            print("🌐 Refactoring question with no context - using web search")
            answer_call = run_web_search_agent.spawn(request.question, "")
    else:
        # Default to appropriate agent
        if context:
            print("🧠 Using deep thinker with context")
            answer_call = run_deep_thinker.spawn(request.question, context)
        else:
            print("🌐 No context available - using web search")
            answer_call = run_web_search_agent.spawn(request.question, "")
    
    # Generate follow-ups
    followup_call = generate_followup_questions_fast.spawn(request.question)
    
    answer = answer_call.get()
    followup_questions = followup_call.get()
    
    print(f"✅ Response generated: {len(answer)} characters")
    
    return {
        "answer": answer, 
        "followup_questions": followup_questions,
        "intent": intent_result
    }


@web_app.get("/get_smart_questions")
def get_smart_questions(repo_id: int):
    """Get project-specific smart questions based on the repository blueprint"""
    try:
        db_service = DatabaseService()
        blueprint = None
        
        # Try to get the blueprint from database with error handling
        try:
            with db_service.get_connection() as conn:
                stmt = text("SELECT blueprint FROM repositories WHERE id = :repo_id")
                result = conn.execute(stmt, {"repo_id": repo_id}).scalar()
                blueprint = result
        except Exception as e:
            print(f"Error accessing blueprint: {e}")
            blueprint = None
        
        # Fallback to generic questions if no blueprint or error
        if not blueprint:
            return {"questions": [
                {"category": "ARCHITECTURE", "question": "What is the main purpose of this code?"},
                {"category": "IMPLEMENTATION", "question": "How are the core components implemented?"},
                {"category": "DEBUGGING", "question": "What are potential bugs or error states?"},
                {"category": "OPTIMIZATION", "question": "How can performance be optimized?"},
                {"category": "SECURITY", "question": "Are there any security vulnerabilities?"},
                {"category": "TESTING", "question": "What tests should be written?"}
            ]}
        
        # Generate smart questions based on blueprint
        try:
            questions_call = generate_smart_questions_from_blueprint.spawn(blueprint, repo_id)
            questions = questions_call.get()
            return {"questions": questions}
        except Exception as e:
            print(f"Error generating smart questions: {e}")
            # Return fallback questions
            return {"questions": [
                {"category": "ARCHITECTURE", "question": "What is the main purpose of this code?"},
                {"category": "IMPLEMENTATION", "question": "How are the core components implemented?"},
                {"category": "DEBUGGING", "question": "What are potential bugs or error states?"},
                {"category": "OPTIMIZATION", "question": "How can performance be optimized?"},
                {"category": "SECURITY", "question": "Are there any security vulnerabilities?"},
                {"category": "TESTING", "question": "What tests should be written?"}
            ]}
            
    except Exception as e:
        print(f"Critical error in get_smart_questions: {e}")
        # Return basic fallback questions
        return {"questions": [
            {"category": "GENERAL", "question": "What is the main purpose of this code?"},
            {"category": "GENERAL", "question": "How are the core components implemented?"},
            {"category": "GENERAL", "question": "What are potential bugs or error states?"},
            {"category": "GENERAL", "question": "How can performance be optimized?"},
            {"category": "GENERAL", "question": "Are there any security vulnerabilities?"},
            {"category": "GENERAL", "question": "What tests should be written?"}
        ]}


@web_app.post("/prefetch_answers")
def prefetch_answers(request: PrefetchRequest):
    """Ultra-fast parallel prefetch with comprehensive debug logging"""
    start_time = time.time()
    db_service = DatabaseService()
    
    questions = request.questions[:6] 
    print(f"\n🚀 PREFETCH DEBUG START: {len(questions)} questions")
    print(f"   Repo ID: {request.repo_id}")
    for i, q in enumerate(questions):
        print(f"   Q{i+1}: '{q}'")
    
    results = {}
    
    try:
        # Generate embeddings
        print(f"\n📊 Step 1: Generating embeddings...")
        embedding_start = time.time()
        all_embeddings_call = embed_chunks_batch.spawn(questions)
        all_embeddings = all_embeddings_call.get()
        embedding_time = time.time() - embedding_start
        print(f"   ✅ Embeddings generated in {embedding_time:.1f}s")
        
        # Process each question
        print(f"\n📊 Step 2: Processing questions with context...")
        answer_calls = []
        
        for i, question in enumerate(questions):
            try:
                print(f"\n   Processing Q{i+1}: '{question}'")
                query_embedding = all_embeddings[i]
                
                context_chunks = db_service.vector_search_chunks(
                    query_embedding, request.repo_id, top_k=3
                )
                
                if context_chunks:
                    print(f"     ✅ Found {len(context_chunks)} context chunks")
                    context = "\n".join([
                        f"{chunk['chunk_text'][:150]}" 
                        for chunk in context_chunks[:2]
                    ])
                    print(f"     📄 Context length: {len(context)} chars")
                    
                    answer_call = run_deep_thinker.spawn(question, context)
                    answer_calls.append((question, answer_call, True))
                    print(f"     🚀 LLM call queued")
                else:
                    print(f"     ❌ No context chunks found")
                    results[question] = {
                        "answer": "No relevant context found for this question in the codebase.",
                        "followup_questions": ["What is the main architecture?", "How does this work?"],
                        "intent": "summary"
                    }
                    
            except Exception as e:
                print(f"     💥 Error processing Q{i+1}: {e}")
                results[question] = {
                    "answer": f"Processing error: {str(e)[:50]}",
                    "followup_questions": [],
                    "intent": "error"
                }
        
        # Collect results
        print(f"\n📊 Step 3: Collecting {len(answer_calls)} LLM results...")
        collection_start = time.time()
        
        for i, (question, answer_call, has_context) in enumerate(answer_calls):
            try:
                print(f"\n   Collecting Q{i+1}: '{question[:50]}...'")
                result_start = time.time()
                
                answer = answer_call.get(timeout=25)
                result_time = time.time() - result_start
                
                print(f"     ✅ Got answer in {result_time:.1f}s ({len(answer)} chars)")
                
                # Generate followups based on question type
                followup_questions = []
                question_lower = question.lower()
                if "purpose" in question_lower or "main" in question_lower:
                    followup_questions = ["How is this implemented?", "What are the key components?"]
                elif "security" in question_lower:
                    followup_questions = ["How can these be fixed?", "What are best practices?"]
                elif "architecture" in question_lower:
                    followup_questions = ["What design patterns are used?", "How do components interact?"]
                elif "implement" in question_lower:
                    followup_questions = ["What are the main design patterns?", "How do components interact?", "Are there any limitations?"]
                elif "improve" in question_lower or "optimize" in question_lower:
                    followup_questions = ["What would be the impact?", "Are there trade-offs?"]
                else:
                    followup_questions = ["Tell me more", "How does this work?"]
                
                # Store result with debug info
                results[question] = {
                    "answer": answer,
                    "followup_questions": followup_questions,
                    "intent": "summary"
                }
                
                print(f"     📝 Stored answer with {len(followup_questions)} followups")
                
            except Exception as e:
                print(f"     ⏰ Timeout/Error for Q{i+1}: {e}")
                results[question] = {
                    "answer": f"Request timed out or failed: {str(e)[:50]}. Please try asking this individually.",
                    "followup_questions": ["Try this question again", "Ask about specific components"],
                    "intent": "error"
                }
        
        collection_time = time.time() - collection_start
        total_time = time.time() - start_time
        
        print(f"\n🎉 PREFETCH COMPLETE:")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Embedding time: {embedding_time:.1f}s")
        print(f"   Collection time: {collection_time:.1f}s")
        print(f"   Results: {len(results)} answers")
        
        # Debug: Show final results structure
        print(f"\n📊 FINAL RESULTS TO CACHE:")
        for i, (key, value) in enumerate(results.items()):
            answer_preview = value.get('answer', '')[:80] + '...' if len(value.get('answer', '')) > 80 else value.get('answer', 'No answer')
            followup_count = len(value.get('followup_questions', []))
            print(f"   Result {i+1}:")
            print(f"     Key: '{key}'")
            print(f"     Answer: {answer_preview}")
            print(f"     Followups: {followup_count}")
            print(f"     Intent: {value.get('intent', 'unknown')}")
        
        print(f"\n📤 RETURNING: {{'answers': {len(results)} results}}")
        return {"answers": results}
        
    except Exception as e:
        print(f"\n💥 CRITICAL PREFETCH ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        error_results = {}
        for q in questions:
            error_results[q] = {
                "answer": "Service temporarily unavailable due to error. Please try asking questions individually.",
                "followup_questions": [],
                "intent": "error"
            }
        
        print(f"📤 RETURNING ERROR RESULTS: {len(error_results)} entries")
        return {"answers": error_results}


@web_app.get("/get_available_functions") 
def get_available_functions(repo_id: int):
    """Get all available functions/methods from the codebase - optimized"""
    db_service = DatabaseService()
    
    # Search for function definitions in the code with timeout
    function_patterns = [
        "def ", "function ", "class ", "const ", "@app.function"
    ]
    
    functions = []
    try:
        with db_service.get_connection() as conn:
            for pattern in function_patterns[:3]:  # Limit patterns for speed
                stmt = text("""
                    SELECT DISTINCT c.chunk_text, sf.file_path 
                    FROM code_chunks c 
                    JOIN source_files sf ON c.file_id = sf.id 
                    WHERE sf.repo_id = :repo_id AND c.chunk_text LIKE :pattern
                    LIMIT 15
                """)
                results = conn.execute(stmt, {"repo_id": repo_id, "pattern": f"%{pattern}%"}).mappings().all()
                
                for result in results:
                    # Extract function names (simplified)
                    lines = result['chunk_text'].split('\n')
                    for line in lines[:5]:  # Only check first 5 lines
                        if pattern in line and ('(' in line or '{' in line):
                            functions.append({
                                "signature": line.strip()[:80],  # Truncate for speed
                                "file_path": result['file_path'],
                                "type": "function" if "def " in line or "function " in line else "other"
                            })
                            break
    except Exception as e:
        print(f"Error getting functions: {e}")
    
    return {"functions": functions[:30]}  # Limit to 30 functions


@web_app.get("/health")
def health_check():
    """Monitor system health and performance"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "gpu_available": True,
        "web_search_enabled": True,
        "message": "System operational with Kimi web search"
    }


# --- Deploy the FastAPI app ---
@app.function()
@modal.asgi_app()
def api():
    return web_app