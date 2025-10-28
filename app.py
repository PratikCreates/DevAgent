import os
import re
import json
import hashlib
import requests
import streamlit as st
from pathlib import Path
from difflib import get_close_matches
import time

st.set_page_config(page_title="DevAgent", layout="wide", page_icon="🕵️")

BACKEND_BASE_URL = os.environ.get("BACKEND_BASE_URL", "http://localhost:8000")
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

# Enhanced CSS with better performance indicators
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.stApp { 
    background: #0f0f0f; 
    color: #e5e5e5; 
    font-family: 'Inter', sans-serif; 
}

.main-header { 
    background: #1e1e1e; 
    border: 1px solid #333; 
    padding: 1rem; 
    border-radius: 10px; 
    margin-bottom: 1rem; 
    color: #f5f5f5; 
    text-align: center; 
    box-shadow: 0 6px 24px rgba(0,0,0,0.25); 
}

.sidebar-section { 
    background: #141414; 
    border: 1px solid #2e2e2e; 
    padding: 0.7rem; 
    border-radius: 8px; 
    margin-bottom: 0.6rem; 
    font-size: 0.85rem; 
}

.category-badge { 
    display:inline-block; 
    padding:0.2rem 0.6rem; 
    border-radius:8px; 
    font-size:0.72rem; 
    font-weight:600; 
    margin-bottom:0.35rem; 
    text-transform:uppercase; 
}

.badge-architecture{background:#2b6cb0;color:#fff}
.badge-implementation{background:#2f855a;color:#fff}
.badge-debugging{background:#b7791f;color:#fff}
.badge-optimization{background:#6b46c1;color:#fff}
.badge-security{background:#c53030;color:#fff}
.badge-testing{background:#c05621;color:#fff}
.badge-general{background:#2c7a7b;color:#fff}

.welcome-card { 
    background: #1a1a1a; 
    border: 1px solid #333; 
    padding: 1.1rem; 
    border-radius: 12px; 
    margin: 1rem 0; 
    text-align:center; 
}

.stButton > button { 
    background: #2b6cb0 !important; 
    color: white !important; 
    border-radius: 8px !important; 
    padding: 0.45rem 1rem !important; 
    transition: all 0.14s ease !important; 
}

.stTextInput input { 
    background: #151515 !important; 
    color: #eee !important; 
}

.status-indicator { 
    display:inline-flex; 
    align-items:center; 
    padding:0.2rem 0.6rem; 
    border-radius:10px; 
    font-size:0.75rem; 
    font-weight:600; 
    margin:0.2rem; 
}

.status-online{background:#2f855a;color:#fff}
.status-loading{background:#b7791f;color:#fff}
.status-error{background:#c53030;color:#fff}
.status-prefetching{background:#6b46c1;color:#fff}
.status-cached{background:#0891b2;color:#fff}
.status-websearch{background:#0d9488;color:#fff}

.docs-main-panel { 
    background:#0f1720; 
    border:1px solid #23303a; 
    padding:1.2rem; 
    border-radius:10px; 
    margin-bottom:1rem; 
    font-size:1.05rem; 
    line-height:1.5; 
    color:#e6eef6; 
    max-width: 800px;
    margin: 0 auto 2rem auto;
}

.docs-main-panel h3 { 
    margin:0 0 1rem 0; 
    color: #4fc3f7;
    font-size: 1.4rem;
}

.docs-main-panel h4 { 
    margin: 1.2rem 0 0.6rem 0; 
    color: #81c784;
    font-size: 1.1rem;
}

.docs-main-panel ul {
    margin-bottom: 1rem;
}

.docs-main-panel li {
    margin-bottom: 0.4rem;
    padding-left: 0.5rem;
}

.prefetch-status {
    background: #1a1a2e;
    border: 1px solid #16213e;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
    text-align: center;
}

.small-muted { 
    font-size:0.85rem; 
    color:#9aa6b2; 
}

.cache-indicator {
    background: #0f4c75;
    color: #00d4aa;
    padding: 0.3rem 0.6rem;
    border-radius: 6px;
    font-size: 0.7rem;
    font-weight: 600;
    display: inline-block;
    margin-left: 0.5rem;
}

.websearch-indicator {
    background: #065f46;
    color: #34d399;
    padding: 0.3rem 0.6rem;
    border-radius: 6px;
    font-size: 0.7rem;
    font-weight: 600;
    display: inline-block;
    margin-left: 0.5rem;
}

.performance-metrics {
    background: #1a1a1a;
    border: 1px solid #333;
    padding: 0.8rem;
    border-radius: 6px;
    font-size: 0.8rem;
    margin: 0.5rem 0;
}

.feature-highlight {
    background: linear-gradient(135deg, #0d9488 0%, #059669 100%);
    padding: 0.8rem;
    border-radius: 8px;
    margin: 1rem 0;
    text-align: center;
    color: white;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# Improved cache management with better key generation
def normalize_question(question):
    """Normalize question for consistent cache key generation"""
    # Remove emojis, special characters, extra spaces
    cleaned = re.sub(r'[^\w\s]', '', question.lower())
    # Remove extra whitespace and normalize
    cleaned = ' '.join(cleaned.split())
    return cleaned.strip()

def get_question_hash(question):
    """Generate consistent hash for question caching"""
    normalized = normalize_question(question)
    return hashlib.md5(normalized.encode()).hexdigest()[:12]

def get_cache_path(repo_id, cache_type):
    """Get cache file path for a repo and cache type"""
    return CACHE_DIR / f"{repo_id}_{cache_type}.json"

def save_to_cache(repo_id, cache_type, data):
    """Save data to disk cache using JSON"""
    try:
        cache_path = get_cache_path(repo_id, cache_type)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Failed to save cache: {e}")
        return False

def load_from_cache(repo_id, cache_type):
    """Load data from disk cache"""
    try:
        cache_path = get_cache_path(repo_id, cache_type)
        if cache_path.exists():
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"Failed to load cache: {e}")
    return None

# API Functions with timeout handling
def start_ingestion(repo_url):
    resp = requests.post(f"{BACKEND_BASE_URL}/start_ingestion", json={"repo_url": repo_url}, timeout=1800) # Increased timeout for sync ingestion
    resp.raise_for_status()
    return resp.json()

def get_smart_questions(repo_id):
    resp = requests.get(f"{BACKEND_BASE_URL}/get_smart_questions?repo_id={repo_id}", timeout=30)
    resp.raise_for_status()
    return resp.json()

def get_available_functions(repo_id):
    resp = requests.get(f"{BACKEND_BASE_URL}/get_available_functions?repo_id={repo_id}", timeout=30)
    resp.raise_for_status()
    return resp.json()

def prefetch_answers(repo_id, questions):
    resp = requests.post(f"{BACKEND_BASE_URL}/prefetch_answers", json={"repo_id": repo_id, "questions": questions}, timeout=300)
    resp.raise_for_status()
    return resp.json()

def ask_question(repo_id, question):
    try:
        resp = requests.post(f"{BACKEND_BASE_URL}/ask", json={"repo_id": repo_id, "question": question}, timeout=120)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        return {"answer": "⏰ Request timed out.", "followup_questions": [], "intent": "error"}
    except requests.exceptions.RequestException as e:
        return {"answer": f"🔌 Connection error: {e}", "followup_questions": [], "intent": "error"}
    except Exception as e:
        return {"answer": f"❌ Unexpected error: {e}", "followup_questions": [], "intent": "error"}

def is_web_search_question(question):
    """Detect if a question likely needs web search"""
    web_search_keywords = [
        "deploy", "hosting", "aws", "azure", "gcp", "docker", "kubernetes", "ci/cd",
        "latest", "current", "recent", "new version", "updated",
        "best practices", "recommended", "how to", "tutorial",
        "similar projects", "alternatives", "community",
        "documentation", "guide", "examples", "npm install", "pip install"
    ]
    return any(keyword in question.lower() for keyword in web_search_keywords)

# Enhanced prefetch pipeline with better error handling
def run_prefetch_pipeline(repo_id):
    """Run prefetch pipeline with improved caching and error handling"""
    try:
        # Check if we have valid cached data first
        cached_answers = load_from_cache(repo_id, "prefetched_answers")
        cached_functions = load_from_cache(repo_id, "functions") 
        cached_questions = load_from_cache(repo_id, "smart_questions")
        
        if cached_answers and cached_functions and cached_questions:
            # Load from cache
            st.session_state.prefetched_answers = cached_answers
            st.session_state.available_functions = cached_functions
            st.session_state.smart_questions = cached_questions
            st.session_state.prefetch_status = "loaded_from_cache"
            st.success("⚡ Loaded from cache - ready instantly!")
            
            # Show cache metrics
            st.markdown(f'<div class="performance-metrics">📊 <strong>Cache Stats:</strong> {len(cached_answers)} prefetched answers, {len(cached_functions)} functions, {len(cached_questions)} smart questions</div>', unsafe_allow_html=True)
            
            # DEBUG: Show what's in cache
            print(f"🔍 Loaded cache contents:")
            for i, (key, value) in enumerate(cached_answers.items()):
                if i < 5:  # Show first 5
                    answer_preview = value.get('answer', '')[:30] + '...' if value.get('answer') else 'No answer'
                    print(f"   Cache[{i}]: '{key}' -> {answer_preview}")
            
            return True
        
        st.session_state.prefetch_status = "running"
        
        # Create progress container
        progress_container = st.empty()
        with progress_container.container():
            st.markdown('<div class="prefetch-status">', unsafe_allow_html=True)
            st.markdown("🚀 **Starting AI Analysis Pipeline**")
            progress_bar = st.progress(0)
            status_text = st.empty()
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Define standard questions for prefetching
        standard_questions = [
            "What is the main purpose of this code?",
            "What is the architecture?",
            "How does authentication work?", 
            "What are the main API endpoints?",
            "Are there any obvious security issues?",
            "How can I improve performance?",
            "What are the main dependencies?",
            "How does the database integration work?",
            "What is the project structure?",
            "How do I set up and run this project?"
        ]
        
        print(f"🚀 Starting prefetch pipeline with {len(standard_questions)} standard questions")
        
        # Step 1: Get smart questions (with caching)
        status_text.text("📝 Getting smart questions...")
        progress_bar.progress(15)
        
        try:
            smart_q_response = get_smart_questions(repo_id)
            smart_questions = smart_q_response.get("questions", [])
            st.session_state.smart_questions = smart_questions
            save_to_cache(repo_id, "smart_questions", smart_questions)
            status_text.text(f"✅ Found {len(smart_questions)} smart questions")
        except Exception as e:
            st.warning(f"Failed to get smart questions: {e}")
            smart_questions = []
            
        progress_bar.progress(30)
        
        # Step 2: Get available functions (with caching)
        status_text.text("🔧 Analyzing code functions...")
        
        try:
            funcs_response = get_available_functions(repo_id)
            functions = funcs_response.get("functions", [])
            st.session_state.available_functions = functions
            save_to_cache(repo_id, "functions", functions)
            status_text.text(f"✅ Found {len(functions)} functions")
        except Exception as e:
            st.warning(f"Failed to get functions: {e}")
            functions = []
            
        progress_bar.progress(50)
        
        # Step 3: Prefetch answers for standard questions
        status_text.text("🧠 Prefetching answers for common questions...")
        progress_bar.progress(60)
        
        prefetched_answers = {}
        
        try:
            # Make the prefetch API call
            start_time = time.time()
            print(f"🌐 Calling prefetch API with {len(standard_questions)} questions")
            prefetch_response = prefetch_answers(repo_id, standard_questions)
            end_time = time.time()
            
            print(f"🔍 Prefetch API response structure: {list(prefetch_response.keys())}")
            answers_data = prefetch_response.get("answers", {})
            
            if answers_data:
                print(f"✅ Got {len(answers_data)} answers from API")
                
                # Store answers with proper normalization AND original keys
                for question, answer_data in answers_data.items():
                    # Store with original question as key
                    prefetched_answers[question] = answer_data
                    print(f"   Stored: '{question}' -> {len(answer_data.get('answer', ''))} chars")
                    
                    # Also store with normalized key for better matching
                    normalized_key = normalize_question(question)
                    if normalized_key != question:
                        prefetched_answers[normalized_key] = answer_data
                        print(f"   Also stored normalized: '{normalized_key}'")
                
                status_text.text(f"✅ Prefetched {len(answers_data)} answers in {end_time-start_time:.1f}s")
                
                # DEBUG: Show what's in cache
                print(f"🔍 Final cache contents:")
                for i, (key, value) in enumerate(prefetched_answers.items()):
                    if i < 5:  # Show first 5
                        answer_preview = value.get('answer', '')[:30] + '...' if value.get('answer') else 'No answer'
                        print(f"   Cache[{i}]: '{key}' -> {answer_preview}")
                
            else:
                status_text.text("⚠️ No answers returned from prefetch API")
                print("❌ Empty answers_data from API")
                
        except Exception as e:
            st.error(f"❌ Failed to prefetch answers: {e}")
            print(f"💥 Prefetch API error: {e}")
            
        progress_bar.progress(80)
        
        # Step 4: Save everything to cache
        status_text.text("💾 Saving to cache...")
        
        st.session_state.prefetched_answers = prefetched_answers
        print(f"🔄 Session state updated with {len(prefetched_answers)} cached answers")
        
        # Save all cache data
        save_to_cache(repo_id, "prefetched_answers", prefetched_answers)
        
        progress_bar.progress(100)
        status_text.text("✅ Pipeline completed successfully!")
        
        # Show final metrics
        time.sleep(1)
        progress_container.empty()
        
        st.session_state.prefetch_status = "completed"
        
        # Display success metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("💬 Prefetched Answers", len(prefetched_answers))
        with col2:
            st.metric("🔧 Functions Found", len(functions))
        with col3:
            st.metric("💡 Smart Questions", len(smart_questions))
        
        # New feature highlight
        st.markdown('<div class="feature-highlight">🌐 <strong>NEW:</strong> Web search enabled! Ask about deployment, hosting, latest practices, and more.</div>', unsafe_allow_html=True)
        
        st.success("🎉 Repository analysis completed! Chat is now super fast with prefetched answers and web search.")
        return True
        
    except Exception as e:
        st.session_state.prefetch_status = "error"
        st.error(f"❌ Prefetch pipeline failed: {e}")
        print(f"💥 Pipeline error: {e}")
        return False

def find_cached_answer(question, prefetched_answers):
    """Enhanced answer lookup with debug logging"""
    if not prefetched_answers:
        print(f"❌ No prefetched answers available")
        return None, "no_match"
        
    question_clean = question.strip()
    question_normalized = normalize_question(question)
    
    # DEBUG: Print what we're looking for
    print(f"🔍 CACHE LOOKUP DEBUG:")
    print(f"   Question: '{question_clean}'")
    print(f"   Normalized: '{question_normalized}'")
    print(f"   Cache has {len(prefetched_answers)} entries")
    print(f"   First 5 cache keys: {list(prefetched_answers.keys())[:5]}")
    
    # Strategy 1: Exact match (case insensitive)  
    for key, value in prefetched_answers.items():
        if key.lower() == question_clean.lower():
            print(f"✅ EXACT MATCH found: '{key}'")
            return value, "exact_match"
    
    # Strategy 2: Normalized match  
    if question_normalized in prefetched_answers:
        print(f"✅ NORMALIZED MATCH found: '{question_normalized}'")
        return prefetched_answers[question_normalized], "normalized_match"
    
    # Strategy 3: Fuzzy matching with debug
    try:
        all_keys = list(prefetched_answers.keys())
        normalized_keys = [normalize_question(k) for k in all_keys]
        
        print(f"   Checking fuzzy matches against {len(normalized_keys)} normalized keys...")
        print(f"   Sample normalized keys: {normalized_keys[:3]}")
        
        close_matches = get_close_matches(question_normalized, normalized_keys, n=3, cutoff=0.6)  # Lowered cutoff, more matches
        
        if close_matches:
            print(f"✅ FUZZY MATCHES found: {close_matches}")
            
            # Find the original key for the best match
            best_match_normalized = close_matches[0]
            for i, norm_key in enumerate(normalized_keys):
                if norm_key == best_match_normalized:
                    original_key = all_keys[i]
                    value = prefetched_answers[original_key]
                    print(f"   Using original key: '{original_key}'")
                    return value, "fuzzy_match"
        else:
            print(f"❌ No fuzzy matches found (tried cutoff 0.6)")
            
    except Exception as e:
        print(f"❌ Fuzzy match error: {e}")
    
    # Strategy 4: Partial word matching with debug
    try:
        question_words = set(question_normalized.split())
        print(f"   Question words: {question_words}")
        
        best_match = None
        best_score = 0
        best_key = None
        
        for key, value in prefetched_answers.items():
            key_normalized = normalize_question(key)
            key_words = set(key_normalized.split())
            
            if len(key_words) > 1:  # Only for meaningful keys
                intersection = question_words & key_words
                union = question_words | key_words
                score = len(intersection) / len(union) if union else 0
                
                if score > 0.4:  # Lower threshold
                    print(f"   Partial match candidate: '{key}' (score: {score:.2f})")
                    print(f"     Question words: {question_words}")
                    print(f"     Key words: {key_words}")
                    print(f"     Intersection: {intersection}")
                    
                    if score > best_score:
                        best_score = score
                        best_match = value
                        best_key = key
        
        if best_match and best_score > 0.5:  # Require decent match
            print(f"✅ PARTIAL MATCH found: '{best_key}' (score: {best_score:.2f})")
            return best_match, "partial_match"
        else:
            print(f"❌ No good partial matches (best score: {best_score:.2f})")
            
    except Exception as e:
        print(f"❌ Partial match error: {e}")
    
    # Strategy 5: Last resort - contains matching
    try:
        question_lower = question_clean.lower()
        for key, value in prefetched_answers.items():
            key_lower = key.lower()
            
            # Check if key words are in question or vice versa
            key_words = key_lower.split()
            question_words = question_lower.split()
            
            # If most key words are in the question
            matches = sum(1 for word in key_words if word in question_lower)
            if len(key_words) > 0 and matches / len(key_words) > 0.7:
                print(f"✅ CONTAINS MATCH found: '{key}' ({matches}/{len(key_words)} words match)")
                return value, "contains_match"
                
    except Exception as e:
        print(f"❌ Contains match error: {e}")
    
    print(f"❌ NO MATCH FOUND for: '{question_clean}'")
    print(f"   Available cache keys:")
    for i, key in enumerate(list(prefetched_answers.keys())[:10]):
        print(f"     {i+1}: '{key}'")
    
    return None, "no_match"

def handle_ingestion():
    """Handle repository ingestion process"""
    if "ingestion_status" not in st.session_state or st.session_state.ingestion_status != "starting":
        return
        
    try:
        data = start_ingestion(st.session_state.repo_url)
        if data.get("repo_id"):
            st.session_state.repo_id = data.get("repo_id")
            st.session_state.blueprint = data.get("blueprint", "")
            st.session_state.ingestion_status = "completed"
        else:
            st.session_state.ingestion_status = "error"
            st.session_state.ingestion_error = data.get("blueprint", "Unknown ingestion error")
    except Exception as e:
        st.session_state.ingestion_status = "error"
        st.session_state.ingestion_error = str(e)

    st.rerun()

def render_sidebar():
    """Render the sidebar with controls"""
    with st.sidebar:
        st.markdown('<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">', unsafe_allow_html=True)
        st.markdown('<div><h3 style="margin:0">🕵️ DevAgent</h3><div class="small-muted">Decompose public GitHub repos</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Repository Analysis Section
        with st.expander("🔎 Analyze Repository", expanded=True):
            repo_url = st.text_input("GitHub repo URL", placeholder="https://github.com/owner/repo", key="repo_input")
            if st.button("Analyze Repository", use_container_width=True):
                if repo_url and repo_url.startswith("https://github.com/"):
                    st.session_state.repo_url = repo_url
                    st.session_state.ingestion_status = "starting"
                    st.session_state.screen = "chat"
                    # Reset all cache states
                    st.session_state.prefetch_status = "pending"
                    st.session_state.prefetched_answers = {}
                    st.session_state.smart_questions = []
                    st.session_state.available_functions = []
                    st.rerun()
                else:
                    st.error("Please enter a valid GitHub repository URL (must start with https://github.com/).")
        
        # System Status Section
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("🖥️ System Status")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🤖 AI Status", "Online", delta="Ready")
        with col2:
            message_count = len([m for m in st.session_state.get("messages", []) if m.get("role") == "user"])
            st.metric("💬 Questions", message_count)
        
        # New feature callout
        st.markdown('<div class="status-indicator status-websearch">🌐 Web Search Enabled</div>', unsafe_allow_html=True)
        
        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("📋 Export Chat", use_container_width=True):
                repo_name = os.path.basename(st.session_state.get("repo_url", "") or "devagent").replace(".git", "")
                chat_export = "\n\n".join([f"{'You' if m['role']=='user' else 'AI'}: {m['content']}" for m in st.session_state.get("messages", [])])
                st.download_button("💾 Download", chat_export, file_name=f"devagent_chat_{repo_name}.txt", mime="text/plain", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Only show Smart Questions and Functions after prefetch is completed
        if st.session_state.get("prefetch_status") in ["completed", "loaded_from_cache"]:
            # Smart Questions Section
            smart_questions = st.session_state.get("smart_questions", [])[:4]
            if smart_questions:
                st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
                st.markdown("<strong>💡 Smart Questions</strong>", unsafe_allow_html=True)
                
                for i, sq in enumerate(smart_questions):
                    q = sq.get("question", "")
                    cat = (sq.get("category", "GENERAL") or "GENERAL").upper()
                    badge_class = f"badge-{cat.lower()}" if cat.lower() in ["architecture", "implementation", "debugging", "optimization", "security", "testing"] else "badge-general"
                    
                    st.markdown(f'<span class="category-badge {badge_class}">{cat}</span>', unsafe_allow_html=True)
                    st.markdown(f"**{q}**")
                    
                    if st.button("Ask", key=f"ask_sq_{i}", use_container_width=True):
                        st.session_state._last_question = q
                        st.rerun()
                    st.markdown("---")
                    
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Available Functions Section  
            available_functions = st.session_state.get("available_functions", [])[:6]
            if available_functions:
                st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
                st.markdown("<strong>⚙️ Code Functions</strong>", unsafe_allow_html=True)
                
                options = [f.get("signature", "")[:65] + ("..." if len(f.get("signature", "")) > 65 else "") for f in available_functions]
                sel = st.selectbox("Choose function to explain", ["Select a function..."] + options, key="fn_select")
                
                if sel and sel != "Select a function..." and st.button("Explain Function", use_container_width=True):
                    st.session_state._last_question = f"Explain how this function works: {sel}"
                    st.rerun()
                    
                st.markdown('</div>', unsafe_allow_html=True)
        
        elif st.session_state.get("prefetch_status") == "running":
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown('<div class="status-indicator status-prefetching">⚡ Analyzing Repository</div>', unsafe_allow_html=True)
            st.markdown("Smart questions and functions will appear here once analysis is complete.", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

def render_docs_main_screen():
    """Render documentation on the main screen"""
    st.markdown('<div class="docs-main-panel">', unsafe_allow_html=True)
    st.markdown("<h3>📖 DevAgent Documentation</h3>", unsafe_allow_html=True)
    
    st.markdown("""
<h4>🚀 Quick Start Guide</h4>
<ul>
<li>🔗 <strong>Enter Repository:</strong> Paste any public GitHub repo URL in the sidebar</li>
<li>⚡ <strong>AI Analysis:</strong> DevAgent analyzes the entire codebase automatically</li>
<li>💬 <strong>Interactive Chat:</strong> Ask questions about the code like you're talking to the developer</li>
<li>🔍 <strong>Smart Insights:</strong> Get architectural insights, find bugs, understand patterns</li>
</ul>

<h4>✨ Key Features</h4>
<ul>
<li>🧠 <strong>Smart Questions:</strong> AI-generated project-specific questions for deeper insights</li>
<li>🛡️ <strong>Security Analysis:</strong> Automatic vulnerability detection and recommendations</li>
<li>🛠️ <strong>Code Functions:</strong> Explore and understand individual functions and methods</li>
<li>🏗️ <strong>Architecture Insights:</strong> High-level design patterns and structure analysis</li>
<li>🎯 <strong>Context-Aware:</strong> Understands your conversation history for better answers</li>
<li>⚡ <strong>Ultra-Fast Responses:</strong> Instant answers for common questions via intelligent caching</li>
<li>🌐 <strong>Web Search Integration:</strong> Real-time access to current documentation and best practices</li>
</ul>

<h4>🎯 Best Questions to Ask</h4>
<ul>
<li><strong>"What is the main architecture of this project?"</strong> - Get high-level system design</li>
<li><strong>"How do I deploy this to AWS/Azure/GCP?"</strong> - Get current deployment guides (🌐 Web Search)</li>
<li><strong>"What are the latest security best practices for this stack?"</strong> - Current security recommendations (🌐 Web Search)</li>
<li><strong>"How do I set up CI/CD for this project?"</strong> - Current CI/CD tutorials (🌐 Web Search)</li>
<li><strong>"What are some alternatives to this framework?"</strong> - Community comparisons (🌐 Web Search)</li>
<li><strong>"How does the authentication work?"</strong> - Code-specific analysis</li>
<li><strong>"Are there any security vulnerabilities?"</strong> - Security audit and recommendations</li>
<li><strong>"Explain how [specific function] works"</strong> - Deep-dive into code logic</li>
</ul>

<h4>🌐 NEW: Web Search Capabilities</h4>
<ul>
<li>🚀 <strong>Deployment Questions:</strong> "How to deploy this on AWS?" gets latest AWS documentation</li>
<li>📚 <strong>Current Best Practices:</strong> "What are the latest security practices?" gets recent guides</li>
<li>🔄 <strong>Framework Updates:</strong> "How to migrate to the latest version?" gets migration docs</li>
<li>🛠️ <strong>Tooling Setup:</strong> "How to set up Docker for this?" gets current Docker practices</li>
<li>🏆 <strong>Alternatives:</strong> "What are better alternatives?" gets community comparisons</li>
<li>📖 <strong>Documentation:</strong> Automatically finds and references external docs</li>
</ul>

<h4>💡 Pro Tips</h4>
<ul>
<li>🔄 <strong>Follow-up:</strong> Use the suggested follow-up questions for deeper exploration</li>
<li>📁 <strong>Specific Files:</strong> Ask about specific files or directories by name</li>
<li>🐛 <strong>Bug Hunting:</strong> Ask "What potential bugs do you see in [component]?"</li>
<li>📈 <strong>Improvements:</strong> Request refactoring suggestions for better code quality</li>
<li>🔍 <strong>Code Review:</strong> Use it like a senior developer doing code review</li>
<li>⚡ <strong>Speed:</strong> Common questions get instant responses thanks to intelligent prefetching</li>
<li>🌐 <strong>Current Info:</strong> Questions about deployment, hosting, and best practices get real-time web information</li>
</ul>

<p class="small-muted">
<strong>Performance Note:</strong> DevAgent combines lightning-fast cached responses for code analysis with real-time web search for deployment, hosting, and current best practices. Get the best of both worlds!
</p>
""", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def strip_leading_emoji(s):
    """Remove leading emojis and symbols from text"""
    return re.sub(r'^[^\w]+', '', s).strip()

def process_question(question_text):
    """Process user question with enhanced caching and debug logic"""
    start_time = time.time()
    
    print(f"\n🔥 PROCESSING QUESTION: '{question_text}'")
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question_text})
    
    # Enhanced cache lookup - with comprehensive debug logging
    prefetched_answers = st.session_state.get("prefetched_answers", {})
    
    print(f"🔍 CACHE STATE DEBUG:")
    print(f"   Session has {len(prefetched_answers)} cached answers")
    print(f"   Cache keys sample: {list(prefetched_answers.keys())[:3]}")
    
    # Check if this is likely a web search question
    needs_web_search = is_web_search_question(question_text)
    print(f"🌐 Web search needed: {needs_web_search}")
    
    try:
        cached_result, match_type = find_cached_answer(question_text, prefetched_answers)
        print(f"🔍 Cache lookup result: {match_type}")
    except Exception as e:
        print(f"💥 Cache lookup error: {e}")
        st.error(f"Cache lookup error: {e}")
        cached_result, match_type = None, "error"
    
    if cached_result and not needs_web_search:
        # Use cached answer - should be instant!
        response_time = time.time() - start_time
        print(f"✅ CACHE HIT! Response time: {response_time*1000:.0f}ms")
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": cached_result.get("answer", "No answer available"),
            "followup_questions": cached_result.get("followup_questions", []),
            "intent": cached_result.get("intent", "summary"),
            "cached": True,
            "match_type": match_type,
            "response_time": response_time
        })
        return
    
    print(f"❌ CACHE MISS - Making API call{'(web search likely)' if needs_web_search else ''}")
    
    # Check if repo is ready for API call
    if "repo_id" not in st.session_state:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "❌ No repository analyzed yet. Please analyze a public GitHub repository first using the sidebar."
        })
        return
    
    # Build context from recent messages (limit context to avoid token limits)
    recent_context = []
    for msg in st.session_state.messages[-4:]:  # Last 2 exchanges only
        if msg["role"] == "user":
            recent_context.append(f"Q: {msg['content']}")
        else:
            recent_context.append(f"A: {msg['content'][:100]}...")
    
    context = f"{' | '.join(recent_context[-2:])}\n\nCurrent: {question_text}" if recent_context else question_text
    
    # Show loading state for API calls
    with st.chat_message("assistant"):
        loading_placeholder = st.empty()
        progress_placeholder = st.empty()
        
        with loading_placeholder.container():
            if needs_web_search:
                st.info("🌐 Searching the web for current information...")
            else:
                st.warning("🌐 Making API call - not cached yet...")
        with progress_placeholder.container():
            progress_bar = st.progress(0)
        
        try:
            progress_bar.progress(25)
            api_start = time.time()
            
            print(f"🌐 Making API call to /ask endpoint...")
            
            # Make API call
            response = ask_question(st.session_state.repo_id, context)
            
            api_time = time.time() - api_start
            progress_bar.progress(75)
            
            print(f"✅ API response received in {api_time:.1f}s")
            
            content = response.get("answer", "Sorry, I couldn't generate an answer.")
            intent = response.get("intent", "summary")
            
            # If context didn't help, try simpler question
            if "don't have enough information" in content.lower() and len(context) > len(question_text):
                with loading_placeholder.container():
                    st.info("🔄 Retrying with simpler context...")
                
                print(f"🔄 Retrying with simpler context...")
                response = ask_question(st.session_state.repo_id, question_text)
                content = response.get("answer", content)
                intent = response.get("intent", intent)
                api_time = time.time() - api_start
                print(f"✅ Retry response received in {api_time:.1f}s")
            
            progress_bar.progress(100)
            total_time = time.time() - start_time
            
            # Clear loading states
            loading_placeholder.empty()
            progress_placeholder.empty()
            
            # Show appropriate response indicator
            if intent == "web_search":
                st.markdown(f'<span class="websearch-indicator">🌐 WEB SEARCH - {api_time:.1f}s</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="cache-indicator">🌐 API CALL - {api_time:.1f}s</span>', unsafe_allow_html=True)
            
            print(f"💾 Caching response for future use...")
            
            # Add response to chat
            st.session_state.messages.append({
                "role": "assistant",
                "content": content,
                "followup_questions": response.get("followup_questions", []),
                "intent": intent,
                "cached": False,
                "response_time": total_time,
                "api_time": api_time,
                "web_search": intent == "web_search"
            })
            
            # Cache this answer for future use with debug logging (only if not web search)
            if intent != "web_search":
                try:
                    if "prefetched_answers" not in st.session_state:
                        st.session_state.prefetched_answers = {}
                    
                    # Store with multiple keys for better matching
                    answer_data = {
                        "answer": content,
                        "followup_questions": response.get("followup_questions", []),
                        "intent": intent
                    }
                    
                    # Original question
                    st.session_state.prefetched_answers[question_text] = answer_data
                    print(f"   📝 Cached with original key: '{question_text}'")
                    
                    # Normalized version  
                    normalized_question = normalize_question(question_text)
                    if normalized_question != question_text:
                        st.session_state.prefetched_answers[normalized_question] = answer_data
                        print(f"   📝 Also cached with normalized key: '{normalized_question}'")
                    
                    print(f"   💾 Total cached answers now: {len(st.session_state.prefetched_answers)}")
                    
                    # Save to disk cache
                    repo_id = st.session_state.get("repo_id")
                    if repo_id:
                        save_to_cache(repo_id, "prefetched_answers", st.session_state.prefetched_answers)
                        print(f"   💽 Saved to disk cache")
                        
                except Exception as e:
                    print(f"💥 Failed to cache answer: {e}")
                    st.warning(f"Failed to cache answer: {e}")
            else:
                print(f"🌐 Not caching web search result (time-sensitive)")
                
        except Exception as e:
            loading_placeholder.empty()
            progress_placeholder.empty()
            print(f"💥 API call failed: {e}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ Sorry, I encountered an error: {str(e)}"
            })

def WelcomeScreen():
    """Render the welcome screen"""
    st.markdown('<div class="main-header"><h1 style="margin:0">🕵️ DevAgent</h1><div style="margin-top:6px;color:#bfc7cf">Decompose any public GitHub repository with AI</div></div>', unsafe_allow_html=True)
    
    render_sidebar()
    render_docs_main_screen()
    
    # Quick start section
    st.markdown("---")
    st.markdown("### 🚀 Ready to Start?")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 Analyze Repository", key="go_analyzer", use_container_width=True, type="primary"):
            st.session_state.screen = "chat"
            st.rerun()

def ChatScreen():
    """Render the main chat interface"""
    # Initialize session state
    for key, default in [
        ("messages", []),
        ("available_functions", []),
        ("smart_questions", []),
        ("prefetched_answers", {}),
        ("prefetch_status", "pending")
    ]:
        if key not in st.session_state:
            st.session_state[key] = default
    
    render_sidebar()
    handle_ingestion()
    
    # Handle prefetch pipeline after ingestion completes
    if (st.session_state.get("ingestion_status") == "completed" and 
        st.session_state.get("prefetch_status") == "pending" and
        st.session_state.get("repo_id")):
        
        st.warning("⚡ Repository ingested! Run the AI Analysis Pipeline to enable ultra-fast responses.")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Start AI Analysis Pipeline", type="primary", use_container_width=True, key="start_pipeline"):
                run_prefetch_pipeline(st.session_state.repo_id)
                st.rerun()
    
    # Header
    repo_name = os.path.basename(st.session_state.get("repo_url", "") or "repository").replace(".git", "")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f'<div class="main-header"><h2 style="margin:0">🕵️ {repo_name}</h2></div>', unsafe_allow_html=True)
    
    with col2:
        status = st.session_state.get("ingestion_status", "starting")
        prefetch_status = st.session_state.get("prefetch_status", "pending")
        
        if status == "starting":
            st.markdown('<div class="status-indicator status-loading">⚡ Processing Repository</div>', unsafe_allow_html=True)
        elif status == "completed" and prefetch_status == "pending":
            st.markdown('<div class="status-indicator status-loading">⏳ Ready for Analysis</div>', unsafe_allow_html=True)
        elif prefetch_status == "running":
            st.markdown('<div class="status-indicator status-prefetching">🧠 AI Analyzing</div>', unsafe_allow_html=True)
        elif prefetch_status in ["completed", "loaded_from_cache"]:
            st.markdown('<div class="status-indicator status-cached">⚡ Ultra-Fast Mode</div>', unsafe_allow_html=True)
        elif status == "error" or prefetch_status == "error":
            st.markdown('<div class="status-indicator status-error">❌ Error</div>', unsafe_allow_html=True)
            if st.session_state.get("ingestion_error"):
                st.error(st.session_state.ingestion_error)
    
    st.markdown("---")
    
    # Chat interface
    if not st.session_state.get("messages"):
        # Welcome message with sample questions
        st.markdown('<div class="welcome-card"><h3 style="margin:0">🎉 Welcome to DevAgent!</h3><p style="margin-top:6px">Start by asking a question about the codebase, or try one of these suggestions:</p></div>', unsafe_allow_html=True)
        
        # Enhanced sample questions including web search examples
        sample_questions = [
            "🎯 What is the main purpose of this code?",
            "🏗️ What is the architecture?", 
            "🚀 How do I deploy this to AWS?",  # Web search example
            "🛡️ What are the latest security best practices?"  # Web search example
        ]
        
        cols = st.columns(2)
        for i, sample in enumerate(sample_questions):
            with cols[i % 2]:
                if st.button(sample, key=f"sample_{i}", use_container_width=True):
                    process_question(strip_leading_emoji(sample))
                    st.rerun()
    
    # Display chat history
    for i, msg in enumerate(st.session_state.get("messages", [])):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Show performance info for assistant messages
            if msg["role"] == "assistant" and msg.get("response_time"):
                if msg.get("cached"):
                    st.caption(f"⚡ Cached response ({msg.get('match_type', 'unknown')}) - {msg['response_time']*1000:.0f}ms")
                elif msg.get("web_search"):
                    st.caption(f"🌐 Web search response - {msg.get('api_time', 0):.1f}s")
                else:
                    st.caption(f"🌐 API response - {msg.get('api_time', 0):.1f}s")
            
            # Show follow-up questions
            if msg.get("followup_questions") and msg["role"] == "assistant":
                st.markdown("**💭 Follow-up questions:**")
                followup_cols = st.columns(min(len(msg["followup_questions"]), 2))
                
                for j, followup in enumerate(msg.get("followup_questions", [])):
                    with followup_cols[j % 2]:
                        if st.button(f"💭 {followup}", key=f"followup_{i}_{j}"):
                            process_question(followup)
                            st.rerun()
    
    # Chat input
    if hasattr(st, "chat_input"):
        user_input = st.chat_input("💬 Ask me anything about this codebase...")
        if user_input:
            process_question(user_input)
            st.rerun()
    else:
        # Fallback form for older Streamlit versions
        with st.form("question_form", clear_on_submit=True):
            question = st.text_input("Your Question", placeholder="Ask me anything about this codebase...")
            submitted = st.form_submit_button("Send 🚀", type="primary", use_container_width=True)
            
            if submitted and question.strip():
                process_question(question.strip())
                st.rerun()

def main():
    """Main application entry point"""
    # Initialize session state
    if "screen" not in st.session_state:
        st.session_state.screen = "welcome"
    
    # Route to appropriate screen
    if st.session_state.screen == "welcome":
        WelcomeScreen()
    else:
        ChatScreen()
    
    # Handle deferred question processing
    if st.session_state.get("_last_question"):
        question = st.session_state.pop("_last_question")
        process_question(question)
        st.rerun()

if __name__ == "__main__":
    main()