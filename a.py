import os
import streamlit as st
import requests
import json
from datetime import datetime
import time
import textwrap
import re
from typing import Dict, List, Tuple, Optional
import threading

# =====================================================================
# 🎯 ENHANCED STREAMLIT CONFIGURATION
# =====================================================================

st.set_page_config(
    page_title="AI Multi-Platform Content Generator Pro",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "# Multi-Platform Content Generator\nCreate viral content for LinkedIn, Facebook & WhatsApp with AI!"
    }
)

# =====================================================================
# 🎨 CUSTOM CSS FOR BETTER UI/UX
# =====================================================================

st.markdown("""
<style>
    /* Main container improvements */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        .stButton button {
            width: 100% !important;
        }
    }
    
    /* Card-like containers */
    .card {
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        background-color: #fafafa;
        margin-bottom: 1rem;
    }
    
    /* Progress indicators */
    .stStatus {
        border-left: 4px solid #0d6efd;
        padding-left: 1rem;
    }
    
    /* Better spacing for mobile */
    .css-1d391kg {
        padding: 1rem 0.5rem;
    }
    
    /* Custom header styles */
    .custom-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    
    /* Character counter animation */
    .char-counter {
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    
    .char-perfect {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .char-warning {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    
    .char-danger {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    
    /* URL list styling */
    .url-list {
        background: #f8f9fa;
        border-left: 4px solid #0077b5;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 5px 5px 0;
    }
    
    .url-item {
        padding: 0.3rem 0;
        font-family: monospace;
        font-size: 0.9em;
    }
    
    /* Ready status styling */
    .ready-status {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    /* Verification status styling */
    .verification-approved {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        margin: 0.5rem 0;
    }
    
    .verification-pending {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.5rem;
        border-radius: 5px;
        border: 1px solid #ffeaa7;
        margin: 0.5rem 0;
    }
    
    .verification-failed {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.5rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
        margin: 0.5rem 0;
    }
    
    /* Iteration counter */
    .iteration-counter {
        background-color: #e9ecef;
        color: #495057;
        padding: 0.5rem;
        border-radius: 5px;
        border: 1px solid #dee2e6;
        text-align: center;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    /* Rate limit status */
    .rate-limit-status {
        background-color: #e3f2fd;
        color: #1565c0;
        padding: 0.5rem;
        border-radius: 5px;
        border: 1px solid #bbdefb;
        font-size: 0.9em;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# 🚀 ENHANCED HEADER
# =====================================================================

st.markdown("""
<div class="custom-header">
    <h1 style="margin:0; color:white;">💼 AI Multi-Platform Content Generator Pro</h1>
    <p style="margin:0; opacity:0.9; font-size:1.1em;">Create viral, SEO-optimized content for LinkedIn, Facebook & WhatsApp that drives engagement</p>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# 🔑 ENHANCED SIDEBAR CONFIGURATION
# =====================================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key inputs with better UX
    with st.container():
        st.subheader("🔐 API Keys")
        groq_api_key = st.text_input(
            "Groq API Key", 
            type="password",
            placeholder="gsk_... or leave blank for env var",
            help="Get your free API key from https://console.groq.com",
            value=os.getenv("GROQ_API_KEY", "")  # Pre-fill from env if available
        )
        
        serper_api_key = st.text_input(
            "Serper API Key", 
            type="password",
            placeholder="Enter your Serper API key",
            help="Get free key from https://serper.dev - 2,500 search free",
            value=os.getenv("SERPER_API_KEY", "")  # Pre-fill from env if available
        )
    
    st.markdown("---")
    
    # Model selection with enhanced descriptions
    st.subheader("🤖 AI Model")
    model_options = {
        "Llama 3.1 8B Instant": "llama-3.1-8b-instant",
        "Llama 3.3 70B Versatile": "llama-3.3-70b-versatile", 
        "Llama 3.1 70B Versatile": "llama-3.1-70b-versatile",
        "Mixtral 8x7B": "mixtral-8x7b-32768",
        "Gemma2 9B": "gemma2-9b-it"
    }
    
    selected_model = st.selectbox(
        "Select AI Model",
        options=list(model_options.keys()),
        index=0,
        help="Llama 3.1 8B recommended for faster generation with fewer rate limits"
    )
    
    # Content customization
    st.subheader("🎭 Content Style")
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("Creativity", 0.0, 1.0, 0.3, 0.1,
                              help="Lower = more consistent, Higher = more creative")
    with col2:
        max_results = st.slider("Search Results", 1, 10, 5,
                              help="More results = more comprehensive research")
    
    # Tone selection
    tone_options = ["Professional", "Conversational", "Inspirational", "Humorous", "Authoritative"]
    selected_tone = st.selectbox("Post Tone", tone_options, index=1,
                               help="Choose the tone that matches your brand voice")
    
    # SIMPLIFIED VERIFICATION SETTINGS
    st.markdown("---")
    st.subheader("🔍 Smart Verification")
    
    enable_verification = st.checkbox(
        "Enable Smart Verification", 
        value=True,
        help="Verify factual accuracy and auto-correct if needed"
    )
    
    if enable_verification:
        verification_threshold = st.slider(
            "Accuracy Threshold", 
            90, 100, 95, 1,
            help="Minimum % accuracy required for approval"
        )
        max_iterations = st.slider(
            "Max Editing Attempts", 
            1, 3, 2, 1,
            help="Maximum number of verification-editing cycles"
        )
    
    # RATE LIMIT SETTINGS
    st.markdown("---")
    with st.expander("⚡ Rate Limit Settings", expanded=False):
        min_call_interval = st.slider(
            "Minimum seconds between API calls", 
            3, 15, 8, 1,
            help="Higher values reduce rate limit errors but slow down generation"
        )
        enable_rate_limit_display = st.checkbox(
            "Show rate limit status", 
            value=True,
            help="Show real-time API call timing"
        )
    
    st.markdown("---")
    st.info("💡 **Pro Tip**: Keep 'Minimum seconds between API calls' at 8+ to avoid rate limits!")

# =====================================================================
# 🔍 SERPER API SEARCH
# =====================================================================

def serper_search(query: str, max_results: int = 5, api_key: str = None):
    """Search using Serper API"""
    if not api_key:
        return {"formatted_results": "❌ Serper API key not provided", "urls": [], "raw_text": ""}

    try:
        st.write(f"🔍 Searching for: '{query}'")
        
        url = "https://google.serper.dev/search"
        payload = json.dumps({
            "q": query, 
            "num": max_results,
            "gl": "us",
            "hl": "en"
        })
        headers = {
            'X-API-KEY': api_key,
            'Content-Type': 'application/json'
        }
        
        with st.spinner("🔍 Searching the web for latest information..."):
            response = requests.post(url, headers=headers, data=payload, timeout=30)
            response.raise_for_status()
        
        data = response.json()
        results = []
        url_list = []
        raw_text_parts = []
        
        if 'organic' in data and data['organic']:
            for i, result in enumerate(data['organic'][:max_results], 1):
                title = result.get('title', 'No title')
                link = result.get('link', 'No URL')
                snippet = result.get('snippet', 'No description')
                results.append(f"### 📄 Result {i}: {title}\n**URL:** {link}\n**Summary:** {snippet}\n")
                url_list.append(link)
                raw_text_parts.extend([title, snippet])
        
        if results:
            st.success(f"✅ Found {len(results)} relevant sources")
            return {
                "formatted_results": "\n\n".join(results),
                "urls": url_list,
                "raw_text": " ".join(raw_text_parts)
            }
        else:
            st.warning("⚠️ No search results found for this topic")
            return {
                "formatted_results": "❌ No search results found",
                "urls": [],
                "raw_text": ""
            }
            
    except Exception as e:
        st.error(f"❌ Search failed: {str(e)}")
        return {
            "formatted_results": f"❌ Search failed: {str(e)}",
            "urls": [],
            "raw_text": ""
        }

# =====================================================================
# 🧠 INTELLIGENT GROQ LLM WITH RATE LIMIT MANAGEMENT
# =====================================================================

class RateLimitedGroqLLM:
    """Groq LLM with intelligent rate limiting"""
    
    def __init__(self, api_key, model, temperature=0.7, min_call_interval=8):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.min_call_interval = min_call_interval
        self.last_call_time = 0
        self.call_count = 0
        self.lock = threading.Lock()
        
    def _enforce_rate_limit(self):
        """Enforce minimum time between API calls"""
        with self.lock:
            current_time = time.time()
            time_since_last_call = current_time - self.last_call_time
            
            if time_since_last_call < self.min_call_interval:
                wait_time = self.min_call_interval - time_since_last_call
                if enable_rate_limit_display:
                    st.markdown(f'<div class="rate-limit-status">⏳ Rate limit cooldown: {wait_time:.1f}s</div>', 
                               unsafe_allow_html=True)
                time.sleep(wait_time)
            
            self.last_call_time = time.time()
            self.call_count += 1
    
    def call(self, prompt, system_message=None, max_retries=2, purpose="generation"):
        """Make API call with intelligent rate limiting"""
        # Enforce rate limit before making call
        self._enforce_rate_limit()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        # Adjust tokens based on purpose
        max_tokens = 2000 if purpose == "verification" else 4000
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "top_p": 1,
            "stream": False
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(self.base_url, headers=headers, json=payload, timeout=45)
                
                if response.status_code == 429:
                    # Rate limit hit - wait longer
                    wait_time = (attempt + 1) * 5  # 5, 10, 15 seconds
                    if enable_rate_limit_display:
                        st.markdown(f'<div class="rate-limit-status">⏳ Rate limit hit. Waiting {wait_time}s (attempt {attempt + 1})</div>', 
                                   unsafe_allow_html=True)
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                
                result = response.json()
                return result["choices"][0]["message"]["content"]
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    if purpose == "verification":
                        return None  # Silent fail for verification
                    else:
                        st.error(f"❌ API request failed: {str(e)[:100]}")
                    return None
                else:
                    time.sleep((attempt + 1) * 2)  # Exponential backoff
            
            except Exception as e:
                if purpose == "verification":
                    return None
                else:
                    st.error(f"❌ Unexpected error: {str(e)}")
                return None
        
        return None

# =====================================================================
# 🔍 SMART VERIFICATION SYSTEM (Based on your successful pattern)
# =====================================================================

class SmartVerifierAgent:
    """Verifier agent following your successful pattern"""
    
    def __init__(self, groq_llm):
        self.llm = groq_llm
    
    def verify_post(self, linkedin_post: str, research_findings: str, threshold: int = 95) -> Dict:
        """
        Verify LinkedIn post against research findings
        Returns: {"status": "APPROVED" or "NEEDS_EDIT", "feedback": "specific suggestions", "alignment_percentage": X}
        """
        
        verification_prompt = f"""You are a quality assurance specialist for LinkedIn posts about AI in Agriculture. 
        Your CRITICAL job is to verify that the LinkedIn post accurately reflects the research findings.

        **RESEARCH FINDINGS:**
        {research_findings[:2500]}

        **LINKEDIN POST:**
        {linkedin_post}

        **VERIFICATION PROCESS:**
        1. Compare the research_findings with the LinkedIn post line by line
        2. Check for:
           - Factual consistency (no hallucinations)
           - Statistics match research exactly (same numbers, same context)
           - No added fictional details or unsupported claims
           - Geographic accuracy (don't present global/African data as Nigeria-specific without evidence)
           - Potential vs proven claims (use "can help" not "is improving" for potential benefits)
           - Symptoms/benefits accurately described

        **DECISION CRITERIA:**
        - If post is {threshold}%+ accurate to research: Output EXACTLY "APPROVED"
        - If discrepancies found: Provide 3-5 SPECIFIC, ACTIONABLE editing suggestions

        **EDITING SUGGESTIONS FORMAT (when not approved):**
        Provide clear instructions like:
        - "Add information about [specific innovation] from the research: [quote from research]"
        - "Correct the statistic about [topic] from [wrong number]% to [correct number]% to match research"
        - "Remove unverified claim about [specific detail]"
        - "Change 'in Nigeria' to 'in Africa' or 'globally' because research doesn't mention Nigeria"
        - "Use 'can help' instead of 'is improving' for [benefit] since research shows potential not proven results"
        - "Add citation for statistic about [topic] from research finding: [quote]"

        **CRITICAL RULES:**
        - Be strict about accuracy - farmers depend on this information
        - Focus on evidence-based claims only
        - Nigeria-specific claims MUST have Nigeria-specific research
        - Statistical claims MUST match research numbers exactly
        - If research says "potential benefit", post must say "can help" not "does help"

        **OUTPUT:**
        Start with either "APPROVED" or "NEEDS_EDIT" on its own line.
        Then provide feedback/suggestions if needed."""
        
        try:
            response = self.llm.call(
                verification_prompt,
                "You are a meticulous fact-checker. Follow instructions exactly. Be strict but fair.",
                purpose="verification"
            )
            
            if not response:
                return {
                    "status": "NEEDS_EDIT",
                    "feedback": "Verification failed - system error",
                    "alignment_percentage": 0,
                    "suggestions": ["Verification system error - please review manually"]
                }
            
            # Parse the response
            lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
            
            if not lines:
                return {
                    "status": "NEEDS_EDIT",
                    "feedback": "No verification response received",
                    "alignment_percentage": 0,
                    "suggestions": ["Please review manually"]
                }
            
            first_line = lines[0].upper()
            
            if "APPROVED" in first_line:
                return {
                    "status": "APPROVED",
                    "feedback": "Post accurately reflects research findings",
                    "alignment_percentage": 100,
                    "suggestions": []
                }
            
            # Extract suggestions
            suggestions = []
            for line in lines[1:]:
                if (line.startswith("-") or line.startswith("•") or 
                    line.startswith("1.") or line.startswith("2.") or line.startswith("3.") or
                    "add information" in line.lower() or 
                    "correct" in line.lower() or 
                    "remove" in line.lower() or 
                    "change" in line.lower() or
                    "use " in line.lower() and " instead of " in line.lower()):
                    suggestions.append(line)
            
            # If no structured suggestions found, use the response as feedback
            if not suggestions and len(lines) > 1:
                suggestions = lines[1:3]  # Take first few lines as suggestions
            
            # Calculate alignment based on suggestion count
            suggestion_count = min(len(suggestions), 5)
            alignment = max(0, 100 - (suggestion_count * 20))
            
            return {
                "status": "NEEDS_EDIT",
                "feedback": response,
                "alignment_percentage": alignment,
                "suggestions": suggestions[:5]  # Limit to 5 suggestions
            }
                    
        except Exception as e:
            return {
                "status": "NEEDS_EDIT",
                "feedback": f"Verification error: {str(e)[:100]}",
                "alignment_percentage": 0,
                "suggestions": ["System error occurred - please review manually"]
            }


class SmartEditorAgent:
    """Editor agent that uses verifier's specific suggestions"""
    
    def __init__(self, groq_llm):
        self.llm = groq_llm
    
    def edit_post(self, original_post: str, verifier_feedback: str, research_findings: str) -> str:
        """Edit post based on verifier's specific suggestions"""
        
        # Extract actionable suggestions from feedback
        suggestions = []
        lines = verifier_feedback.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for actionable suggestions
            if (len(line) > 20 and  # Reasonable length for a suggestion
                (line.startswith("-") or 
                 line.startswith("•") or
                 line.startswith("1.") or line.startswith("2.") or line.startswith("3.") or
                 "add information" in line.lower() or 
                 "correct" in line.lower() or 
                 "remove" in line.lower() or 
                 "change" in line.lower() or
                 "use " in line.lower() and " instead of " in line.lower())):
                suggestions.append(line)
        
        if not suggestions:
            # Fallback: look for any lines that seem like instructions
            for line in lines:
                if len(line) > 30 and ("should" in line.lower() or "need" in line.lower() or "must" in line.lower()):
                    suggestions.append(line)
        
        if not suggestions:
            suggestions = ["Please improve factual accuracy based on the research findings"]
        
        editing_prompt = f"""Edit this LinkedIn post based on specific verification feedback:

        **ORIGINAL POST:**
        {original_post}

        **RESEARCH FINDINGS (for reference - do not add new information not in here):**
        {research_findings[:1500]}

        **SPECIFIC EDITING INSTRUCTIONS:**
        {chr(10).join(suggestions[:3])}

        **EDITING RULES:**
        1. Make ONLY the changes specified above
        2. Keep the same tone, style, and structure
        3. Preserve emojis, hashtags, and engagement questions
        4. Maintain similar length (2500-3500 characters)
        5. If a suggestion conflicts with research, prioritize research accuracy
        6. If unsure about a change, keep the original wording

        Return ONLY the edited LinkedIn post, no explanations or notes."""
        
        edited = self.llm.call(
            editing_prompt,
            "You are a precise editor who makes minimal, targeted changes based on specific instructions.",
            purpose="editing"
        )
        
        return clean_text(edited) if edited else original_post


def execute_smart_verification_loop(initial_post: str, search_data: Dict, groq_llm, 
                                   threshold: int = 95, max_iterations: int = 2) -> Tuple[str, List[Dict]]:
    """Smart verification loop using your successful agent pattern"""
    
    verifier = SmartVerifierAgent(groq_llm)
    editor = SmartEditorAgent(groq_llm)
    
    current_post = initial_post
    verification_history = []
    best_post = initial_post
    best_alignment = 0
    
    for iteration in range(1, max_iterations + 1):
        st.markdown(f'<div class="iteration-counter">🔄 Smart Verification {iteration}/{max_iterations}</div>', 
                   unsafe_allow_html=True)
        
        # Step 1: Verify
        with st.spinner(f"🔍 Checking factual accuracy..."):
            verification_result = verifier.verify_post(
                current_post, 
                search_data["formatted_results"], 
                threshold
            )
        
        verification_result["iteration"] = iteration
        verification_history.append(verification_result)
        
        # Track best result
        current_alignment = verification_result.get("alignment_percentage", 0)
        if current_alignment > best_alignment:
            best_alignment = current_alignment
            best_post = current_post
        
        # Display results
        status = verification_result["status"]
        alignment = verification_result["alignment_percentage"]
        
        if status == "APPROVED":
            st.markdown(f'<div class="verification-approved">✅ APPROVED - 100% accurate</div>', 
                       unsafe_allow_html=True)
            st.success(f"✅ Post approved after {iteration} iteration(s)")
            return current_post, verification_history
        
        else:
            st.markdown(f'<div class="verification-pending">⚠️ NEEDS EDIT - {alignment}% accurate</div>', 
                       unsafe_allow_html=True)
            
            # Show specific suggestions
            with st.expander(f"📝 Specific Changes Needed (Iteration {iteration})", expanded=iteration==1):
                if verification_result.get("suggestions"):
                    st.write("**Make these specific changes:**")
                    for i, suggestion in enumerate(verification_result["suggestions"], 1):
                        st.write(f"{i}. {suggestion}")
                else:
                    st.write("**Feedback:**", verification_result["feedback"][:500])
            
            # If not last iteration, edit
            if iteration < max_iterations:
                with st.spinner(f"✍️ Applying suggested edits..."):
                    edited_post = editor.edit_post(
                        current_post,
                        verification_result["feedback"],
                        search_data["formatted_results"]
                    )
                    
                    if edited_post != current_post:
                        current_post = edited_post
                        
                        # Show what changed
                        with st.expander(f"📄 Edited Version", expanded=False):
                            st.text_area(f"Post after iteration {iteration}", 
                                        value=current_post, 
                                        height=200, 
                                        key=f"edited_{iteration}")
                    else:
                        st.info("ℹ️ No changes made - continuing with current version")
    
    # Max iterations reached - use best version
    final_alignment = verification_history[-1].get("alignment_percentage", best_alignment)
    
    if final_alignment >= 80:
        st.markdown(f'<div class="verification-pending">⚠️ BEST AVAILABLE - {final_alignment}% accurate</div>', 
                   unsafe_allow_html=True)
        st.warning(f"⚠️ Using best version with {final_alignment}% accuracy")
        return best_post, verification_history
    elif final_alignment > 0:
        st.markdown(f'<div class="verification-pending">⚠️ MODERATE - {final_alignment}% accurate</div>', 
                   unsafe_allow_html=True)
        st.warning(f"⚠️ Moderate accuracy: {final_alignment}% - review recommended")
        return best_post, verification_history
    else:
        st.markdown(f'<div class="verification-failed">❌ VERIFICATION FAILED</div>', 
                   unsafe_allow_html=True)
        st.error("❌ Verification failed - using initial post")
        return initial_post, verification_history

# =====================================================================
# 📱 CONTENT GENERATION FUNCTIONS
# =====================================================================

def clean_text(text: str) -> str:
    """Clean text by removing markdown code blocks and extra whitespace"""
    if not text:
        return ""
    
    # Remove markdown code blocks
    t = text.strip()
    if t.startswith("```") and t.endswith("```"):
        t = t[3:-3].strip()
    elif t.startswith("`") and t.endswith("`"):
        t = t[1:-1].strip()
    
    # Remove any remaining markdown headers
    t = re.sub(r'^#+\s+', '', t, flags=re.MULTILINE)
    
    return t

def generate_facebook_post(linkedin_post: str, groq_llm) -> str:
    """Generate Facebook version from LinkedIn post"""
    if not linkedin_post or len(linkedin_post) < 100:
        return "Facebook post generation skipped"
    
    facebook_prompt = f"""Create a Facebook post from this LinkedIn content:

    {linkedin_post[:1200]}

    Requirements:
    - Friendly, conversational tone
    - 3-5 emojis naturally placed
    - 500-700 characters
    - Start with a hook question
    - End with engagement question
    - Include 3-5 relevant hashtags
    - Make it feel personal and authentic

    Facebook post only:"""
    
    facebook_post = groq_llm.call(
        facebook_prompt,
        "You create engaging, conversational Facebook content that drives interaction.",
        purpose="facebook"
    )
    
    return clean_text(facebook_post) if facebook_post else "Failed to generate Facebook post"

def generate_whatsapp_hook(linkedin_post: str, groq_llm) -> str:
    """Generate WhatsApp teaser"""
    if not linkedin_post or len(linkedin_post) < 100:
        return "WhatsApp hook generation skipped"
    
    whatsapp_prompt = f"""Create a short WhatsApp teaser from this content:

    {linkedin_post[:800]}

    Requirements:
    - 1-2 lines maximum (very concise)
    - Casual, intriguing hook
    - Use 1-2 relevant emojis
    - End with: 🔗 Read full post: [Link]

    WhatsApp message only:"""
    
    whatsapp_hook = groq_llm.call(
        whatsapp_prompt,
        "You create short, intriguing messages that drive clicks.",
        purpose="whatsapp"
    )
    
    return clean_text(whatsapp_hook) if whatsapp_hook else "Failed to generate WhatsApp hook"

# =====================================================================
# 💼 MAIN LINKEDIN POST GENERATION
# =====================================================================

def build_linkedin_prompt(query: str, search_results: str, tone: str) -> str:
    """Build prompt for LinkedIn post generation"""
    date_str = datetime.now().strftime("%Y-%m-%d")
    
    return f"""Create an engaging LinkedIn post about: "{query}"

TONE: {tone}, informative, professional
CURRENT DATE: {date_str}

RESEARCH FINDINGS:
{search_results[:2000]}

POST STRUCTURE:
1. [HOOK] - Start with surprising fact/statistic + emoji
2. [PERSONAL CONNECTION] - 1-2 sentences as AI in Agriculture enthusiast
3. [MAIN TREND] - What's happening now in this space
4. [KEY INNOVATIONS] - 3-4 bullet points with emojis (🤖 🛰 💧 📊)
5. [IMPACT DATA] - Evidence-based statistics (only from research)
6. [FUTURE OUTLOOK] - What's coming next
7. [ENGAGEMENT QUESTION] - End with thought-provoking question
8. [HASHTAGS] - Include #9jaAI_Farmer + 4-5 relevant hashtags

CRITICAL RULES:
- Use evidence-based language: "can help", "may improve", "has potential"
- Only use statistics explicitly in research
- If no Nigeria-specific data, say "in the region" or "globally"
- Keep paragraphs short (1-2 lines)
- Target length: 2500-3200 characters
- Focus on solutions and positive developments
- Be informative but not overly technical

Return ONLY the LinkedIn post text, no explanations."""

def build_system_message():
    return """You are an "Educator-Innovator" - a knowledgeable content creator who makes complex AI and agriculture topics accessible and engaging.

Your expertise:
- Creating compelling hooks with surprising facts
- Structuring content for maximum readability  
- Using emojis and formatting for visual appeal
- Driving engagement through thoughtful questions
- Maintaining credibility with evidence-based claims

CRITICAL: You never invent statistics or misrepresent geographic scope. You distinguish between demonstrated outcomes and potential benefits."""

def execute_linkedin_workflow(query: str, groq_llm, max_results: int, serper_key: str, tone: str):
    """Main LinkedIn post generation workflow"""
    
    # 1. Search for information
    with st.status("🔍 Researching your topic...", expanded=True) as status:
        search_data = serper_search(query, max_results, serper_key)
        
        if search_data["formatted_results"].startswith("❌"):
            status.update(label="❌ Research failed", state="error")
            return None
            
        status.update(label="✅ Research completed", state="complete")
    
    # 2. Generate initial LinkedIn post
    with st.status("✍️ Crafting your LinkedIn post...", expanded=True) as status:
        prompt = build_linkedin_prompt(query, search_data["formatted_results"], tone)
        system_msg = build_system_message()
        
        linkedin_post = groq_llm.call(prompt, system_msg, purpose="generation")
        
        if not linkedin_post:
            status.update(label="❌ Failed to generate post", state="error")
            return None
        
        linkedin_post = clean_text(linkedin_post)
        char_len = len(linkedin_post)
        
        if char_len < 1500:
            # Post is too short, try to expand
            expansion_prompt = f"Expand this LinkedIn post to 2500-3000 characters while keeping the same structure and information:\n\n{linkedin_post}"
            expanded = groq_llm.call(expansion_prompt, "You expand content while maintaining quality.", purpose="expansion")
            if expanded:
                linkedin_post = clean_text(expanded)
                char_len = len(linkedin_post)
        
        st.info(f"📊 Generated post: {char_len} characters")
        status.update(label=f"✅ Post generated ({char_len} chars)", state="complete")
    
    # 3. Smart Verification (if enabled)
    if enable_verification:
        with st.status("🔍 Verifying factual accuracy...", expanded=True) as status:
            final_post, verification_history = execute_smart_verification_loop(
                linkedin_post,
                search_data,
                groq_llm,
                verification_threshold,
                max_iterations
            )
            
            # Update status based on verification result
            final_result = verification_history[-1]
            final_alignment = final_result.get("alignment_percentage", 0)
            
            if final_result.get("status") == "APPROVED":
                status.update(label=f"✅ Verified: 100% accurate", state="complete")
            elif final_alignment >= 80:
                status.update(label=f"⚠️ Verified: {final_alignment}% accurate", state="complete")
            else:
                status.update(label=f"❌ Verification issues: {final_alignment}%", state="error")
    else:
        final_post = linkedin_post
        verification_history = []
    
    # 4. Generate Facebook post
    facebook_post = "Facebook post generation paused"
    if groq_llm and final_post and len(final_post) > 100:
        facebook_post = generate_facebook_post(final_post, groq_llm)
    
    # 5. Generate WhatsApp hook
    whatsapp_hook = "WhatsApp hook generation paused"
    if groq_llm and final_post and len(final_post) > 100:
        whatsapp_hook = generate_whatsapp_hook(final_post, groq_llm)
    
    return {
        "linkedin_post": final_post,
        "character_count": len(final_post),
        "search_results": search_data["formatted_results"],
        "search_urls": search_data["urls"],
        "facebook_post": facebook_post,
        "whatsapp_hook": whatsapp_hook,
        "verification_history": verification_history,
        "initial_linkedin_post": linkedin_post
    }

# =====================================================================
# 📱 MAIN APPLICATION
# =====================================================================

def main():
    """Main application logic"""
    
    # Topic input
    st.subheader("🎯 What 'AI in Agriculture' Topic do you want to post about?")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_area(
            "Enter your topic:", 
            placeholder="e.g., AI in Nigerian agriculture, Climate smart farming technologies, AI for crop disease detection...",
            height=100,
            label_visibility="collapsed",
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        generate_clicked = st.button("🚀 Generate Content", use_container_width=True, type="primary")
    
    # Get API keys (prefer environment variables)
    final_groq_key = groq_api_key.strip() if groq_api_key and groq_api_key.strip() else os.getenv("GROQ_API_KEY")
    final_serper_key = serper_api_key.strip() if serper_api_key and serper_api_key.strip() else os.getenv("SERPER_API_KEY")
    
    # Validate API keys
    if not final_groq_key:
        st.error("""
        ❌ **Groq API Key Required**
        Please enter your Groq API key in the sidebar or set GROQ_API_KEY environment variable.
        """)
        st.info("💡 **Get your free API key:** https://console.groq.com")
        return
        
    if not final_serper_key:
        st.error("""
        ❌ **Serper API Key Required** 
        Please enter your Serper API key in the sidebar to enable web search.
        """)
        st.info("💡 **Get your free API key:** https://serper.dev - 2,500 free searches")
        return
    
    # Initialize Groq LLM with rate limiting
    groq_llm = RateLimitedGroqLLM(
        api_key=final_groq_key,
        model=model_options[selected_model],
        temperature=temperature,
        min_call_interval=min_call_interval
    )
    
    # Execute content generation
    if generate_clicked:
        if not query.strip():
            st.warning("⚠️ Please enter a topic for your content.")
        else:
            try:
                # Show generation progress
                with st.spinner("🎯 Creating your multi-platform content..."):
                    time.sleep(1)  # Initial delay
                
                # Execute workflow
                result = execute_linkedin_workflow(
                    query, groq_llm, max_results, final_serper_key, selected_tone
                )
                
                if result is None:
                    st.error("❌ Content generation failed. Please check your API keys and try again.")
                else:
                    st.success("🎉 Your multi-platform content is ready!")
                    
                    # Character count display
                    char_count_val = result["character_count"]
                    st.markdown(f"""
                    <div class="char-counter char-warning">
                        <strong>Character Count:</strong> {char_count_val} characters
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create tabs for outputs
                    tab_names = ["💼 LinkedIn", "📱 Facebook", "💬 WhatsApp", "🔍 Research URLs"]
                    if enable_verification and result["verification_history"]:
                        tab_names.append("🔍 Verification")
                    tabs = st.tabs(tab_names)
                    
                    with tabs[0]:  # LinkedIn
                        st.subheader("💼 Your LinkedIn Post")
                        
                        # Show verification status if applicable
                        if enable_verification and result["verification_history"]:
                            final_verification = result["verification_history"][-1]
                            status = final_verification.get("status", "")
                            alignment = final_verification.get("alignment_percentage", 0)
                            
                            if status == "APPROVED":
                                st.markdown(f'<div class="verification-approved">✅ VERIFIED & APPROVED - 100% accurate</div>', 
                                           unsafe_allow_html=True)
                            elif alignment >= 80:
                                st.markdown(f'<div class="verification-pending">⚠️ VERIFIED - {alignment}% accurate</div>', 
                                           unsafe_allow_html=True)
                            elif alignment > 0:
                                st.markdown(f'<div class="verification-pending">⚠️ REVIEW RECOMMENDED - {alignment}% accurate</div>', 
                                           unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="verification-failed">❌ VERIFICATION FAILED - manual review needed</div>', 
                                           unsafe_allow_html=True)
                        
                        st.text_area(
                            "Copy your LinkedIn post below:",
                            value=result["linkedin_post"],
                            height=400,
                            key="linkedin_post_output"
                        )
                    
                    with tabs[1]:  # Facebook
                        st.subheader("📱 Facebook Post")
                        st.info("🎭 **Facebook-Optimized Version**")
                        st.text_area(
                            "Copy your Facebook post below:",
                            value=result["facebook_post"],
                            height=300,
                            key="facebook_post_output"
                        )
                    
                    with tabs[2]:  # WhatsApp
                        st.subheader("💬 WhatsApp Hook")
                        st.info("🎣 **Ultra-Short Teaser**")
                        st.text_area(
                            "Copy your WhatsApp message below:",
                            value=result["whatsapp_hook"],
                            height=150,
                            key="whatsapp_hook_output"
                        )
                    
                    with tabs[3]:  # Research URLs
                        st.subheader("🔍 Research URLs")
                        if result["search_urls"]:
                            st.info(f"📚 **Found {len(result['search_urls'])} URLs**")
                            url_text = "\n".join(result["search_urls"])
                            st.text_area(
                                "All search URLs:",
                                value=url_text,
                                height=200,
                                key="urls_output"
                            )
                        else:
                            st.warning("No URLs found in the search results.")
                    
                    # Verification tab (if enabled)
                    if enable_verification and result["verification_history"] and len(tabs) > 4:
                        with tabs[4]:
                            st.subheader("🔍 Verification Details")
                            history = result["verification_history"]
                            
                            for i, verification in enumerate(history, 1):
                                with st.expander(f"Iteration {i}: {verification.get('status', 'Unknown')} - {verification.get('alignment_percentage', 0)}%", 
                                               expanded=i == len(history)):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.write(f"**Alignment:** {verification.get('alignment_percentage', 0)}%")
                                        st.write(f"**Status:** {verification.get('status', 'Unknown')}")
                                    with col2:
                                        st.write(f"**Iteration:** {verification.get('iteration', i)}")
                                    
                                    if verification.get("suggestions"):
                                        st.write("**Specific Suggestions Made:**")
                                        for j, suggestion in enumerate(verification["suggestions"], 1):
                                            st.write(f"{j}. {suggestion}")
                                    elif verification.get("feedback"):
                                        st.write("**Feedback:**")
                                        st.write(verification["feedback"][:500])
                    
            except Exception as e:
                st.error(f"❌ Error during content generation: {str(e)}")
    
    # Footer
    st.markdown("---")
    footer_col1, footer_col2, footer_col3 = st.columns([2, 1, 1])
    with footer_col1:
        st.caption("💼 **Multi-Platform Content Generator Pro**")
    with footer_col2:
        st.caption("⚡ Powered by 9jaAI_Farmer")
    with footer_col3:
        st.caption("🔍 Enhanced by Groq & Serper API")

# =====================================================================
# 🚀 RUN APPLICATION
# =====================================================================

if __name__ == "__main__":
    main()