import os
import streamlit as st
import requests
import json
from datetime import datetime
import time
import textwrap
import re
from typing import Dict, List, Tuple, Optional

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
    
    .verification-warning {
        background-color: #ffeaa7;
        color: #5d4037;
        padding: 0.5rem;
        border-radius: 5px;
        border: 1px solid #ffcc80;
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
    
    /* Rate limit warning */
    .rate-limit-warning {
        background-color: #fff3cd;
        color: #856404;
        border-left: 4px solid #ffc107;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0 5px 5px 0;
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
            help="Get your free API key from https://console.groq.com"
        )
        
        serper_api_key = st.text_input(
            "Serper API Key", 
            type="password",
            placeholder="Enter your Serper API key",
            help="Get free key from https://serper.dev - 2,500 search free"
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
        index=0,  # Default to 8B for faster, less rate-limited
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
    
    # Verification settings
    st.markdown("---")
    st.subheader("🔍 Verification Settings")
    verification_threshold = st.slider("Factual Accuracy Threshold", 80, 100, 95, 1,
                                     help="Minimum % of factual alignment required for approval (95% recommended)")
    max_iterations = st.slider("Max Verification Iterations", 1, 5, 3, 1,
                             help="Maximum number of verification-editing cycles")
    
    # Advanced verification options
    with st.expander("⚙️ Advanced Verification Options"):
        enable_simple_fallback = st.checkbox("Enable Simple Fallback Check", value=True,
                                           help="Use keyword matching if LLM verification fails")
        skip_editing_on_limits = st.checkbox("Skip editing on rate limits", value=True,
                                           help="Skip editing phase if rate limits are hit")
        debug_mode = st.checkbox("Enable Debug Mode", value=False,
                               help="Show raw verification responses for troubleshooting")
    
    st.markdown("---")
    
    # Performance tips based on model selection
    st.subheader("⚡ Performance Tips")
    if "70B" in selected_model or "Mixtral" in selected_model:
        st.markdown('<div class="rate-limit-warning">⚠️ <strong>Large Model Selected</strong><br>More likely to hit rate limits</div>', 
                   unsafe_allow_html=True)
        st.info("💡 For verification workflows:")
        st.markdown("- Consider **Llama 3.1 8B** for fewer limits")
        st.markdown("- Enable **'Skip editing on rate limits'**")
    else:
        st.success("✅ **Good choice!** This model should handle verification well.")
    
    st.markdown("---")
    st.info("💡 **Pro Tip**: Start with Llama 3.1 8B Instant to avoid rate limits!")

# =====================================================================
# 🔍 ENHANCED SERPER API SEARCH - NOW RETURNS URLS
# =====================================================================

def serper_search(query: str, max_results: int = 5, api_key: str = None):
    """Search using Serper API - returns both formatted results and clean URLs"""
    if not api_key:
        return {"formatted_results": "❌ Serper API key not provided", "urls": []}

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
        
        # Process organic results
        if 'organic' in data and data['organic']:
            for i, result in enumerate(data['organic'][:max_results], 1):
                title = result.get('title', 'No title')
                link = result.get('link', 'No URL')
                snippet = result.get('snippet', 'No description')
                results.append(f"### 📄 Result {i}: {title}\n**URL:** {link}\n**Summary:** {snippet}\n")
                url_list.append(link)
        
        # Also check for news results if organic results are limited
        if len(results) < max_results and 'news' in data and data['news']:
            news_to_add = max_results - len(results)
            for i, result in enumerate(data['news'][:news_to_add], len(results) + 1):
                title = result.get('title', 'No title')
                link = result.get('link', 'No URL')
                snippet = result.get('snippet', 'No description')
                results.append(f"### 📰 News {i}: {title}\n**URL:** {link}\n**Summary:** {snippet}\n")
                url_list.append(link)
        
        if results:
            st.success(f"✅ Found {len(results)} relevant sources")
            return {
                "formatted_results": "\n\n".join(results),
                "urls": url_list
            }
        else:
            st.warning("⚠️ No search results found for this topic")
            return {
                "formatted_results": "❌ No search results found",
                "urls": []
            }
            
    except requests.exceptions.RequestException as e:
        error_msg = f"Search failed: {str(e)}"
        st.error(f"❌ {error_msg}")
        return {
            "formatted_results": f"❌ {error_msg}",
            "urls": []
        }
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        st.error(f"❌ {error_msg}")
        return {
            "formatted_results": f"❌ {error_msg}",
            "urls": []
        }

# =====================================================================
# 🧠 GROQ LLM INTEGRATION WITH RATE LIMIT HANDLING
# =====================================================================

class GroqLLM:
    """Custom Groq LLM wrapper with rate limit handling"""

    def __init__(self, api_key, model, temperature=0.7):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        
    def call(self, prompt, system_message=None, max_retries=3):
        """Make API call to Groq with retry logic"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 4000,
            "top_p": 1,
            "stream": False
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
                
                if response.status_code == 429:
                    # Rate limit hit
                    wait_time = 2 ** attempt  # Exponential backoff
                    st.warning(f"⏳ Rate limit hit. Waiting {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                    
                response.raise_for_status()
                
                result = response.json()
                return result["choices"][0]["message"]["content"]
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:  # Last attempt
                    st.error(f"❌ Groq API Request failed: {str(e)}")
                    if hasattr(e, 'response') and e.response is not None:
                        try:
                            error_data = e.response.json()
                            error_msg = error_data.get('error', {}).get('message', 'Unknown error')
                            st.error(f"Error details: {error_msg}")
                        except:
                            st.error(f"Response: {e.response.text[:200]}")
                    return None
                else:
                    wait_time = 2 ** attempt
                    st.warning(f"⚠️ Request failed, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
                return None
        
        return None

# =====================================================================
# 🔍 VERIFICATION SYSTEM - VERIFIER AGENT & EDITOR AGENT
# =====================================================================

class VerifierAgent:
    """Agent that verifies factual alignment between LinkedIn post and search results"""
    
    def __init__(self, groq_llm):
        self.llm = groq_llm
        
    def verify_factual_alignment(self, linkedin_post: str, search_results: str, threshold: int = 95) -> Dict:
        """
        Verify if the LinkedIn post aligns factually with search results.
        Returns dict with verification results.
        """
        # First, check if search results are meaningful
        if not search_results or "No search results" in search_results or "❌" in search_results:
            return {
                "alignment_percentage": 0,
                "status": "NEEDS_REVISION",
                "feedback": "No search results available for verification. Please try a different search query.",
                "unsupported_claims": ["No research data available"],
                "partially_supported_claims": [],
                "supported_claims": [],
                "verification_notes": "SEARCH_RESULTS_MISSING"
            }
        
        verification_prompt = f"""
        You are a Fact-Checking Agent. Your task is to analyze a LinkedIn post against provided research findings
        and determine the factual alignment percentage.
        
        INSTRUCTIONS:
        1. Read the LinkedIn post carefully.
        2. Read the research findings thoroughly.
        3. Identify 3-5 key factual claims in the LinkedIn post (statistics, specific developments, outcomes, innovations).
        4. For EACH claim, determine if it is:
           - FULLY_SUPPORTED: Directly stated in research with specific evidence
           - PARTIALLY_SUPPORTED: Generally supported but lacking specific details
           - UNSUPPORTED: Not mentioned or contradicted by research
        
        5. Calculate alignment percentage using this formula:
           Alignment = (FULLY_SUPPORTED × 1.0 + PARTIALLY_SUPPORTED × 0.5) / Total_Claims × 100
        
        RESEARCH FINDINGS:
        {search_results[:3000]}  # Limit to avoid token limits
        
        LINKEDIN POST:
        {linkedin_post}
        
        OUTPUT FORMAT (MUST BE VALID JSON):
        {{
            "alignment_percentage": 85.5,
            "status": "APPROVED",
            "feedback": "Specific feedback here",
            "unsupported_claims": ["Claim 1", "Claim 2"],
            "partially_supported_claims": ["Claim 3"],
            "supported_claims": ["Claim 4", "Claim 5"],
            "total_claims_analyzed": 5
        }}
        
        IMPORTANT: Return ONLY the JSON object, nothing else.
        """
        
        try:
            response = self.llm.call(
                verification_prompt,
                "You are a meticulous fact-checker. Return ONLY valid JSON with the specified fields. No additional text."
            )
            
            if not response:
                raise ValueError("Empty response from LLM")
            
            # Show debug info if enabled
            if debug_mode:
                with st.expander("🔍 Debug: Raw Verification Response", expanded=False):
                    st.code(response, language="text")
            
            # Clean and parse JSON
            cleaned_response = re.sub(r'```json\s*|\s*```', '', response.strip())
            
            # Try to parse JSON
            result = json.loads(cleaned_response)
            
            # Validate required fields
            required_fields = ["alignment_percentage", "feedback"]
            for field in required_fields:
                if field not in result:
                    raise KeyError(f"Missing required field: {field}")
            
            # Ensure alignment_percentage is a number
            alignment = float(result.get("alignment_percentage", 0))
            
            # Set default arrays if missing
            if "unsupported_claims" not in result:
                result["unsupported_claims"] = []
            if "partially_supported_claims" not in result:
                result["partially_supported_claims"] = []
            if "supported_claims" not in result:
                result["supported_claims"] = []
            
            # Determine status
            result["status"] = "APPROVED" if alignment >= threshold else "NEEDS_REVISION"
            
            return result
            
        except json.JSONDecodeError as e:
            st.error(f"❌ Failed to parse verification JSON: {str(e)[:100]}")
            
            # Try to extract percentage from text
            percentage_match = re.search(r'(\d+\.?\d*)%', response or "")
            extracted_percentage = float(percentage_match.group(1)) if percentage_match else 0
            
            return {
                "alignment_percentage": extracted_percentage,
                "status": "NEEDS_REVISION",
                "feedback": f"JSON parsing failed. Extracted alignment: {extracted_percentage}%",
                "unsupported_claims": ["Verification system error - manual review needed"],
                "partially_supported_claims": [],
                "supported_claims": [],
                "verification_notes": "JSON_PARSE_ERROR"
            }
            
        except Exception as e:
            st.error(f"❌ Verification error: {str(e)}")
            return {
                "alignment_percentage": 0,
                "status": "NEEDS_REVISION",
                "feedback": f"Verification system error: {str(e)[:200]}",
                "unsupported_claims": ["System error occurred"],
                "partially_supported_claims": [],
                "supported_claims": [],
                "verification_notes": "SYSTEM_ERROR"
            }

class EditorAgent:
    """Agent that edits LinkedIn posts based on verifier feedback"""
    
    def __init__(self, groq_llm):
        self.llm = groq_llm
        
    def edit_post(self, original_post: str, verification_feedback: Dict, search_results: str, target_tone: str) -> str:
        """
        Edit the LinkedIn post to address verification feedback.
        """
        unsupported_claims = verification_feedback.get("unsupported_claims", [])
        partially_supported = verification_feedback.get("partially_supported_claims", [])
        feedback_text = verification_feedback.get("feedback", "")
        
        editing_prompt = f"""
        You are an Expert Editor Agent. Your task is to revise a LinkedIn post to improve its factual accuracy
        while maintaining its engaging style and structure.
        
        ORIGINAL LINKEDIN POST:
        {original_post}
        
        VERIFICATION FEEDBACK:
        {feedback_text}
        
        UNSUPPORTED CLAIMS (MUST BE REMOVED OR REWRITTEN):
        {chr(10).join(f'- {claim}' for claim in unsupported_claims) if unsupported_claims else 'None identified'}
        
        PARTIALLY SUPPORTED CLAIMS (SHOULD BE REVISED FOR ACCURACY):
        {chr(10).join(f'- {claim}' for claim in partially_supported) if partially_supported else 'None identified'}
        
        RESEARCH FINDINGS (FOR REFERENCE):
        {search_results[:2000]}
        
        EDITING INSTRUCTIONS:
        1. REMOVE or REPLACE all unsupported claims with evidence-based statements from the research.
        2. REVISE partially supported claims to be more accurate and precise.
        3. KEEP the overall structure, tone ({target_tone}), and formatting of the original post.
        4. PRESERVE emoji usage, bullet points, hashtags, and engagement elements.
        5. If research doesn't support a claim, either:
           - Remove it entirely
           - Rephrase it as a potential benefit (use "can help", "may improve", "has potential to")
           - Replace with a different, supported insight from the research
        6. Maintain the post's length and readability.
        7. DO NOT add new unsupported claims.
        8. Ensure all statistics and specific data points are explicitly in the research.
        
        Return ONLY the revised LinkedIn post, nothing else.
        """
        
        edited_post = self.llm.call(
            editing_prompt,
            "You are a skilled editor who specializes in making technical content both accurate and engaging. You excel at replacing unsupported claims with evidence-based statements while maintaining the original post's voice and structure."
        )
        
        return clean_text(edited_post) if edited_post else original_post

# =====================================================================
# 🛡️ ENHANCED VERIFICATION LOOP WITH FALLBACKS
# =====================================================================

def simple_factual_check(post: str, search_results: str, threshold: int) -> Dict:
    """Fallback simple factual checking using keyword matching"""
    if not search_results or len(search_results) < 100:
        return {
            "alignment_percentage": 0,
            "status": "NEEDS_REVISION",
            "feedback": "Insufficient search results for proper verification",
            "unsupported_claims": ["Cannot verify without adequate research data"],
            "partially_supported_claims": [],
            "supported_claims": [],
            "verification_notes": "INSUFFICIENT_DATA"
        }
    
    # Extract key terms from search results
    search_lower = search_results.lower()
    post_lower = post.lower()
    
    # Count matching technical terms (simplified approach)
    tech_terms = ["ai", "artificial intelligence", "machine learning", "agriculture", "farm", 
                  "crop", "yield", "sensor", "drone", "iot", "data", "analysis", "technology",
                  "efficiency", "productivity", "sustainable", "climate", "precision", "monitoring"]
    
    matches = sum(1 for term in tech_terms if term in post_lower and term in search_lower)
    total_terms = sum(1 for term in tech_terms if term in post_lower)
    
    if total_terms > 0:
        alignment = (matches / total_terms) * 100
    else:
        alignment = 0
    
    status = "APPROVED" if alignment >= threshold else "NEEDS_REVISION"
    
    return {
        "alignment_percentage": alignment,
        "status": status,
        "feedback": f"Simple keyword analysis: {matches}/{total_terms} technical terms match research. Estimated alignment: {alignment:.1f}%",
        "unsupported_claims": ["Using simplified verification due to system limitations"],
        "partially_supported_claims": [],
        "supported_claims": [],
        "verification_notes": "SIMPLE_FALLBACK_CHECK"
    }

def safe_verification_call(verifier, post, search_results, threshold, iteration, max_retries=2):
    """Wrapper for verification calls with better error handling"""
    for retry in range(max_retries):
        try:
            result = verifier.verify_factual_alignment(post, search_results, threshold)
            
            # Check if we got a valid result
            if result.get("alignment_percentage", 0) > 0:
                return result
            elif retry < max_retries - 1:
                st.warning(f"⚠️ Verification returned 0%, retrying... (Attempt {retry + 1})")
                time.sleep(2 ** retry)  # Exponential backoff
                
        except Exception as e:
            if retry < max_retries - 1:
                st.warning(f"⚠️ Verification error: {str(e)[:100]}... Retrying in {2**retry}s")
                time.sleep(2 ** retry)
            else:
                st.error(f"❌ Verification failed after {max_retries} attempts")
                # Try simple fallback if enabled
                if enable_simple_fallback:
                    st.info("🔄 Trying simple keyword-based verification...")
                    return simple_factual_check(post, search_results, threshold)
                else:
                    return {
                        "alignment_percentage": 0,
                        "status": "NEEDS_REVISION",
                        "feedback": f"Verification system error after {max_retries} attempts",
                        "unsupported_claims": ["Verification system temporarily unavailable"],
                        "partially_supported_claims": [],
                        "supported_claims": [],
                        "verification_notes": "VERIFICATION_ERROR"
                    }
    
    # Final fallback
    if enable_simple_fallback:
        return simple_factual_check(post, search_results, threshold)
    else:
        return {
            "alignment_percentage": 0,
            "status": "NEEDS_REVISION",
            "feedback": "All verification attempts failed",
            "unsupported_claims": ["Verification unavailable"],
            "partially_supported_claims": [],
            "supported_claims": [],
            "verification_notes": "ALL_ATTEMPTS_FAILED"
        }

def safe_editing_call(editor, post, verification_result, search_results, tone, iteration, max_retries=2):
    """Wrapper for editing calls with better error handling"""
    # Skip editing if rate limits are a concern
    if skip_editing_on_limits and verification_result.get("verification_notes") == "JSON_PARSE_ERROR":
        st.warning("⏳ Skipping editing due to previous rate limit issues")
        return post
    
    for retry in range(max_retries):
        try:
            edited = editor.edit_post(post, verification_result, search_results, tone)
            if edited and edited != post:
                return edited
            elif retry < max_retries - 1:
                st.warning(f"⚠️ Editing didn't produce changes, retrying... (Attempt {retry + 1})")
                time.sleep(2 ** retry)
        except Exception as e:
            if retry < max_retries - 1:
                st.warning(f"⚠️ Editing error: {str(e)[:100]}... Retrying in {2**retry}s")
                time.sleep(2 ** retry)
            else:
                st.error(f"❌ Editing failed after {max_retries} attempts")
                return post  # Return original if editing fails
    
    return post  # Return original if all retries fail

def execute_verification_loop(initial_linkedin_post: str, search_data: Dict, groq_llm, 
                              selected_tone: str, threshold: int, max_iterations: int) -> Tuple[str, List[Dict]]:
    """
    Execute the verification-editing loop until approval or max iterations reached.
    Returns: (final_linkedin_post, verification_history)
    """
    verifier = VerifierAgent(groq_llm)
    editor = EditorAgent(groq_llm)
    
    current_post = initial_linkedin_post
    verification_history = []
    best_post = initial_linkedin_post
    best_alignment = 0
    
    for iteration in range(1, max_iterations + 1):
        # Display iteration counter
        st.markdown(f'<div class="iteration-counter">🔄 Verification Iteration {iteration}/{max_iterations}</div>', 
                   unsafe_allow_html=True)
        
        # Step 1: Verify the current post
        with st.spinner(f"🔍 Verifying factual alignment (Iteration {iteration})..."):
            verification_result = safe_verification_call(
                verifier, current_post, search_data["formatted_results"], threshold, iteration
            )
        
        # Add iteration info to result
        verification_result["iteration"] = iteration
        verification_history.append(verification_result)
        
        # Track best result
        current_alignment = verification_result.get("alignment_percentage", 0)
        if current_alignment > best_alignment:
            best_alignment = current_alignment
            best_post = current_post
        
        # Display verification result
        alignment_pct = verification_result.get("alignment_percentage", 0)
        status = verification_result.get("status", "NEEDS_REVISION")
        
        if status == "APPROVED":
            st.markdown(f'<div class="verification-approved">✅ Approved! Factual Alignment: {alignment_pct:.1f}%</div>', 
                       unsafe_allow_html=True)
            st.success(f"✅ Post approved after {iteration} iteration(s)")
            return current_post, verification_history
        
        else:
            # Needs revision
            if alignment_pct > 0:
                st.markdown(f'<div class="verification-pending">⚠️ Needs Revision: Factual Alignment: {alignment_pct:.1f}% (Target: {threshold}%)</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="verification-warning">⚠️ Verification Issue: Using fallback method</div>', 
                           unsafe_allow_html=True)
            
            # Show feedback
            with st.expander(f"📝 Verifier Feedback (Iteration {iteration})", expanded=iteration==1):
                st.write(f"**Alignment Score:** {alignment_pct:.1f}%")
                st.write(f"**Verification Method:** {verification_result.get('verification_notes', 'LLM Verification')}")
                st.write(f"**Feedback:** {verification_result.get('feedback', 'No specific feedback')}")
                
                if verification_result.get("unsupported_claims"):
                    st.write("**Unsupported Claims to Fix:**")
                    for claim in verification_result.get("unsupported_claims", []):
                        st.write(f"- {claim}")
            
            # If not the last iteration, proceed to editing
            if iteration < max_iterations:
                # Skip editing if simple fallback was used
                if verification_result.get("verification_notes") == "SIMPLE_FALLBACK_CHECK":
                    st.info("ℹ️ Using simple verification - skipping editing phase")
                    continue
                
                with st.spinner(f"✍️ Editing post based on feedback (Iteration {iteration})..."):
                    edited_post = safe_editing_call(
                        editor, current_post, verification_result, 
                        search_data["formatted_results"], selected_tone, iteration
                    )
                    
                    if edited_post and edited_post != current_post:
                        current_post = edited_post
                        # Show the edited version
                        with st.expander(f"📄 Edited Version (Iteration {iteration})", expanded=False):
                            st.text_area(f"Post after iteration {iteration}", value=current_post, height=200, key=f"post_iteration_{iteration}")
                    else:
                        st.info(f"ℹ️ No significant changes made in iteration {iteration}")
    
    # If we reach here, max iterations reached without approval
    final_alignment = verification_history[-1].get("alignment_percentage", best_alignment)
    
    if final_alignment > 70:
        st.markdown(f'<div class="verification-pending">⚠️ Max Iterations Reached: Good Alignment: {final_alignment:.1f}%</div>', 
                   unsafe_allow_html=True)
        st.warning(f"⚠️ Maximum iterations ({max_iterations}) reached. Using best version with {final_alignment:.1f}% alignment.")
    elif final_alignment > 0:
        st.markdown(f'<div class="verification-warning">⚠️ Max Iterations Reached: Moderate Alignment: {final_alignment:.1f}%</div>', 
                   unsafe_allow_html=True)
        st.warning(f"⚠️ Maximum iterations ({max_iterations}) reached. Moderate alignment: {final_alignment:.1f}%")
    else:
        st.markdown(f'<div class="verification-failed">❌ Verification System Failed</div>', 
                   unsafe_allow_html=True)
        st.error(f"❌ Verification system failed. Using initial version with manual review recommended.")
        best_post = initial_linkedin_post
    
    return best_post, verification_history

# =====================================================================
# 📱 ENHANCED FACEBOOK POST GENERATION
# =====================================================================

def generate_facebook_post(linkedin_post, groq_llm):
    """Generate Facebook version from LinkedIn post - proper Facebook-optimized post"""
    
    # If LinkedIn post is empty or too short, return placeholder
    if not linkedin_post or len(linkedin_post) < 100:
        return "Failed to generate Facebook post: LinkedIn content too short"
    
    facebook_prompt = f"""
    Transform this LinkedIn post into a complete, engaging Facebook post:
    
    KEY REQUIREMENTS FOR FACEBOOK:
    - LENGTH: 600-800 characters (substantial but not too long)
    - TONE: Conversational, friendly, relatable
    - Use emojis naturally throughout (3-5 emojis)
    - Include a compelling hook at the beginning
    - Share key insights in an easy-to-understand way
    - End with an engaging question to encourage comments
    - Include 3-5 relevant hashtags at the end
    - Make it feel personal and authentic
    - Break up long paragraphs for better readability
    
    IMPORTANT: This should be a proper Facebook post with actual content, not just a teaser.
    It should provide value while being optimized for Facebook's casual, social environment.
    
    LINKEDIN POST:
    {linkedin_post[:2000]}
    
    Return ONLY the Facebook post text, nothing else.
    """
    
    with st.spinner("📱 Creating Facebook post..."):
        facebook_post = groq_llm.call(
            facebook_prompt,
            "You are a social media expert who specializes in creating engaging Facebook content. You know how to adapt professional topics for Facebook's friendly, conversational audience while keeping the content substantial and valuable."
        )
    
    return clean_text(facebook_post) if facebook_post else "Failed to generate Facebook post"

# =====================================================================
# 💬 WHATSAPP HOOK GENERATION
# =====================================================================

def generate_whatsapp_hook(linkedin_post, groq_llm):
    """Generate ultra-short WhatsApp teaser with link placeholder"""
    
    # If LinkedIn post is empty or too short, return placeholder
    if not linkedin_post or len(linkedin_post) < 100:
        return "Failed to generate WhatsApp hook: LinkedIn content too short"
    
    whatsapp_prompt = f"""
    Create a SUPER SHORT WhatsApp teaser from this LinkedIn post:
    
    KEY REQUIREMENTS:
    - 1-3 lines MAX (very concise)
    - Intriguing hook that makes people want to read more
    - Casual, conversational tone
    - End with: 🔗 Read full post: [LinkedIn URL]
    
    LINKEDIN POST:
    {linkedin_post[:1000]}
    
    Return ONLY the WhatsApp message text, nothing else.
    """
    
    with st.spinner("💬 Creating WhatsApp hook..."):
        whatsapp_hook = groq_llm.call(
            whatsapp_prompt,
            "You are a messaging expert who creates compelling, ultra-short teasers that drive clicks."
        )
    
    return clean_text(whatsapp_hook) if whatsapp_hook else "Failed to generate WhatsApp hook"

# =====================================================================
# 💼 ENHANCED LINKEDIN POST GENERATION WITH VERIFICATION LOOP
# =====================================================================

def clean_text(text: str) -> str:
    """Trim whitespace and remove extraneous surrounding quotes/newlines."""
    if not text:
        return ""
    t = text.strip()
    if (t.startswith("```") and t.endswith("```")) or (t.startswith("`") and t.endswith("`")):
        t = t.strip("`").strip()
    return t

def char_count(text: str) -> int:
    return len(text)

def build_user_prompt(query, search_results, tone, date_str, TARGET, TOLERANCE):
    return textwrap.dedent(f"""
    Create an informative and engaging LinkedIn article-style post about: "{query}"
    
    TARGET LENGTH: {TARGET} characters (±{TOLERANCE})
    TONE: {tone}, informative, positive
    
    RESEARCH FINDINGS:
    {search_results}
    
    POST STRUCTURE (CRITICAL - FOLLOW EXACTLY):
    
    [HOOK] - Start with a surprising fact/statistic about the topic + relevant emoji
    
    [PERSONAL BRIDGE] - 1-2 sentences connecting you to the topic as an informed observer and an "AI in Agriculture" Enthusiast
    
    [MAIN TREND] - What's happening now in this space
    
    [KEY INNOVATIONS] - 3-4 semi-detailed bullet points with emojis highlighting specific developments
    • Use emoji bullets like 🤖 🛰 💧 📊
    • Focus on concrete, specific innovations
    
    [IMPACT DATA] - Share measurable results/statistics from research.
    ***CRITICAL: Use EVIDENCE-BASED language:***
    - If research shows specific outcomes (e.g., "increased yields by 20%"), present as fact
    - If research discusses potential benefits, use "can help", "may improve", "has potential"
    - Do not present potential benefits as established outcomes without specific evidence
    - Focus on what the research actually demonstrates, not logical extrapolations
    
    [FUTURE OUTLOOK] - What's coming next in this field
    
    [ENGAGEMENT QUESTION] - End with a thought-provoking question for readers
    
    [HASHTAGS] - 4-6 relevant, strategic hashtags and also include #9jaAI_Farmer
    
    CURRENT DATE: {date_str}
    
    CRITICAL FORMATTING RULES:
    - Keep paragraphs SHORT (1-2 lines maximum)
    - Use emojis strategically (4-6 total in the post)
    - Focus on SOLUTIONS and POSITIVE developments
    - Be INFORMATIVE but not overly technical
    - Position yourself as a KNOWLEDGEABLE CURATOR, not claiming deep expertise
    - Include specific data points and statistics where available in ways that can be easily understood by the audience 
    - Make it MOBILE-FRIENDLY and easy to scan
    
    CRITICAL RULES ABOUT DATA AND STATISTICS:
    - **PRECISION & ATTRIBUTION:** Only use statistics, numbers, and specific claims that are explicitly stated in the RESEARCH FINDINGS.
    - **NO HALLUCINATION:** DO NOT invent or assume numerical data. If a specific number isn't in the research, describe the benefit qualitatively (e.g., "can significantly improve yields").
    - **GEOGRAPHIC SCOPE:** If data refers to Africa or is global, do not present it as Nigeria-specific. Always attribute the correct scope (e.g., "A project in Kenya..." or "Globally, AI can...").
    - **HANDLING ABSENCE:** If no Nigeria-specific statistic is found, it is acceptable to state the broader trend and connect it to Nigeria's potential. Never convert regional data into local numbers.
    - **EVIDENCE-BASED CLAIMS:** Only present benefits as "established facts" if the research explicitly states they are currently happening. Otherwise, use potential language like "can help", "may improve", "has the potential to".
    - **NO MARKET ACCESS ASSUMPTIONS:** Do not claim AI creates market access opportunities unless the research specifically mentions market connections, buyer linkages, or similar evidence.
    
    Return ONLY the post text, no explanations or additional text.
    """).strip()

def build_system_message():
    return textwrap.dedent("""
    You are an "Educator-Innovator" - a knowledgeable content creator who specializes in making complex AI and technology topics easy to understand, accessible and engaging for LinkedIn audiences.
    
    Your style is:
    - Informative but conversational
    - Data-driven but not academic
    - Solution-focused and positive
    - Curator of valuable insights
    - Bridge between innovations and audience understanding
    
    You excel at:
    - Creating compelling hooks with surprising facts
    - Structuring content for maximum readability
    - Using emojis and formatting for visual appeal
    - Driving engagement through thoughtful questions
    - Maintaining credibility without claiming deep expertise
    - **Evidence-Based Reporting:** You distinguish between demonstrated outcomes and potential benefits, never presenting logical conclusions as established facts without research evidence.
    
    CRITICAL: You are committed to factual accuracy and never invent statistics or misrepresent geographic scope.
    """).strip()

def optimize_post_length(post_text, target_length, groq_llm):
    """Optimize post length while preserving the enhanced structure"""
    current_length = len(post_text)
    diff = target_length - current_length
    
    if abs(diff) <= 250:
        return post_text
    
    if diff > 0:
        optimization_prompt = f"""
        This LinkedIn post is {current_length} characters. Please expand it by approximately {diff} characters while MAINTAINING THE EXACT STRUCTURE.
        
        Focus on adding:
        - Additional context about the future outlook
        - More detail in the key innovations section
        - Qualitative descriptions of benefits
        
        CRITICAL: 
        - Maintain the same structure, tone, and formatting with emoji bullets.
        - DO NOT add new statistics or numerical claims. Only elaborate using qualitative descriptions.
        - **MAINTAIN EVIDENCE-BASED LANGUAGE:** Keep "can help", "may improve", "has potential" language for benefits not explicitly demonstrated in research.
        
        POST TO EXPAND:
        {post_text}
        
        Return only the expanded post.
        """
    else:
        optimization_prompt = f"""
        This LinkedIn post is {current_length} characters. Please shorten it by approximately {abs(diff)} characters while PRESERVING ALL KEY SECTIONS.
        
        Focus on:
        - Making sentences more concise without losing meaning
        - Removing any redundant phrases
        - Keeping all emoji bullets and key data points
        
        CRITICAL: Maintain the hook, personal bridge, key innovations with emojis, impact data, future outlook, engagement question, and hashtags.
        
        POST TO SHORTEN:
        {post_text}
        
        Return only the shortened post.
        """
    
    optimized_post = groq_llm.call(
        optimization_prompt, 
        "You are a skilled editor who specializes in optimizing LinkedIn content length while preserving engagement-driven structure and formatting. You never invent statistics and maintain evidence-based language."
    )
    return clean_text(optimized_post) if optimized_post else post_text

def execute_linkedin_workflow(query, groq_llm, max_results, serper_key, tone, threshold, max_iterations):
    """Execute the enhanced LinkedIn post generation workflow with verification loop"""
    
    # Step 1: Perform web search with Serper API
    with st.status("🔍 Researching your topic...", expanded=True) as status:
        search_data = serper_search(query, max_results, serper_key)
        
        if search_data["formatted_results"].startswith("❌"):
            st.error("Search failed. Please check your Serper API key and try again.")
            return None
            
        status.update(label="✅ Research completed", state="complete")
    
    # Step 2: Generate initial LinkedIn post
    with st.status("✍️ Crafting your LinkedIn post...", expanded=True) as status:
        
        TARGET = 2800
        TOLERANCE = 300
        
        date_str = datetime.now().strftime("%Y-%m-%d")
        user_prompt = build_user_prompt(query, search_data["formatted_results"], tone, date_str, TARGET, TOLERANCE)
        system_msg = build_system_message()

        # Generate initial post
        initial_linkedin_post = groq_llm.call(user_prompt, system_msg)
        initial_linkedin_post = clean_text(initial_linkedin_post)
        
        if not initial_linkedin_post:
            status.update(label="❌ Failed to generate post", state="error")
            return None
        
        char_len = char_count(initial_linkedin_post)
        st.info(f"📊 Initial draft: {char_len} characters")
        
        # One optimization pass if needed
        if not (abs(char_len - TARGET) <= TOLERANCE):
            st.info("🔄 Optimizing length...")
            initial_linkedin_post = optimize_post_length(initial_linkedin_post, TARGET, groq_llm)
            char_len = char_count(initial_linkedin_post)
            st.info(f"📊 Optimized: {char_len} characters")
        
        status.update(label=f"✅ Initial post generated ({char_len} chars)", state="complete")
    
    # Step 3: Execute verification loop
    with st.status("🔍 Verifying factual accuracy...", expanded=True) as status:
        st.info(f"Starting verification with {threshold}% threshold (max {max_iterations} iterations)")
        
        final_linkedin_post, verification_history = execute_verification_loop(
            initial_linkedin_post,
            search_data,
            groq_llm,
            tone,
            threshold,
            max_iterations
        )
        
        # Show verification summary
        final_alignment = verification_history[-1].get("alignment_percentage", 0)
        iterations_used = len(verification_history)
        verification_method = verification_history[-1].get("verification_notes", "LLM Verification")
        
        if verification_history[-1].get("status") == "APPROVED":
            status.update(label=f"✅ Verification passed! {final_alignment:.1f}% alignment after {iterations_used} iteration(s)", 
                         state="complete")
        elif final_alignment > 70:
            status.update(label=f"⚠️ Good alignment: {final_alignment:.1f}% after {iterations_used} iteration(s)", 
                         state="complete")
        elif final_alignment > 0:
            status.update(label=f"⚠️ Moderate alignment: {final_alignment:.1f}% after {iterations_used} iteration(s)", 
                         state="complete")
        else:
            status.update(label=f"❌ Verification failed after {iterations_used} iteration(s)", 
                         state="error")
    
    # Step 4: Generate Facebook post (updated to reflect verified LinkedIn post)
    facebook_post = "Facebook post generation paused due to rate limits."
    if groq_llm and final_linkedin_post and len(final_linkedin_post) > 100:
        facebook_post = generate_facebook_post(final_linkedin_post, groq_llm)
    
    # Step 5: Generate WhatsApp hook
    whatsapp_hook = "WhatsApp hook generation paused due to rate limits."
    if groq_llm and final_linkedin_post and len(final_linkedin_post) > 100:
        whatsapp_hook = generate_whatsapp_hook(final_linkedin_post, groq_llm)
    
    return {
        "linkedin_post": final_linkedin_post,
        "character_count": char_count(final_linkedin_post),
        "search_results": search_data["formatted_results"],
        "search_urls": search_data["urls"],
        "facebook_post": facebook_post,
        "whatsapp_hook": whatsapp_hook,
        "verification_history": verification_history,
        "initial_linkedin_post": initial_linkedin_post
    }

# =====================================================================
# 📱 ENHANCED MAIN EXECUTION WITH VERIFICATION SYSTEM
# =====================================================================

def main():
    """Main application logic."""
    
    # Topic input with better UX
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
    
    # Get API keys
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
        st.info("💡 **Get your free API key:** https://serper.dev - and Get 2,500 free searches")
        return
    
    # Initialize Groq LLM
    groq_llm = GroqLLM(
        api_key=final_groq_key,
        model=model_options[selected_model],
        temperature=temperature
    )
    
    # Execute content generation
    if generate_clicked:
        if not query.strip():
            st.warning("⚠️ Please enter a topic for your content.")
        else:
            try:
                # Show generation progress
                with st.spinner("🎯 Creating your multi-platform content..."):
                    time.sleep(1)
                    
                # Execute workflow with verification
                result = execute_linkedin_workflow(
                    query, groq_llm, max_results, final_serper_key, selected_tone, verification_threshold, max_iterations
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
                    
                    # Create tabs for different platform outputs
                    tab_names = ["💼 LinkedIn", "📱 Facebook", "💬 WhatsApp", "🔍 Research URLs", "⚙️ Details", "🔍 Verification History"]
                    tabs = st.tabs(tab_names)
                    
                    with tabs[0]:  # LinkedIn
                        st.subheader("💼 Your Verified LinkedIn Post")
                        final_history = result["verification_history"][-1]
                        approval_status = final_history.get("status", "UNKNOWN")
                        alignment_pct = final_history.get("alignment_percentage", 0)
                        verification_method = final_history.get("verification_notes", "LLM Verification")
                        
                        if approval_status == "APPROVED":
                            st.markdown(f'<div class="ready-status">✅ APPROVED - {alignment_pct:.1f}% factual alignment</div>', 
                                       unsafe_allow_html=True)
                        elif alignment_pct > 70:
                            st.markdown(f'<div class="verification-pending">⚠️ GOOD - {alignment_pct:.1f}% alignment ({verification_method})</div>', 
                                       unsafe_allow_html=True)
                        elif alignment_pct > 0:
                            st.markdown(f'<div class="verification-warning">⚠️ MODERATE - {alignment_pct:.1f}% alignment ({verification_method})</div>', 
                                       unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="verification-failed">❌ VERIFICATION FAILED - Manual review recommended</div>', 
                                       unsafe_allow_html=True)
                        
                        st.text_area(
                            "Copy your LinkedIn post below:",
                            value=result["linkedin_post"],
                            height=400,
                            key="linkedin_post_output"
                        )
                    
                    with tabs[1]:  # Facebook
                        st.subheader("📱 Facebook Post")
                        st.info("🎭 **Facebook-Optimized Version** - Conversational & Engaging")
                        st.markdown('<div class="ready-status">✅ Facebook post ready for copy</div>', unsafe_allow_html=True)
                        st.text_area(
                            "Copy your Facebook post below:",
                            value=result["facebook_post"],
                            height=300,
                            key="facebook_post_output"
                        )
                    
                    with tabs[2]:  # WhatsApp
                        st.subheader("💬 WhatsApp Hook")
                        st.info("🎣 **Ultra-Short Teaser**")
                        st.markdown('<div class="ready-status">✅ WhatsApp post ready for copy</div>', unsafe_allow_html=True)
                        st.text_area(
                            "Copy your WhatsApp message below:",
                            value=result["whatsapp_hook"],
                            height=150,
                            key="whatsapp_hook_output"
                        )
                    
                    with tabs[3]:  # Research URLs
                        st.subheader("🔍 Research URLs")
                        st.info("📚 **All Search URLs for Quick Reuse**")
                        
                        if result["search_urls"]:
                            st.markdown(f"**Found {len(result['search_urls'])} URLs:**")
                            st.markdown('<div class="ready-status">✅ URLs ready for copy</div>', unsafe_allow_html=True)
                            
                            url_text = "\n".join(result["search_urls"])
                            st.text_area(
                                "All search URLs:",
                                value=url_text,
                                height=200,
                                key="urls_output"
                            )
                            
                            st.markdown("**Individual URLs:**")
                            for i, url in enumerate(result["search_urls"], 1):
                                st.markdown(f"`{i}. {url}`")
                        else:
                            st.warning("No URLs found in the search results.")
                    
                    with tabs[4]:  # Details
                        st.subheader("⚙️ Generation Details")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.write("**🤖 Model:**", selected_model)
                        with col2:
                            st.write("**🎭 Tone:**", selected_tone)
                        with col3:
                            st.write("**🔍 Sources:**", max_results)
                        with col4:
                            st.write("**📅 Date:**", datetime.now().strftime('%Y-%m-%d'))
                        
                        with st.expander("🔍 Full Research Results", expanded=False):
                            st.markdown(result["search_results"])
                        
                        with st.expander("📊 Verification Summary", expanded=True):
                            history = result["verification_history"]
                            iterations = len(history)
                            final_alignment = history[-1].get("alignment_percentage", 0)
                            final_status = history[-1].get("status", "UNKNOWN")
                            verification_method = history[-1].get("verification_notes", "LLM Verification")
                            
                            st.write(f"**Total Iterations:** {iterations}")
                            st.write(f"**Final Alignment:** {final_alignment:.1f}%")
                            st.write(f"**Final Status:** {final_status}")
                            st.write(f"**Verification Method:** {verification_method}")
                            st.write(f"**Threshold:** {verification_threshold}%")
                    
                    with tabs[5]:  # Verification History
                        st.subheader("🔍 Verification History")
                        st.info("📈 **Detailed verification progress across iterations**")
                        
                        history = result["verification_history"]
                        
                        for i, verification in enumerate(history, 1):
                            with st.expander(f"Iteration {i}: {verification.get('status', 'UNKNOWN')} - {verification.get('alignment_percentage', 0):.1f}%", 
                                           expanded=i == len(history)):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**Alignment:** {verification.get('alignment_percentage', 0):.1f}%")
                                    st.write(f"**Status:** {verification.get('status', 'UNKNOWN')}")
                                with col2:
                                    st.write(f"**Iteration:** {verification.get('iteration', i)}")
                                    st.write(f"**Method:** {verification.get('verification_notes', 'LLM Verification')}")
                                
                                st.write("**Feedback:**")
                                st.write(verification.get('feedback', 'No feedback available'))
                                
                                if verification.get('unsupported_claims'):
                                    st.write("**Unsupported Claims Identified:**")
                                    for claim in verification.get('unsupported_claims', []):
                                        st.write(f"- {claim}")
                        
                        # Show iteration comparison chart
                        if len(history) > 1:
                            st.subheader("📊 Alignment Progress")
                            import pandas as pd
                            df = pd.DataFrame([
                                {
                                    "Iteration": f"Iter {i}", 
                                    "Alignment": h.get("alignment_percentage", 0),
                                    "Status": h.get("status", "UNKNOWN")
                                }
                                for i, h in enumerate(history, 1)
                            ])
                            st.bar_chart(df.set_index("Iteration")["Alignment"])
                    
            except Exception as e:
                st.error(f"❌ Error during content generation: {str(e)}")
    
    # Information sections
    col1, col2 = st.columns([1, 1])
    
    with col1:
        with st.expander("📱 Mobile Tips", expanded=True):
            st.markdown("""
            **Perfect for Mobile:**
            - 🎯 Multi-platform content optimized for each platform
            - 📱 Easy copy-paste functionality  
            - ⚡ Fast generation on any device
            - 🎨 Mobile-responsive design
            """)
    
    with col2:
        with st.expander("🎯 Best Practices", expanded=True):
            st.markdown("""
            **For Viral Content:**
            - LinkedIn: Professional, detailed stories
            - Facebook: Friendly, conversational tone
            - WhatsApp: Short, intriguing hooks
            - Use platform-appropriate hashtags
            - Post during peak hours for each platform
            """)

# =====================================================================
# 🎯 FOOTER WITH ENHANCED UX
# =====================================================================

st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns([2, 1, 1])
with footer_col1:
    st.caption("💼 **Multi-Platform Content Generator Pro** | Create viral content for LinkedIn, Facebook & WhatsApp")
with footer_col2:
    st.caption("⚡ Powered by 9jaAI_Farmer")
with footer_col3:
    st.caption("🔍 Enhanced by Groq & Serper API")

# Add some spacing for mobile
st.markdown("<br>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
