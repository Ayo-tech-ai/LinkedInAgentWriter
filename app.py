import os
import streamlit as st
import requests
import json
from datetime import datetime
import time
import textwrap

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
                            st.error(f"Error details: {error_data.get('error', {}).get('message', 'Unknown error')}")
                        except:
                            st.error(f"Response: {e.response.text}")
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
# 📱 NEW: FACEBOOK POST GENERATION
# =====================================================================

def generate_facebook_post(linkedin_post, groq_llm):
    """Generate Facebook version from LinkedIn post - shorter, conversational, friendly"""
    
    facebook_prompt = f"""
    Transform this LinkedIn post into an engaging Facebook post:
    
    KEY REQUIREMENTS:
    - SHORTER: 200-400 characters max
    - CONVERSATIONAL & FRIENDLY tone
    - Use emojis naturally
    - Keep the core message but make it more personal
    - End with a question to encourage engagement
    - Include 3-5 relevant hashtags
    
    LINKEDIN POST:
    {linkedin_post}
    
    Return ONLY the Facebook post text, nothing else.
    """
    
    with st.spinner("🎯 Creating Facebook post..."):
        facebook_post = groq_llm.call(
            facebook_prompt,
            "You are a social media expert who specializes in adapting professional content for Facebook's friendly, conversational audience."
        )
    
    return clean_text(facebook_post) if facebook_post else "Failed to generate Facebook post"

# =====================================================================
# 💬 NEW: WHATSAPP HOOK GENERATION
# =====================================================================

def generate_whatsapp_hook(linkedin_post, groq_llm):
    """Generate ultra-short WhatsApp teaser with link placeholder"""
    
    whatsapp_prompt = f"""
    Create a SUPER SHORT WhatsApp teaser from this LinkedIn post:
    
    KEY REQUIREMENTS:
    - 1-3 lines MAX (very concise)
    - Intriguing hook that makes people want to read more
    - Casual, conversational tone
    - End with: 🔗 Read full post: [LinkedIn URL]
    
    LINKEDIN POST:
    {linkedin_post}
    
    Return ONLY the WhatsApp message text, nothing else.
    """
    
    with st.spinner("💬 Creating WhatsApp hook..."):
        whatsapp_hook = groq_llm.call(
            whatsapp_prompt,
            "You are a messaging expert who creates compelling, ultra-short teasers that drive clicks."
        )
    
    return clean_text(whatsapp_hook) if whatsapp_hook else "Failed to generate WhatsApp hook"

# =====================================================================
# 💼 ENHANCED LINKEDIN POST GENERATION WORKFLOW
# =====================================================================

def clean_text(text: str) -> str:
    """Trim whitespace and remove extraneous surrounding quotes/newlines."""
    if not text:
        return ""
    t = text.strip()
    # remove outer triple/back quotes if any
    if (t.startswith("```") and t.endswith("```")) or (t.startswith("`") and t.endswith("`")):
        # remove single/block backticks
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
    
    [PERSONAL BRIDGE] - 1-2 sentences connecting you to the topic as an informed observer
    
    [MAIN TREND] - What's happening now in this space
    
    [KEY INNOVATIONS] - 3-4 bullet points with emojis highlighting specific developments
    • Use emoji bullets like 🤖 🛰 💧 📊
    • Focus on concrete, specific innovations
    
    [IMPACT DATA] - Share measurable results/statistics from research
    
    [FUTURE OUTLOOK] - What's coming next in this field
    
    [ENGAGEMENT QUESTION] - End with a thought-provoking question for readers
    
    [HASHTAGS] - 8-12 relevant, strategic hashtags
    
    CURRENT DATE: {date_str}
    
    CRITICAL FORMATTING RULES:
    - Keep paragraphs SHORT (1-3 lines maximum)
    - Use emojis strategically (4-6 total in the post)
    - Focus on SOLUTIONS and POSITIVE developments
    - Be INFORMATIVE but not overly technical
    - Position yourself as a KNOWLEDGEABLE CURATOR, not claiming deep expertise
    - Include specific data points and statistics where available
    - Make it MOBILE-FRIENDLY and easy to scan
    
    Return ONLY the post text, no explanations or additional text.
    """).strip()

def build_system_message():
    return textwrap.dedent("""
    You are an "Educator-Innovator" - a knowledgeable content creator who specializes in making complex AI and technology topics accessible and engaging for LinkedIn audiences.
    
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
    """).strip()

def optimize_post_length(post_text, target_length, groq_llm):
    """Optimize post length while preserving the enhanced structure"""
    current_length = len(post_text)
    diff = target_length - current_length
    
    if abs(diff) <= 250:  # Close enough
        return post_text
    
    if diff > 0:
        # Need to add content
        optimization_prompt = f"""
        This LinkedIn post is {current_length} characters. Please expand it by approximately {diff} characters while MAINTAINING THE EXACT STRUCTURE.
        
        Focus on adding:
        - One more specific data point or statistic
        - Additional context about the future outlook
        - More detail in the key innovations section
        
        CRITICAL: Maintain the same structure, tone, and formatting with emoji bullets.
        
        POST TO EXPAND:
        {post_text}
        
        Return only the expanded post.
        """
    else:
        # Need to shorten
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
        "You are a skilled editor who specializes in optimizing LinkedIn content length while preserving engagement-driven structure and formatting."
    )
    return clean_text(optimized_post) if optimized_post else post_text

def validate_post_structure(post_text):
    """Validate that the generated post follows our enhanced structure"""
    required_elements = [
        "🌱", "🚀", "🤖", "📊", "💧", "🛰"  # Common emojis we use
    ]
    
    # Check for emoji usage (at least 2)
    emoji_count = sum(1 for char in post_text if char in required_elements)
    
    # Check for engagement question (ends with ?)
    has_question = '?' in post_text[-100:]
    
    # Check for hashtags
    has_hashtags = '#' in post_text[-200:]
    
    return emoji_count >= 2 and has_question and has_hashtags

def execute_linkedin_workflow(query, groq_llm, max_results, serper_key, tone):
    """Execute the enhanced LinkedIn post generation workflow"""
    
    # Step 1: Perform web search with Serper API
    with st.status("🔍 Researching your topic...", expanded=True) as status:
        search_data = serper_search(query, max_results, serper_key)
        
        # Check if search was successful
        if search_data["formatted_results"].startswith("❌"):
            st.error("Search failed. Please check your Serper API key and try again.")
            return None
            
        status.update(label="✅ Research completed", state="complete")
    
    # Step 2: Enhanced LinkedIn post generation
    with st.status("✍️ Crafting your LinkedIn post...", expanded=True) as status:
        
        # Fixed length settings for optimal engagement
        TARGET = 2200  # Slightly shorter for better engagement
        TOLERANCE = 300
        
        date_str = datetime.now().strftime("%Y-%m-%d")
        user_prompt = build_user_prompt(query, search_data["formatted_results"], tone, date_str, TARGET, TOLERANCE)
        system_msg = build_system_message()

        # Single generation attempt first
        linkedin_post = groq_llm.call(user_prompt, system_msg)
        linkedin_post = clean_text(linkedin_post)
        
        if linkedin_post:
            char_len = char_count(linkedin_post)
            st.info(f"📊 First draft: {char_len} characters (target: {TARGET}±{TOLERANCE})")
            
            # Validate structure
            if validate_post_structure(linkedin_post):
                st.success("✅ Post structure validated")
            else:
                st.warning("⚠️ Post may need structural adjustments")
            
            # One optimization pass if needed
            if not (abs(char_len - TARGET) <= TOLERANCE):
                st.info("🔄 Optimizing length...")
                linkedin_post = optimize_post_length(linkedin_post, TARGET, groq_llm)
                char_len = char_count(linkedin_post)
                st.info(f"📊 Optimized: {char_len} characters")
            
            # Final structure check
            if validate_post_structure(linkedin_post):
                structure_status = "✅ Optimal structure"
            else:
                structure_status = "⚠️ Basic structure"
            
            if abs(char_len - TARGET) <= TOLERANCE:
                status.update(label=f"✅ LinkedIn post crafted ({char_len} chars, {structure_status})", state="complete")
            else:
                status.update(label=f"⚠️ Post generated ({char_len} chars, {structure_status})", state="complete")
                st.warning(f"Best effort: {char_len} characters (target: {TARGET}±{TOLERANCE})")
        else:
            status.update(label="❌ Failed to generate post", state="error")
            return None
    
    if not linkedin_post:
        return None
    
    # Step 3: Generate engagement tips (optional - can skip if rate limits are hit)
    engagement_tips = "Enable engagement tips in settings to get posting strategies."
    
    # Step 4: Generate Facebook post
    facebook_post = "Facebook post generation paused due to rate limits."
    if groq_llm:
        facebook_post = generate_facebook_post(linkedin_post, groq_llm)
    
    # Step 5: Generate WhatsApp hook
    whatsapp_hook = "WhatsApp hook generation paused due to rate limits."
    if groq_llm:
        whatsapp_hook = generate_whatsapp_hook(linkedin_post, groq_llm)
    
    return {
        "linkedin_post": linkedin_post,
        "character_count": char_count(linkedin_post),
        "engagement_tips": engagement_tips,
        "search_results": search_data["formatted_results"],
        "search_urls": search_data["urls"],
        "facebook_post": facebook_post,
        "whatsapp_hook": whatsapp_hook
    }
# =====================================================================
# 📱 ENHANCED MAIN EXECUTION WITH MULTI-PLATFORM OUTPUTS
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
        st.info("💡 **Get your free API key:** https://serper.dev - 2,500 free searches/month")
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
                    
                # Execute workflow
                result = execute_linkedin_workflow(
                    query, groq_llm, max_results, final_serper_key, selected_tone
                )
                
                if result is None:
                    st.error("❌ Content generation failed. Please check your API keys and try again.")
                else:
                    st.success("🎉 Your multi-platform content is ready!")
                    
                    # Character count display
                    char_count = result["character_count"]
                    st.markdown(f"""
                    <div class="char-counter char-warning">
                        <strong>Character Count:</strong> {char_count} characters
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create tabs for different platform outputs
                    tab1, tab2, tab3, tab4, tab5 = st.tabs([
                        "💼 LinkedIn", 
                        "📱 Facebook", 
                        "💬 WhatsApp", 
                        "🔍 Research URLs", 
                        "⚙️ Details"
                    ])
                    
                    with tab1:
                        st.subheader("💼 Your LinkedIn Post")
                        st.text_area(
                            "Copy your LinkedIn post below:",
                            value=result["linkedin_post"],
                            height=400,
                            key="linkedin_post_output"
                        )
                        
                        # Quick copy button
                        st.download_button(
                            label="📋 Copy LinkedIn Post",
                            data=result["linkedin_post"],
                            file_name=f"linkedin_post_{datetime.now().strftime('%Y%m%d')}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    
                    with tab2:
                        st.subheader("📱 Facebook Post")
                        st.info("🎭 **Friendly & Conversational Version**")
                        st.text_area(
                            "Copy your Facebook post below:",
                            value=result["facebook_post"],
                            height=200,
                            key="facebook_post_output"
                        )
                        
                        st.download_button(
                            label="📋 Copy Facebook Post",
                            data=result["facebook_post"],
                            file_name=f"facebook_post_{datetime.now().strftime('%Y%m%d')}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    
                    with tab3:
                        st.subheader("💬 WhatsApp Hook")
                        st.info("🎣 **Ultra-Short Teaser**")
                        st.text_area(
                            "Copy your WhatsApp message below:",
                            value=result["whatsapp_hook"],
                            height=150,
                            key="whatsapp_hook_output"
                        )
                        
                        st.download_button(
                            label="📋 Copy WhatsApp Hook",
                            data=result["whatsapp_hook"],
                            file_name=f"whatsapp_hook_{datetime.now().strftime('%Y%m%d')}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    
                    with tab4:
                        st.subheader("🔍 Research URLs")
                        st.info("📚 **All Search URLs for Quick Reuse**")
                        
                        if result["search_urls"]:
                            st.markdown(f"**Found {len(result['search_urls'])} URLs:**")
                            
                            # Display URLs in a clean, copy-friendly format
                            url_text = "\n".join(result["search_urls"])
                            st.text_area(
                                "All search URLs:",
                                value=url_text,
                                height=200,
                                key="urls_output"
                            )
                            
                            st.download_button(
                                label="📋 Copy All URLs",
                                data=url_text,
                                file_name=f"research_urls_{datetime.now().strftime('%Y%m%d')}.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                            
                            # Individual URL display with copy buttons
                            st.markdown("**Individual URLs:**")
                            for i, url in enumerate(result["search_urls"], 1):
                                col1, col2 = st.columns([4, 1])
                                with col1:
                                    st.markdown(f"`{i}. {url}`")
                                with col2:
                                    st.download_button(
                                        label="📋",
                                        data=url,
                                        file_name=f"url_{i}.txt",
                                        mime="text/plain",
                                        key=f"url_{i}"
                                    )
                        else:
                            st.warning("No URLs found in the search results.")
                    
                    with tab5:
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
    st.caption("⚡ Powered by Groq")
with footer_col3:
    st.caption("🔍 Enhanced by Serper API")

# Add some spacing for mobile
st.markdown("<br>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
