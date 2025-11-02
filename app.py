import os
import streamlit as st
import requests
import json
from datetime import datetime
import time

# =====================================================================
# 🎯 ENHANCED STREAMLIT CONFIGURATION
# =====================================================================

st.set_page_config(
    page_title="LinkedIn Post Generator Pro",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "# LinkedIn Post Generator\nCreate viral LinkedIn content with AI!"
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
</style>
""", unsafe_allow_html=True)

# =====================================================================
# 🚀 ENHANCED HEADER
# =====================================================================

st.markdown("""
<div class="custom-header">
    <h1 style="margin:0; color:white;">💼 LinkedIn Post Generator Pro</h1>
    <p style="margin:0; opacity:0.9; font-size:1.1em;">Create viral, SEO-optimized LinkedIn posts that drive engagement and build authority</p>
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
            help="Get free key from https://serper.dev - 2,500 searches/month free"
        )
    
    st.markdown("---")
    
    # Model selection with enhanced descriptions
    st.subheader("🤖 AI Model")
    model_options = {
        "Llama 3.3 70B Versatile": "llama-3.3-70b-versatile",
        "Llama 3.1 8B Instant": "llama-3.1-8b-instant", 
        "Llama 3.1 70B Versatile": "llama-3.1-70b-versatile",
        "Mixtral 8x7B": "mixtral-8x7b-32768",
        "Gemma2 9B": "gemma2-9b-it"
    }
    
    selected_model = st.selectbox(
        "Select AI Model",
        options=list(model_options.keys()),
        index=0,
        help="Llama 3.3 70B recommended for best quality"
    )
    
    # Content customization
    st.subheader("🎭 Content Style")
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("Creativity", 0.0, 1.0, 0.7, 0.1,
                              help="Higher = more creative, Lower = more factual")
    with col2:
        max_results = st.slider("Search Results", 1, 10, 5,
                              help="More results = more comprehensive research")
    
    # Tone selection
    tone_options = ["Professional", "Conversational", "Inspirational", "Humorous", "Authoritative"]
    selected_tone = st.selectbox("Post Tone", tone_options, index=1,
                               help="Choose the tone that matches your brand voice")
    
    st.markdown("---")
    st.info("💡 **Pro Tip**: Use specific, trending topics for best engagement!")

# =====================================================================
# 🔍 SERPER API SEARCH (UNCHANGED BUT OPTIMIZED)
# =====================================================================

def serper_search(query: str, max_results: int = 5, api_key: str = None) -> str:
    """Search using Serper API - our primary search method"""
    if not api_key:
        return "❌ Serper API key not provided"

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
        
        # Process organic results
        if 'organic' in data and data['organic']:
            for i, result in enumerate(data['organic'][:max_results], 1):
                title = result.get('title', 'No title')
                link = result.get('link', 'No URL')
                snippet = result.get('snippet', 'No description')
                results.append(f"### 📄 Result {i}: {title}\n**URL:** {link}\n**Summary:** {snippet}\n")
        
        # Also check for news results if organic results are limited
        if len(results) < max_results and 'news' in data and data['news']:
            news_to_add = max_results - len(results)
            for i, result in enumerate(data['news'][:news_to_add], len(results) + 1):
                title = result.get('title', 'No title')
                link = result.get('link', 'No URL')
                snippet = result.get('snippet', 'No description')
                results.append(f"### 📰 News {i}: {title}\n**URL:** {link}\n**Summary:** {snippet}\n")
        
        if results:
            st.success(f"✅ Found {len(results)} relevant sources")
            return "\n\n".join(results)
        else:
            st.warning("⚠️ No search results found for this topic")
            return "❌ No search results found"
            
    except requests.exceptions.RequestException as e:
        error_msg = f"Search failed: {str(e)}"
        st.error(f"❌ {error_msg}")
        return f"❌ {error_msg}"
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        st.error(f"❌ {error_msg}")
        return f"❌ {error_msg}"

# =====================================================================
# 🧠 GROQ LLM INTEGRATION (UNCHANGED)
# =====================================================================

class GroqLLM:
    """Custom Groq LLM wrapper"""

    def __init__(self, api_key, model, temperature=0.7):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        
    def call(self, prompt, system_message=None):
        """Make API call to Groq"""
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
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Groq API Request failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                st.error(f"Response: {e.response.text}")
            return None
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            return None

# =====================================================================
# 💼 LINKEDIN POST GENERATION WORKFLOW
# =====================================================================

def execute_linkedin_workflow(query, groq_llm, max_results, serper_key, tone):
    """Execute the complete LinkedIn post generation workflow using the user's writing style"""
    
    # Step 1: Perform web search with Serper API
    with st.status("🔍 Researching your topic...", expanded=True) as status:
        search_results = serper_search(query, max_results, serper_key)
        
        # Check if search was successful
        if search_results.startswith("❌"):
            st.error("Search failed. Please check your Serper API key and try again.")
            return None
            
        status.update(label="✅ Research completed", state="complete")
    
    # Step 2: Generate LinkedIn post in the user's style with exact character count
    with st.status("✍️ Crafting your LinkedIn post...", expanded=True) as status:
        linkedin_prompt = f"""
        CRITICAL: Create a LinkedIn post about "{query}" that is EXACTLY 3,000 CHARACTERS (not 2,999, not 3,001).

        WRITING STYLE TO MIMIC:
        - Starts with a bold statement and emoji
        - Uses hashtags within sentences naturally
        - Personal storytelling with emotional journey
        - Technical details made accessible
        - Progress narrative: "from X to Y"
        - Tool/feature showcases with emoji bullets
        - Honest about challenges and frustrations
        - Vision-oriented conclusion
        - Strategic hashtag clusters at the end

        SEARCH RESULTS:
        {search_results}

        Current Date: {datetime.now().strftime('%Y-%m-%d')}

        POST STRUCTURE (MUST FOLLOW):

        1. 🎯 HOOK & INTRODUCTION (2-3 paragraphs)
           - Bold opening statement with emoji
           - Personal connection to the topic
           - From X to Y journey framing

        2. 🚀 THE EVOLUTION STORY (4-5 paragraphs)  
           - Where it started (basic version)
           - Key realization moment
           - Decision to dream bigger
           - Current advanced state

        3. 🛠 FEATURES/TOOLS SHOWCASE (4-5 paragraphs)
           - Emoji bullet points for each feature
           - Technical capabilities made simple
           - Real-world impact focus
           - How features work together

        4. 💪 CHALLENGES & REWARDS (2-3 paragraphs)
           - Honest about difficulties
           - Specific technical challenges
           - Emotional payoff moments
           - "Worth it" conclusion

        5. 🌟 VISION & IMPACT (2-3 paragraphs)
           - Work-in-progress acknowledgment
           - Practical daily impact
           - Big vision statement
           - Call to community/engagement

        6. 🏷 STRATEGIC HASHTAGS (15-20 relevant hashtags)
           - Mix of technical and thematic tags
           - Community and program tags
           - Geographic and domain tags

        CHARACTER COUNT PROTOCOL:
        - Write complete post first
        - Count characters precisely  
        - If short: Add more personal anecdotes, specific examples, or feature details
        - If long: Remove repetitive phrases while keeping the story flow
        - Final output MUST be 3,000 characters ± 10

        TONE: {tone} - Personal, technical-but-accessible, visionary, honest about challenges
        """
        
        system_msg = f"""You are a ghostwriter who perfectly mimics the user's unique LinkedIn writing style. 
        You write with:
        - Personal storytelling with emotional depth
        - Technical concepts made accessible and exciting
        - Natural hashtag integration within sentences
        - "From X to Y" progress narratives
        - Honest vulnerability about challenges
        - Vision-driven conclusions
        - Emoji-enhanced bullet points for features
        - Strategic hashtag clusters

        You are OBSESSIVE about hitting 3,000 characters while maintaining the authentic voice and story flow.
        You understand this isn't just content - it's a personal journey story that inspires while educating.
        """
        
        linkedin_post = groq_llm.call(linkedin_prompt, system_msg)
        
        if linkedin_post:
            # Verify character count and provide multiple attempts if needed
            char_count = len(linkedin_post)
            attempts = 1
            max_attempts = 3
            
            while abs(char_count - 3000) > 20 and attempts < max_attempts:
                st.warning(f"🔄 Attempt {attempts}: Post is {char_count} characters. Adjusting to hit 3,000...")
                
                adjustment_prompt = f"""
                CURRENT POST ({char_count} characters):
                {linkedin_post}
                
                TARGET: 3,000 characters exactly
                DIFFERENCE: {3000 - char_count} characters
                
                ADJUSTMENT NEEDED: {'ADD more content' if char_count < 3000 else 'REMOVE excess content'}
                
                SPECIFIC INSTRUCTIONS:
                {'• Add more personal anecdotes or specific examples' if char_count < 3000 else '• Remove repetitive phrases while keeping core story'}
                {'• Expand on feature details or user impact stories' if char_count < 3000 else '• Simplify descriptions without losing emotional impact'}
                {'• Include more technical implementation challenges' if char_count < 3000 else '• Cut redundant explanations'}
                {'• Add vision statements or future roadmap' if char_count < 3000 else '• Maintain all key story elements'}
                
                CRITICAL: Keep the authentic writing style - personal, technical-but-accessible, emotional journey, feature showcases with emojis.
                """
                
                linkedin_post = groq_llm.call(adjustment_prompt, system_msg)
                if linkedin_post:
                    char_count = len(linkedin_post)
                    st.info(f"📊 Adjustment result: {char_count} characters")
                attempts += 1
            
            status.update(label=f"✅ LinkedIn post crafted ({char_count} chars)", state="complete")
        else:
            status.update(label="❌ Failed to generate post", state="error")
            return None
    
    if not linkedin_post:
        return None
    
    # Step 3: Generate engagement tips
    with st.status("🎯 Generating engagement tips...", expanded=True) as status:
        tips_prompt = f"""
        Based on this LinkedIn post written in my signature style, provide 3 specific engagement strategies:
        
        POST:
        {linkedin_post}
        
        CHARACTER COUNT: {len(linkedin_post)}
        
        Provide 3 actionable tips that match my authentic voice and content style:
        1. Best way to frame this post for maximum engagement
        2. How to encourage meaningful discussion in comments
        3. Cross-posting or community sharing strategies
        """
        
        engagement_tips = groq_llm.call(
            tips_prompt,
            "You are a LinkedIn growth expert who understands authentic personal branding and technical storytelling."
        )
        status.update(label="✅ Engagement tips ready", state="complete")
    
    return {
        "linkedin_post": linkedin_post,
        "character_count": len(linkedin_post),
        "engagement_tips": engagement_tips,
        "search_results": search_results
        }

# =====================================================================
# 📱 ENHANCED MAIN EXECUTION WITH MOBILE OPTIMIZATION
# =====================================================================

def main():
    """Main application logic."""
    
    # Topic input with better UX
    st.subheader("🎯 What do you want to post about?")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_area(
            "Enter your topic:", 
            placeholder="e.g., AI in marketing 2024, Remote work productivity tips, Sustainable business practices...",
            height=100,
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        generate_clicked = st.button("🚀 Generate Post", use_container_width=True, type="primary")
    
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
    
    # Execute LinkedIn post generation
    if generate_clicked:
        if not query.strip():
            st.warning("⚠️ Please enter a topic for your LinkedIn post.")
        else:
            try:
                # Show generation progress
                with st.spinner("🎯 Creating your viral LinkedIn post..."):
                    time.sleep(1)  # Better UX feel
                    
                # Execute LinkedIn workflow
                result = execute_linkedin_workflow(
                    query, groq_llm, max_results, final_serper_key, selected_tone
                )
                
                if result is None:
                    st.error("❌ Post generation failed. Please check your API keys and try again.")
                else:
                    st.success("🎉 Your LinkedIn post is ready!")
                    
                    # Character count display with visual feedback
                    char_count = result["character_count"]
                    if char_count == 3000:
                        char_class = "char-perfect"
                        message = "🎯 Perfect! Exactly 3,000 characters"
                    elif 2990 <= char_count <= 3010:
                        char_class = "char-warning"
                        message = f"⚠️ Close! {char_count} characters (target: 3,000)"
                    else:
                        char_class = "char-danger"
                        message = f"❌ Off target: {char_count} characters (target: 3,000)"
                    
                    st.markdown(f"""
                    <div class="char-counter {char_class}">
                        <strong>Character Count:</strong> {message}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display results in expandable sections
                    with st.expander("💼 Your LinkedIn Post", expanded=True):
                        st.text_area(
                            "Copy your post below:",
                            value=result["linkedin_post"],
                            height=400,
                            key="linkedin_post_output"
                        )
                        
                        # Quick copy button
                        st.download_button(
                            label="📋 Copy to Clipboard",
                            data=result["linkedin_post"],
                            file_name=f"linkedin_post_{datetime.now().strftime('%Y%m%d')}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    
                    with st.expander("🎯 Engagement Tips", expanded=False):
                        st.markdown(result["engagement_tips"])
                    
                    with st.expander("🔍 Research Sources", expanded=False):
                        st.markdown(result["search_results"])
                    
                    with st.expander("⚙️ Generation Details", expanded=False):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.write("**🤖 Model:**", selected_model)
                        with col2:
                            st.write("**🎭 Tone:**", selected_tone)
                        with col3:
                            st.write("**🔍 Sources:**", max_results)
                        with col4:
                            st.write("**📅 Date:**", datetime.now().strftime('%Y-%m-%d'))
                    
            except Exception as e:
                st.error(f"❌ Error during post generation: {str(e)}")
    
    # Information sections with better mobile layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        with st.expander("📱 Mobile Tips", expanded=True):
            st.markdown("""
            **Perfect for Mobile:**
            - 🎯 Posts optimized for mobile reading
            - 📱 Easy copy-paste functionality  
            - ⚡ Fast generation on any device
            - 🎨 Mobile-responsive design
            """)
    
    with col2:
        with st.expander("🎯 Best Practices", expanded=True):
            st.markdown("""
            **For Viral Posts:**
            - Use specific, trending topics
            - Include personal stories
            - Ask engaging questions
            - Post during peak hours (9-11 AM)
            - Use 3-5 relevant hashtags
            """)
    
    # API Key Help Section
    with st.expander("🔑 API Setup Guide", expanded=False):
        st.markdown("""
        **Quick Setup:**
        
        1. **Serper API** (Free)
           - Visit: https://serper.dev
           - Sign up & get 2,500 free searches/month
           - Perfect for reliable web search
        
        2. **Groq API** (Free)  
           - Visit: https://console.groq.com
           - Sign up & get generous free limits
           - Blazing fast AI processing
        
        **Mobile Friendly:** Works perfectly on all devices!
        """)

# =====================================================================
# 🎯 FOOTER WITH ENHANCED UX
# =====================================================================

st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns([2, 1, 1])
with footer_col1:
    st.caption("💼 **LinkedIn Post Generator Pro** | Create viral content that builds authority and drives engagement")
with footer_col2:
    st.caption("⚡ Powered by Groq")
with footer_col3:
    st.caption("🔍 Enhanced by Serper API")

# Add some spacing for mobile
st.markdown("<br>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
