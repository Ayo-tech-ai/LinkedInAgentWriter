import os
import streamlit as st
import requests
import json
from datetime import datetime

# =====================================================================
# 🚀 STREAMLIT AGENTIC AGENTIC RESEARCH APP (GROQ + SERPER API)
# Updated: Generates LinkedIn post (exactly 3,000 characters) and improved UI/UX
# =====================================================================

st.set_page_config(
    page_title="Groq LinkedIn Agentic Writer",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔗 Groq-Powered LinkedIn Writer")
st.markdown(
    """
    🤖 **Powered by Groq + Serper API**  
    _Research the web and produce a high-performing LinkedIn post (exactly **3,000 characters**)._ 
    """
)

# -------------------- Sidebar: Configuration --------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    groq_api_key = st.text_input(
        "Enter your Groq API Key", type="password", placeholder="gsk_... or leave blank to use GROQ_API_KEY env var"
    )

    serper_api_key = st.text_input(
        "Serper API Key (Required)", type="password", placeholder="Enter your Serper API key"
    )

    model_options = {
        "Llama 3.3 70B Versatile": "llama-3.3-70b-versatile",
        "Llama 3.1 8B Instant": "llama-3.1-8b-instant",
        "Llama 3.1 70B Versatile": "llama-3.1-70b-versatile",
        "Mixtral 8x7B": "mixtral-8x7b-32768",
        "Gemma2 9B": "gemma2-9b-it",
    }
    selected_model = st.selectbox("Select Model", options=list(model_options.keys()), index=0)

    temperature = st.slider("Creativity", 0.0, 1.0, 0.65, 0.05)
    max_results = st.slider("Max Search Results", 1, 10, 5)

    st.markdown("---")
    st.caption("Tip: Lower temperature for more factual posts; higher for creative/humorous tone.")

# -------------------- Utility: Serper Search --------------------

def serper_search(query: str, max_results: int = 5, api_key: str = None) -> str:
    if not api_key:
        return "❌ Serper API key not provided"

    try:
        st.info(f"🔎 Searching the web for: {query}")
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query, "num": max_results, "gl": "us", "hl": "en"})
        headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

        with st.spinner("Searching the web..."):
            response = requests.post(url, headers=headers, data=payload, timeout=30)
            response.raise_for_status()

        data = response.json()
        results = []

        if "organic" in data and data["organic"]:
            for i, result in enumerate(data["organic"][:max_results], 1):
                title = result.get("title", "No title")
                link = result.get("link", "No URL")
                snippet = result.get("snippet", "No description")
                results.append(f"### 📄 Result {i}: {title}\n**URL:** {link}\n**Summary:** {snippet}\n")
                st.write(f"✅ Found: {title[:80]}...")

        if len(results) < max_results and "news" in data and data["news"]:
            news_to_add = max_results - len(results)
            for i, result in enumerate(data["news"][:news_to_add], len(results) + 1):
                title = result.get("title", "No title")
                link = result.get("link", "No URL")
                snippet = result.get("snippet", "No description")
                results.append(f"### 📰 News {i}: {title}\n**URL:** {link}\n**Summary:** {snippet}\n")
                st.write(f"✅ Found news: {title[:80]}...")

        if results:
            st.success(f"✅ Serper API found {len(results)} results")
            return "\n\n".join(results)
        else:
            st.warning("⚠️ No search results found via Serper API")
            return "❌ No search results found"

    except requests.exceptions.RequestException as e:
        error_msg = f"Serper API request failed: {str(e)}"
        st.error(f"❌ {error_msg}")
        return f"❌ {error_msg}"
    except Exception as e:
        error_msg = f"Unexpected error with Serper API: {str(e)}"
        st.error(f"❌ {error_msg}")
        return f"❌ {error_msg}"

# -------------------- Groq LLM Wrapper --------------------
class GroqLLM:
    def __init__(self, api_key, model, temperature=0.7):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"

    def call(self, prompt, system_message=None):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
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
            "stream": False,
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Groq API Request failed: {str(e)}")
            if hasattr(e, "response") and e.response is not None:
                st.error(f"Response: {e.response.text}")
            return None
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            return None

# -------------------- Research -> LinkedIn Post Workflow --------------------

def execute_linkedin_workflow(query, groq_llm, max_results, serper_key, tone, audience, humor_level, include_hashtags):
    # Step 1: Search
    with st.spinner("🔍 Searching the web with Serper API..."):
        search_results = serper_search(query, max_results, serper_key)
        if search_results.startswith("❌"):
            st.error("Search failed. Please check your Serper API key and try again.")
            return None

    # Step 2: Build prompt for LinkedIn post
    current_date = datetime.now().strftime("%Y-%m-%d")

    system_msg = (
        "You are a high-performing LinkedIn content strategist and writer. Create posts that drive engagement, "
        "establish trust, and optimize for SEO on LinkedIn. Use clear hooks, short paragraphs, and a strong CTA."
    )

    linkedin_prompt = f"""
    Using the SEARCH RESULTS below, write a LinkedIn post about: "{query}".

    SEARCH RESULTS:
    {search_results}

    REQUIREMENTS:
    1) Tone: {tone}.
    2) Target audience: {audience}.
    3) Humor level: {humor_level} (use light humor if asked).
    4) Include a short engaging hook (1-2 lines) at the top.
    5) Keep paragraphs short (1-2 sentences each), use bullet-like lines sparingly.
    6) Use emojis sparingly to increase readability if appropriate.
    7) Include 3 relevant hashtags at the end if requested.
    8) End with a call-to-action (question or action).

    IMPORTANT — CHARACTER LIMIT:
    - The post MUST be exactly 3,000 characters in total (this includes every letter, punctuation, newline, and space).
    - If the first output is not exactly 3,000 characters, adjust the content to meet the 3,000 character requirement precisely.

    CONTEXT: Current date is {current_date}. Base the post strictly on the provided search results. Do not invent facts.

    Output ONLY the final LinkedIn post (no explanations, no extra sections)."""

    # Try generation and allow a few automated refinement attempts to reach exactly 3000 characters
    attempts = 0
    max_attempts = 3
    final_post = None

    while attempts < max_attempts:
        attempts += 1
        with st.spinner(f"🧠 Generating LinkedIn post (attempt {attempts}/{max_attempts})..."):
            candidate = groq_llm.call(linkedin_prompt, system_msg)

        if not candidate:
            st.error("❌ Groq LLM did not return any content.")
            return None

        # Normalize and count characters exactly
        char_count = len(candidate)
        st.write(f"Character count returned: {char_count}")

        if char_count == 3000:
            final_post = candidate
            st.success("✅ Generated post meets 3,000 character requirement.")
            break
        else:
            st.warning(f"⚠️ Generated post is {char_count} characters (not 3000). Attempting automatic adjustment...")
            # Build an adjustment prompt asking the model to strictly modify to 3000 chars
            adjustment_prompt = (
                f"The post below must be adjusted to be exactly 3,000 characters. Change wording, trim or expand sentences, "
                f"but keep meaning, tone ({tone}), humor level ({humor_level}), and target audience ({audience}).\n\nPOST:\n{candidate}"
            )
            linkedin_prompt = (
                """You are a LinkedIn content editor. Adjust the provided post so it is exactly 3,000 characters. """
                + "\n\n"
                + adjustment_prompt
            )
            # loop will call LLM again with updated linkedin_prompt

    if not final_post:
        st.error("❌ Failed to produce a post of exactly 3,000 characters after multiple attempts.")
        # Provide last candidate as fallback
        final_post = candidate

    # Optionally add hashtags (user may want auto-generated ones saved separately)
    hashtags = ""
    if include_hashtags:
        # simple heuristic: allow the model to propose 3 hashtags appended if not already present
        hashtag_prompt = (
            f"Based on the LinkedIn post below, provide exactly 3 short, relevant hashtags (without explanation).\n\nPOST:\n{final_post}"
        )
        tags = groq_llm.call(hashtag_prompt, "You are a hashtag generator for LinkedIn posts.")
        if tags:
            # Clean tags
            tags_clean = tags.strip().replace("\n", " ")
            hashtags = "\n\n" + tags_clean

    return {"post": final_post, "hashtags": hashtags, "search_results": search_results}

# -------------------- Main App UI --------------------

def main():
    query = st.text_area("🔎 Enter your research topic:", placeholder="e.g. Machine learning in Agriculture 2024", height=120)

    # extra UX inputs
    st.subheader("✍️ LinkedIn Post Settings")
    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        tone = st.selectbox("Tone:", ["Professional", "Friendly", "Authoritative", "Curious"], index=0)
    with col2:
        audience = st.text_input("Target audience (comma-separated)", value="Researchers, practitioners, decision-makers")
    with col3:
        humor_level = st.selectbox("Humor:", ["None", "Light", "Medium"], index=1)

    include_hashtags = st.checkbox("Auto-generate 3 hashtags", value=True)

    final_groq_key = groq_api_key.strip() if groq_api_key and groq_api_key.strip() else os.getenv("GROQ_API_KEY")
    final_serper_key = serper_api_key.strip() if serper_api_key and serper_api_key.strip() else os.getenv("SERPER_API_KEY")

    if not final_groq_key:
        st.error("❌ No Groq API key provided. Please enter your API key or set GROQ_API_KEY environment variable.")
        st.stop()

    if not final_serper_key:
        st.error("❌ No Serper API key provided. Please enter your Serper API key.")
        st.stop()

    groq_llm = GroqLLM(api_key=final_groq_key, model=model_options[selected_model], temperature=temperature)

    run_col, preview_col = st.columns([1, 2])
    with run_col:
        if st.button("🚀 Generate LinkedIn Post", use_container_width=True, type="primary"):
            if not query.strip():
                st.warning("⚠️ Please enter a research topic.")
            else:
                with st.spinner("Running research and generating LinkedIn post..."):
                    result = execute_linkedin_workflow(query, groq_llm, max_results, final_serper_key, tone, audience, humor_level, include_hashtags)

                if result is None:
                    st.error("❌ Failed to generate post. See errors above.")
                else:
                    st.success("✅ Post generated — review below.")
                    post = result["post"]
                    hashtags = result["hashtags"]

                    # Preview area
                    with preview_col:
                        st.subheader("Post Preview")
                        st.text_area("LinkedIn Post (final)", value=post + (hashtags or ""), height=420)
                        st.write(f"🔢 Character count: {len(post)}")

                        # Allow download
                        st.download_button("📥 Download Post (.txt)", post + (hashtags or ""), file_name="linkedin_post.txt")

                        # Quick copy: show as code block for easy select
                        st.subheader("Copy / Publish")
                        st.code(post + (hashtags or ""))

                    # Show search results (collapsed)
                    with st.expander("🔍 Raw Search Results (from Serper)", expanded=False):
                        st.markdown(result["search_results"])

                    with st.expander("🔧 Generation Details", expanded=False):
                        st.write(f"**Model:** {selected_model}")
                        st.write(f"**Temperature:** {temperature}")
                        st.write(f"**Max Results:** {max_results}")
                        st.write(f"**Query:** {query}")
                        st.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Helpful tips and examples
    with st.expander("💡 Tips for better LinkedIn posts"):
        st.markdown(
            """
            - Use a strong hook in the first 1-2 lines (this increases readership)
            - Short paragraphs improve scan-ability on LinkedIn
            - Ask a direct question at the end to boost comments
            - Use 1-3 emojis maximum, and place them to highlight — not distract
            - If your post includes stats, link to sources or mention where the data was found (we show raw search results)
            """
        )

    st.markdown("---")
    st.caption("Built with ❤️ using Groq + Serper API | LinkedIn-optimized outputs")

if __name__ == "__main__":
    main()
  
