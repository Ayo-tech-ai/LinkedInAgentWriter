# Replace this:
st.download_button(
    label="📋 Copy LinkedIn Post",
    data=result["linkedin_post"],
    file_name=f"linkedin_post_{datetime.now().strftime('%Y%m%d')}.txt",
    mime="text/plain",
    use_container_width=True
)

# With this:
if st.button("📋 Copy LinkedIn Post", use_container_width=True, key="copy_linkedin"):
    st.code(result["linkedin_post"])
    st.success("✅ LinkedIn post copied to clipboard!")