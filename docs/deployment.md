# Public Streamlit demo deployment (Render)

The publicly hosted DataSage instance is **a demo**, not a confidential data processing service.

Configuration:
- Runtime: Python 3.12
- Build: `pip install -e ".[dashboard]"`
- Start: `streamlit run src/datasage/dashboard.py --server.headless true --server.address 0.0.0.0 --server.port $PORT`
- Environment: `DATASAGE_PUBLIC_DEMO=1` and `PYTHONUNBUFFERED=1`
- Plan: free (subject to provider limits)

In public mode:
- Accept CSV only, maximum 2 MiB per file, 20,000 rows and 100 columns.
- Do not expose LLM API calls or use API credentials on the public deployment.
- Disable the shared server's persistent investigation-saving/listing UI.
- Delete temporary dataset files after analysis. Browser-session findings remain available during the session and can be downloaded.
- Show a clear warning not to upload sensitive data.

## Safety limitations

The public service is not authenticated and not suitable for proprietary, confidential, medical, or personal datasets. It has no per-user isolation layer beyond Streamlit session state, no organization-grade resource quotas, and no audit of third-party hosting policies. The app has a 2 MiB input limit, but complex CSV inputs and concurrent visitors may still exhaust free-tier CPU/memory. Free services may sleep and have limited availability. The instance should not be represented as production hardened.

Local installation using `streamlit run src/datasage/dashboard.py` keeps the original supported formats and optional local report saving.

For secure organizational deployment, add authentication, process isolation, request-level quotas, authorization around saved artifacts, secure storage, and defensive parsers before expanding upload limits.
