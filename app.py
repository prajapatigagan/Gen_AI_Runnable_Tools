import os
import requests
import streamlit as st

from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import HumanMessage, ToolMessage

# ==========================================
# CONFIGURATION
# ==========================================

load_dotenv()

st.set_page_config(
    page_title="City AI | Smart Assistant",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at top left, #182848 0%, transparent 45%),
        linear-gradient(135deg, #080d1b 0%, #111827 55%, #10172a 100%);
    color: #f1f5f9;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #10182b, #0b1120);
    border-right: 1px solid rgba(148, 163, 184, 0.15);
}

.hero {
    padding: 32px 34px;
    border-radius: 24px;
    background: linear-gradient(
        120deg,
        rgba(37, 99, 235, 0.23),
        rgba(124, 58, 237, 0.18),
        rgba(15, 23, 42, 0.65)
    );
    border: 1px solid rgba(129, 140, 248, 0.28);
    margin-bottom: 24px;
}

.hero h1 {
    font-size: 39px;
    font-weight: 800;
    letter-spacing: -1.5px;
    margin: 0;
    color: #ffffff;
}

.hero p {
    color: #b8c5dc;
    font-size: 15px;
    line-height: 1.8;
    margin-top: 12px;
}

.badge {
    display: inline-block;
    padding: 7px 13px;
    border-radius: 30px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    color: #a5f3fc;
    background: rgba(6, 182, 212, 0.12);
    border: 1px solid rgba(34, 211, 238, 0.25);
    margin-bottom: 15px;
}

.section-title {
    font-size: 19px;
    font-weight: 700;
    color: #f8fafc;
    margin: 20px 0 14px 0;
}

.metric-card {
    padding: 20px;
    min-height: 120px;
    border-radius: 18px;
    background: rgba(30, 41, 59, 0.58);
    border: 1px solid rgba(148, 163, 184, 0.16);
}

.metric-card .label {
    color: #94a3b8;
    font-size: 12px;
    margin-bottom: 12px;
}

.metric-card .value {
    color: #f8fafc;
    font-size: 21px;
    font-weight: 800;
}

.metric-card .icon {
    font-size: 23px;
    margin-bottom: 10px;
}

.feature-card {
    padding: 20px;
    border-radius: 17px;
    background: rgba(30, 41, 59, 0.48);
    border: 1px solid rgba(148, 163, 184, 0.13);
    margin-bottom: 12px;
}

.feature-card h4 {
    color: #e2e8f0;
    margin: 0 0 8px 0;
    font-size: 15px;
}

.feature-card p {
    color: #94a3b8;
    font-size: 12px;
    line-height: 1.7;
    margin: 0;
}

.stButton > button {
    border-radius: 12px;
    border: 1px solid rgba(129, 140, 248, 0.35);
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    font-weight: 600;
    transition: 0.2s ease;
}

.stButton > button:hover {
    border-color: #a5b4fc;
    color: white;
    transform: translateY(-1px);
}

.stTextInput input,
.stTextArea textarea {
    background: rgba(15, 23, 42, 0.8);
    color: #f8fafc;
    border-radius: 12px;
    border: 1px solid rgba(148, 163, 184, 0.25);
}

[data-testid="stChatMessage"] {
    background: rgba(30, 41, 59, 0.42);
    border: 1px solid rgba(148, 163, 184, 0.10);
    border-radius: 16px;
    padding: 14px;
}

.stChatInput textarea {
    background: #111b2e !important;
    color: white !important;
}

hr {
    border-color: rgba(148, 163, 184, 0.15);
}

.small-muted {
    color: #94a3b8;
    font-size: 12px;
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 12px;
    padding: 24px 0 10px 0;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# API CLIENTS
# ==========================================

@st.cache_resource
def get_tavily_client():
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        return None
    return TavilyClient(api_key=key)


# ==========================================
# WEATHER TOOL
# ==========================================

@tool
def get_weather(city: str) -> str:
    """Get the current weather of a city in India."""

    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        return "Error: OPENWEATHER_API_KEY is missing from the .env file."

    url = "https://api.openweathermap.org/data/2.5/weather"

    try:
        response = requests.get(
            url,
            params={
                "q": f"{city},IN",
                "appid": api_key,
                "units": "metric"
            },
            timeout=15
        )

        response.raise_for_status()
        data = response.json()

        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        description = data["weather"][0]["description"]
        wind_speed = data["wind"]["speed"]

        return (
            f"Weather in {city.title()}:\n"
            f"Temperature: {temp}°C\n"
            f"Feels like: {feels_like}°C\n"
            f"Condition: {description}\n"
            f"Humidity: {humidity}%\n"
            f"Wind speed: {wind_speed} m/s"
        )

    except requests.exceptions.HTTPError:
        if response.status_code == 401:
            return "Error: Invalid OpenWeather API key."
        if response.status_code == 404:
            return f"Error: City '{city}' was not found."
        return f"Weather API error: HTTP {response.status_code}"

    except requests.exceptions.RequestException as exc:
        return f"Could not connect to the weather service: {exc}"

    except (KeyError, ValueError):
        return "Error: Unexpected response received from the weather API."


# ==========================================
# TAVILY NEWS TOOL
# ==========================================

@tool
def get_news(city: str) -> str:
    """Search the web for the latest news about a city."""

    client = get_tavily_client()

    if client is None:
        return "Error: TAVILY_API_KEY is missing from the .env file."

    try:
        response = client.search(
            query=f"latest news in {city}",
            search_depth="basic",
            max_results=3
        )

        results = response.get("results", [])

        if not results:
            return f"No news found for {city}."

        news_list = []

        for item in results:
            title = item.get("title", "Untitled")
            url = item.get("url", "")
            snippet = item.get("content", "")

            news_list.append(
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Summary: {snippet[:400]}"
            )

        return (
            f"Latest web search results about {city}:\n\n"
            + "\n\n".join(news_list)
        )

    except Exception as exc:
        return f"News search failed: {exc}"


# ==========================================
# HUMAN APPROVAL MIDDLEWARE
# ==========================================

@wrap_tool_call
def human_approval(request, handler):
    """
    Streamlit-compatible tool permission check.

    Tool calls are allowed only when the user enables
    the tool execution switch in the sidebar.
    """

    tool_name = request.tool_call["name"]

    if not st.session_state.get("allow_tools", False):
        return ToolMessage(
            content=(
                f"Permission denied for tool '{tool_name}'. "
                "The user has disabled tool execution. "
                "Ask the user to enable tool execution in the sidebar."
            ),
            tool_call_id=request.tool_call["id"]
        )

    return handler(request)


# ==========================================
# AI AGENT
# ==========================================

@st.cache_resource
def create_city_agent():
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        return None

    llm = ChatOpenAI(
        model="openrouter/free",
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        temperature=0.4,
        max_retries=2,
        timeout=60
    )

    return create_agent(
        llm,
        tools=[get_weather, get_news],
        system_prompt="""
        You are City AI, a helpful city information assistant.

        Your capabilities:
        1. Get current weather for Indian cities.
        2. Search the web for recent city news.

        Instructions:
        - Answer in the same language as the user.
        - You can communicate in English, Hindi, or Hinglish.
        - Use get_weather when the user asks about current weather.
        - Use get_news when the user asks for recent news.
        - Do not invent live weather data or news.
        - Explain tool errors honestly.
        - Keep answers clear, friendly, and easy to understand.
        - When a tool is denied, explain that tool execution is disabled.
        - For general questions, answer directly when possible.
        """,
        middleware=[human_approval]
    )


# ==========================================
# SESSION STATE
# ==========================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []

if "allow_tools" not in st.session_state:
    st.session_state.allow_tools = False


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("""
    <div style="padding: 10px 0 20px 0;">
        <div style="font-size: 38px;">🌍</div>
        <div style="font-size: 23px; font-weight: 800; color: white;">
            City AI
        </div>
        <div class="small-muted">
            Your intelligent city companion
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### ⚙️ Agent Settings")

    st.toggle(
        "Enable tool execution",
        key="allow_tools",
        help=(
            "When enabled, the agent can call the weather and news APIs "
            "without asking for individual confirmation."
        )
    )

    if st.session_state.allow_tools:
        st.success("Tools enabled")
    else:
        st.warning("Tools disabled")

    st.markdown("""
    <div class="feature-card">
        <h4>🌦️ Weather Engine</h4>
        <p>Live weather, temperature, humidity and wind speed.</p>
    </div>
    <div class="feature-card">
        <h4>📰 News Intelligence</h4>
        <p>Web search powered by Tavily for recent city news.</p>
    </div>
    <div class="feature-card">
        <h4>🤖 AI Reasoning</h4>
        <p>OpenRouter-powered agent with tool calling.</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.agent_messages = []
        st.rerun()

    st.markdown("""
    <div class="footer">
        Built with Streamlit + LangChain<br>
        City AI Assistant
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# MAIN DASHBOARD
# ==========================================

st.markdown("""
<div class="hero">
    <div class="badge">✦ INTELLIGENT CITY ASSISTANT</div>
    <h1>Explore your city with AI.</h1>
    <p>
        Ask about live weather, discover the latest news, and get useful
        city information — all from one intelligent assistant.
    </p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# STATUS CARDS
# ==========================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="icon">🌦️</div>
        <div class="label">WEATHER SERVICE</div>
        <div class="value">OpenWeather</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="icon">📰</div>
        <div class="label">WEB SEARCH</div>
        <div class="value">Tavily Search</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="icon">🤖</div>
        <div class="label">LANGUAGE MODEL</div>
        <div class="value">OpenRouter</div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# QUICK PROMPTS
# ==========================================

st.markdown(
    '<div class="section-title">✨ What would you like to explore?</div>',
    unsafe_allow_html=True
)

prompt_col1, prompt_col2, prompt_col3 = st.columns(3)

quick_prompt = None

with prompt_col1:
    if st.button(
        "🌤️ Weather in Gwalior",
        use_container_width=True
    ):
        quick_prompt = "What is the current weather in Gwalior?"

with prompt_col2:
    if st.button(
        "📰 Latest Delhi News",
        use_container_width=True
    ):
        quick_prompt = "Find the latest news about Delhi."

with prompt_col3:
    if st.button(
        "🌍 Weather in Mumbai",
        use_container_width=True
    ):
        quick_prompt = "What is the current weather in Mumbai?"


st.divider()


# ==========================================
# CHAT INTERFACE
# ==========================================

st.markdown(
    '<div class="section-title">💬 Chat with City AI</div>',
    unsafe_allow_html=True
)

# Display existing conversation
for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat input
user_input = st.chat_input(
    "Ask about weather, city news, or anything else..."
)

# Support quick-prompt buttons
if quick_prompt:
    user_input = quick_prompt


# ==========================================
# PROCESS USER MESSAGE
# ==========================================

if user_input:

    user_input = user_input.strip()

    if user_input:

        # Display user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        # Validate API configuration
        required_keys = {
            "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
        }

        missing_keys = [
            key for key, value in required_keys.items()
            if not value
        ]

        if missing_keys:

            error_message = (
                "Missing configuration: "
                + ", ".join(missing_keys)
                + ". Please add the required keys to your .env file."
            )

            with st.chat_message("assistant"):
                st.error(error_message)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": error_message
            })

        else:

            try:

                with st.chat_message("assistant"):

                    with st.spinner(
                        "City AI is thinking and gathering information..."
                    ):

                        agent = create_city_agent()

                        # Keep the conversation context.
                        st.session_state.agent_messages.append(
                            HumanMessage(content=user_input)
                        )

                        result = agent.invoke({
                            "messages": st.session_state.agent_messages
                        })

                        # Save all returned messages for future context.
                        st.session_state.agent_messages = result["messages"]

                        # Find the last AI response with actual text.
                        answer = ""

                        for message in reversed(result["messages"]):
                            if (
                                getattr(message, "type", "") == "ai"
                                and getattr(message, "content", "")
                            ):
                                answer = message.content
                                break

                        if not answer:
                            answer = (
                                "I couldn't generate a response. "
                                "Please try again."
                            )

                    st.markdown(answer)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as exc:

                error_message = (
                    f"Something went wrong: {exc}\n\n"
                    "Check your API keys, internet connection, "
                    "and OpenRouter model availability."
                )

                with st.chat_message("assistant"):
                    st.error(error_message)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_message
                })


# ==========================================
# FOOTER
# ==========================================

st.markdown("""
<div class="footer">
    City AI · Smart Answers for a Smarter City 🌍
</div>
""", unsafe_allow_html=True)

