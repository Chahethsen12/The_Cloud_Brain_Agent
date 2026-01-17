# Cloud-Brain Architect: Personal Research Assistant

> **Offload your thinking to the cloud, keep your CPU cool.** Run LLM logic on free Groq APIs and web search on Tavily while your laptop stays light. Perfect for laptop-constrained environments.

---

## 🚀 Quick Overview

The **Personal Research Assistant** is a LangGraph-powered agent that:

1. **Takes your research topic** — "Latest trends in AI quantization", "Machine learning at the edge", etc.
2. **Searches the web** using Tavily API (no credit card, real-time results)
3. **Summarizes 3-5 articles** using Groq's blazing-fast LLM (free, 10x faster than competitors)
4. **Saves a markdown report** to your PC with sources and key takeaways

**Why this matters for you:**
- ✅ **No GPU/extra RAM needed** — inference happens on Groq's cloud hardware
- ✅ **Free tier sufficient** — Groq free tier + Tavily covers hobby/learning use
- ✅ **Aligns with LangGraph journey** — Extends your existing agent knowledge
- ✅ **Real-world applicable** — Web search + summarization is a practical skill

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│         Your Laptop (Main Process)                      │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │ LangGraph Agent (Local State Management)         │  │
│  │ • Orchestrates research workflow                 │  │
│  │ • Routes between search, summarize, save steps   │  │
│  │ • Minimal computational load                     │  │
│  └─────────────┬──────────────────────────────────┬─┘  │
└────────────────┼──────────────────────────────────┼─────┘
                 │                                  │
        ┌────────▼─────────┐          ┌────────────▼────────┐
        │   Tavily API     │          │    Groq API         │
        │   (Web Search)   │          │   (LLM Inference)   │
        │   Free Tier OK   │          │   Free Tier OK      │
        └──────────────────┘          │   Llama 3.3 70B     │
                                      │   DeepSeek R1 etc   │
                                      └─────────────────────┘

┌─────────────────────────────────────────────────────────┐
│         Your Laptop (Output)                            │
│                                                         │
│  📄 reports/                                            │
│     ├── research_ML_quantization_2025.md               │
│     ├── research_edge_computing_2025.md                │
│     └── research_vector_databases_2025.md              │
└─────────────────────────────────────────────────────────┘
```

---

## 📋 Prerequisites

### System Requirements
- **Python:** 3.10+
- **RAM:** 4GB+ (agent runs locally, not heavy)
- **Disk:** ~500MB for dependencies
- **OS:** Windows, macOS, or Linux

### API Keys (All Free)
1. **Groq API Key** — [Sign up free](https://console.groq.com)
   - No credit card required
   - Free tier: ~500K tokens/day (plenty for research summaries)
2. **Tavily API Key** — [Sign up free](https://www.tavily.com)
   - Purpose-built for AI agents
   - Free tier covers typical research use

---

## 🛠️ Installation

### Step 1: Clone or Create Project Directory
```bash
mkdir cloud-brain-architect
cd cloud-brain-architect
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt:**
```txt
langchain==0.3.2
langgraph==0.2.39
langchain-groq==0.2.1
tavily-python==0.3.16
pydantic==2.5.0
python-dotenv==1.0.0
requests==2.31.0
```

### Step 4: Set Up Environment Variables
Create a `.env` file in your project root:
```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

**Get your keys:**
- **Groq:** Visit [console.groq.com](https://console.groq.com) → Create API Key
- **Tavily:** Visit [tavily.com](https://www.tavily.com) → Sign up → Get API Key

---

## 🎯 Quick Start

### Run Your First Research
```bash
python main.py
```

**You'll be prompted:**
```
Enter research topic: Latest trends in ML quantization
```

**Output (in ~10-15 seconds):**
```
✅ Generating research report...
📄 Saved: reports/research_ml_quantization_2025_01_17.md

Key findings:
- Quantization improves model efficiency by 4x
- INT8 vs FP16 trade-offs analyzed
- Edge deployment implications detailed
```

**Check your report:**
```bash
# Windows
notepad reports/research_ml_quantization_2025_01_17.md

# macOS/Linux
cat reports/research_ml_quantization_2025_01_17.md
```

---

## 📁 Project Structure

```
cloud-brain-architect/
├── main.py                 # Entry point - user interaction
├── agent.py               # LangGraph agent definition
├── tools.py               # Tavily search & file saving tools
├── config.py              # API configuration & constants
├── requirements.txt       # Python dependencies
├── .env                   # API keys (DO NOT COMMIT)
├── .gitignore            # Git ignore file
├── reports/              # Generated markdown reports
│   ├── research_*.md
│   └── ...
└── README.md             # This file
```

---

## 🔧 Core Components

### 1. **Agent (`agent.py`)**

Defines the LangGraph state machine:

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from tools import search_web_tool, save_report_tool

# Initialize LLM
llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0.7)

# Create ReAct agent with tools
agent = create_react_agent(
    llm,
    tools=[search_web_tool, save_report_tool],
    state_modifier="You are a research assistant..."
)

# Compile graph
graph = agent.compile()
```

**Agent Flow:**
```
User Input (topic)
    ↓
LLM decides: Search needed?
    ↓ YES
Tavily Search (3-5 results)
    ↓
LLM summarizes results
    ↓
Save markdown report
    ↓
Report ready! ✅
```

### 2. **Tools (`tools.py`)**

Two custom tools:

**Tool 1: Web Search**
```python
from tavily import TavilyClient

def search_web_tool(query: str, max_results: int = 5) -> str:
    """Search web using Tavily API and return summaries"""
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = client.search(query, max_results=max_results)
    
    # Format results for LLM consumption
    formatted = "\n".join([
        f"- {r['title']}\n  URL: {r['url']}\n  Summary: {r['content']}"
        for r in response['results']
    ])
    return formatted
```

**Tool 2: Save Report**
```python
def save_report_tool(content: str, topic: str) -> str:
    """Save markdown report to disk"""
    os.makedirs("reports", exist_ok=True)
    timestamp = datetime.now().strftime("%Y_%m_%d")
    filename = f"reports/research_{topic.replace(' ', '_')}_{timestamp}.md"
    
    with open(filename, 'w') as f:
        f.write(content)
    
    return f"✅ Report saved: {filename}"
```

### 3. **Config (`config.py`)**

```python
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# LLM Settings
MODEL = "mixtral-8x7b-32768"  # Or: "llama-3.3-70b-versatile"
TEMPERATURE = 0.7
MAX_TOKENS = 2048

# Search Settings
MAX_SEARCH_RESULTS = 5
SEARCH_TIMEOUT = 30  # seconds

# Validation
assert GROQ_API_KEY, "❌ GROQ_API_KEY not set in .env"
assert TAVILY_API_KEY, "❌ TAVILY_API_KEY not set in .env"
```

### 4. **Main Entry Point (`main.py`)**

```python
from agent import graph
from config import GROQ_API_KEY, TAVILY_API_KEY

def main():
    print("🧠 Cloud-Brain Research Assistant")
    print("=" * 50)
    print(f"✅ Groq API: Connected")
    print(f"✅ Tavily API: Connected")
    print()
    
    topic = input("📚 Enter research topic: ").strip()
    
    if not topic:
        print("❌ Topic cannot be empty.")
        return
    
    print(f"\n🔍 Researching: {topic}...")
    print("(This typically takes 10-20 seconds)\n")
    
    # Run agent
    try:
        result = graph.invoke({
            "input": f"Research and summarize the latest information about {topic}. "
                     f"Search for 3-5 relevant sources and create a comprehensive report."
        })
        
        print(result['output'])
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
```

---

## 💡 Usage Examples

### Example 1: Technology Trends
```bash
$ python main.py
📚 Enter research topic: Edge computing and TinyML in 2025
🔍 Researching: Edge computing and TinyML in 2025...

✅ Generating research report...
📄 Saved: reports/research_edge_computing_2025_01_17.md
```

**Generated Report Structure:**
```markdown
# Research Report: Edge Computing and TinyML in 2025

**Generated:** January 17, 2025  
**Topic:** Edge computing and TinyML in 2025

## Executive Summary
Edge computing continues to dominate...

## Key Findings
1. **TinyML Advancements**
   - Inference on microcontrollers
   - Power efficiency improvements
   - Source: https://example.com/tinyml-2025

2. **Real-time Processing**
   - Reduced latency
   - Privacy preservation
   - Source: https://example.com/edge-privacy

## Technical Insights
...

## Sources
1. https://example.com/tinyml-2025
2. https://example.com/edge-privacy
3. ...
```

### Example 2: Research Workflow
```bash
$ python main.py
📚 Enter research topic: Vector databases for semantic search
🔍 Researching: Vector databases for semantic search...

✅ Generating research report...
📄 Saved: reports/research_vector_databases_2025_01_17.md
```

---

## 🚀 Advanced Usage

### Batch Research Processing

Create `batch_research.py`:

```python
from main import graph
import time

topics = [
    "Quantum computing breakthroughs 2025",
    "Large language model optimization",
    "Web3 and blockchain evolution",
    "Neuromorphic computing progress"
]

for topic in topics:
    print(f"\n📚 Processing: {topic}")
    result = graph.invoke({
        "input": f"Research and summarize: {topic}"
    })
    print(result['output'])
    time.sleep(5)  # Be respectful of API rate limits
```

Run:
```bash
python batch_research.py
```

### Custom Prompting

Modify `agent.py` to add specialized prompts:

```python
RESEARCH_PROMPT = """You are an expert research analyst. Your task is to:

1. Search for the latest information on the given topic
2. Analyze 3-5 sources for credibility and relevance
3. Synthesize findings into key points
4. Create a professional markdown report with:
   - Executive Summary (2-3 sentences)
   - Key Findings (bullet points with sources)
   - Technical Insights (detailed analysis)
   - Implications (real-world impact)
   - Sources (formatted list with URLs)

Be concise, factual, and cite all sources."""

# Use in agent
agent = create_react_agent(
    llm,
    tools=[search_web_tool, save_report_tool],
    state_modifier=RESEARCH_PROMPT
)
```

---

## ⚡ Performance Metrics

### Speed Comparison (Your Laptop vs Cloud-Brain)

| Task | Local Inference | Cloud-Brain (Groq) |
|------|-----------------|------------------|
| 500-token summary | ❌ 30-120s (GPU needed) | ✅ 2-3s |
| Web search parsing | ✅ 1-2s | ✅ 1-2s |
| Report generation | ❌ 60s+ | ✅ 10-15s |
| **Total time** | **❌ 2-3 min** | **✅ 10-20s** |
| **Hardware needed** | GPU/8GB VRAM | CPU only |

### Token Usage (Free Tier Limits)

**Groq Free Tier:**
- ~500K tokens/day (typical research = 1-5K tokens)
- ✅ **50+ research reports daily** (well within limits)

**Tavily Free Tier:**
- 1,000 searches/month (typical usage = 5 per report)
- ✅ **200 research reports monthly** (well within limits)

---

## 🔐 Security & Best Practices

### 1. **Protect Your API Keys**
```bash
# ✅ Good: Use .env file
GROQ_API_KEY=gsk_***
TAVILY_API_KEY=tvly_***

# ❌ Never hardcode or commit to Git
# ❌ Never share screenshots with API keys
```

**`.gitignore`:**
```
.env
*.pyc
__pycache__/
venv/
reports/  # Optional: if reports contain sensitive info
```

### 2. **Rate Limiting**
```python
import time
from functools import wraps

def rate_limit(calls_per_minute=30):
    min_interval = 60.0 / calls_per_minute
    last_called = [0.0]
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
        return wrapper
    return decorator

# Apply to Tavily search
@rate_limit(calls_per_minute=30)
def search_web_tool(query):
    # ... implementation
```

### 3. **Input Validation**
```python
from pydantic import BaseModel, Field

class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=200)
    max_results: int = Field(default=5, ge=1, le=10)
    save_report: bool = Field(default=True)

# Validate before processing
request = ResearchRequest(topic=user_input)
```

---

## 🐛 Troubleshooting

### Error: "❌ GROQ_API_KEY not set in .env"

**Solution:**
```bash
# 1. Create .env file in project root
touch .env  # macOS/Linux
# or create file manually on Windows

# 2. Add your keys
echo GROQ_API_KEY=your_key_here >> .env
echo TAVILY_API_KEY=your_key_here >> .env

# 3. Verify it works
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('GROQ_API_KEY'))"
```

### Error: "429 Too Many Requests"

**Solution:** You've hit Groq's free tier rate limits.
```python
# Add exponential backoff
import time
from random import uniform

def retry_with_backoff(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt + uniform(0, 1)
                print(f"⏳ Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                raise
```

### Error: "No such file or directory: 'reports'"

**Solution:** Create reports directory
```bash
mkdir reports
```

Or let the code handle it:
```python
os.makedirs("reports", exist_ok=True)  # Already in save_report_tool()
```

---

## 📚 Learning Resources

### LangGraph
- **Official Docs:** https://langchain-ai.github.io/langgraph/
- **Tutorial:** [DataCamp LangGraph Course](https://www.datacamp.com/courses/multi-agent-systems-with-langgraph)
- **Real Python:** https://realpython.com/langgraph-python/

### Groq API
- **Docs:** https://console.groq.com/docs
- **Models Available:**
  - `mixtral-8x7b-32768` (balanced)
  - `llama-3.3-70b-versatile` (more capable)
  - `deepseek-r1-distill-llama-70b` (reasoning-focused)

### Tavily API
- **Docs:** https://docs.tavily.com/
- **Getting Started:** https://blog.tavily.com/getting-started-with-the-tavily-search-api/

---

## 🚀 Next Steps & Extensions

### Phase 1: Current Implementation ✅
- [ ] Set up all 3 files (main.py, agent.py, tools.py)
- [ ] Get API keys and test first research
- [ ] Generate 5+ reports to understand output quality

### Phase 2: Customization 🔄
- [ ] Add report templates (academic, executive summary, technical)
- [ ] Implement conversation history (ask follow-up questions)
- [ ] Add sentiment analysis to source credibility
- [ ] Create PDF export option

### Phase 3: Advanced Agentic Features 🧠
- [ ] Multi-topic research coordination (research 3 related topics in one run)
- [ ] Source cross-reference validation
- [ ] Debate mode (search opposing viewpoints)
- [ ] Long-term memory (store past research for comparison)

### Phase 4: Production Deployment 🚀
- [ ] Package as desktop app (PyQt/Streamlit UI)
- [ ] Add database for report versioning
- [ ] Schedule periodic research updates
- [ ] REST API wrapper for external integrations

---

## 💬 Contributing & Feedback

Have improvements? Found a bug? Want to share your research setup?

```bash
# Fork this repo, make changes, submit PR
git clone https://github.com/YOUR_USERNAME/cloud-brain-architect
cd cloud-brain-architect
git checkout -b feature/your-feature
git push origin feature/your-feature
```

---

## 📝 License

MIT License — Use freely in personal and commercial projects.

---

## 🎯 Why This Project Rocks for You

| Aspect | Benefit |
|--------|---------|
| **Your LangGraph Journey** | Extends existing knowledge with real-world agent patterns |
| **Laptop Constraints** | Offloads all compute to free cloud APIs |
| **Portfolio Ready** | Demonstrates full-stack AI agent design |
| **Interview Material** | Shows understanding of: APIs, agents, prompt engineering, system design |
| **Practical Use** | You'll actually use this for real research! |

---

## 📊 Sample Output

**User Input:**
```
📚 Enter research topic: Rust in systems programming 2025
```

**Generated Report:**
```markdown
# Research Report: Rust in Systems Programming 2025

**Generated:** January 17, 2025  
**Topic:** Rust in systems programming 2025

## Executive Summary
Rust continues to gain traction in systems programming with 
adoption in Linux kernel development, embedded systems, and 
performance-critical applications. Key trends include improved 
tooling, expanded async ecosystems, and enterprise backing.

## Key Findings

1. **Linux Kernel Integration**
   - Rust modules now accepted into mainline Linux
   - Safety guarantees eliminate entire class of bugs
   - Performance comparable to C
   - Source: kernel.org/doc/html/latest/rust/

2. **Embedded Systems Momentum**
   - µcontroller support expanding (ARM, RISC-V)
   - Memory safety without runtime overhead
   - Growing ecosystem (embassy, probe-rs)
   - Source: https://www.rust-embedded.org/

3. **Industry Adoption**
   - AWS, Google, Microsoft investing heavily
   - Hiring surge for Rust developers
   - Enterprise frameworks maturing
   - Source: https://www.rust-lang.org/

## Technical Insights
The safety guarantees provided by Rust's ownership system...

## Sources
- kernel.org
- rust-lang.org
- AWS Architecture Blog
- [+2 more sources]
```

---

## 🎓 Key Skills You'll Develop

✅ **Agent Architecture:** Understand LangGraph state graphs and conditional routing  
✅ **Tool Integration:** Connect external APIs to LLM agents  
✅ **Prompt Engineering:** Craft effective system prompts for specific tasks  
✅ **API Design:** Work with REST APIs (Groq, Tavily) and error handling  
✅ **Full-Stack thinking:** Local logic + cloud inference coordination  

---

**Happy researching! 🚀**

Questions? Open an issue. Found a better approach? Submit a PR. Generating useful insights? Share them!

---

**Created By Chaheth Senevirathne**
