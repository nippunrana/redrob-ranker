"""JD requirements for Senior AI Engineer @ Redrob AI.

This module encodes the *meaning* of the JD, not just its keywords.
The "read between the lines" section of the JD is the primary source.

Key design choices (all trace to explicit JD text):
- Negative signals (disqualifiers) are explicit, not implied.
- Title features penalise roles that *sound* elite but the JD rejects
  (AI Research Engineer = pure-research red flag; Computer Vision Engineer
  = CV/speech/robotics-primary, explicitly rejected).
- Career-history keywords capture "actually built" systems, which keyword-
  stuffers cannot fake by listing skills.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# JD text — used as the semantic query for embedding recall
# ---------------------------------------------------------------------------

JD_TEXT: str = """
Senior AI Engineer - Founding Team at Redrob AI. Pune/Noida hybrid, India.

We need someone who has built production retrieval and ranking systems — embeddings,
vector databases, hybrid search, learning-to-rank, evaluation frameworks. Not
researchers. Shippers. 5-9 years experience, mostly at product companies not
consulting firms.

Must have: production embeddings-based retrieval (sentence-transformers, BGE, E5,
OpenAI embeddings), vector database or hybrid search infrastructure (Pinecone,
Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS), strong Python, hands-on
evaluation framework design (NDCG, MRR, MAP, A/B testing, offline-to-online
correlation).

Nice to have: LLM fine-tuning (LoRA, QLoRA, PEFT), learning-to-rank (XGBoost,
neural), HR-tech or marketplace experience, distributed systems, open-source ML.

Ideal: 6-8 years total, 4-5 in applied ML at product companies. Shipped at least one
end-to-end ranking, search, or recommendation system to real users at meaningful scale.
Located in or willing to relocate to Noida, Pune, Hyderabad, Mumbai, Delhi NCR.
Active on the job market.
"""

# ---------------------------------------------------------------------------
# Title classification — drawn directly from the 47 distinct titles in data
# ---------------------------------------------------------------------------

# Titles that are strong evidence of genuine ML/IR/RecSys engineering production work.
TITLE_STRONG_POSITIVE: frozenset[str] = frozenset(
    {
        "ML Engineer",
        "Machine Learning Engineer",
        "Senior Machine Learning Engineer",
        "Staff Machine Learning Engineer",
        "Applied ML Engineer",
        "Senior Applied Scientist",
        "NLP Engineer",
        "Senior NLP Engineer",
        "Recommendation Systems Engineer",
        "Search Engineer",
        "Senior AI Engineer",
        "Lead AI Engineer",
    }
)

# Titles the JD explicitly penalises by role type.
# "AI Research Engineer" → pure-research red flag (JD: "no production deployment").
# "Computer Vision Engineer" → CV/robotics/speech-primary (JD: fundamentals mismatch).
# "AI Specialist" → often pure-research / consulting-flavored; needs career check.
TITLE_RESEARCH_OR_CV: frozenset[str] = frozenset(
    {
        "AI Research Engineer",
        "Computer Vision Engineer",
        "AI Specialist",
    }
)

# Titles that are adjacent/neutral — can still be strong if career history shows
# retrieval/ranking production work (the "plain-language tier-5" pattern).
TITLE_ADJACENT_TECH: frozenset[str] = frozenset(
    {
        "AI Engineer",  # inspect career: could be strong or stuffer
        "Data Scientist",
        "Senior Data Scientist",
        "Analytics Engineer",
        "Data Engineer",
        "Senior Data Engineer",
        "Backend Engineer",
        "Software Engineer",
        "Senior Software Engineer",
        "Senior Software Engineer (ML)",
        "Full Stack Developer",
        "Cloud Engineer",
        "DevOps Engineer",
        "QA Engineer",
        "Java Developer",
        ".NET Developer",
        "Mobile Developer",
        "Frontend Engineer",
    }
)

# Titles that are hard-disqualifiers for this JD.
# The sample_submission.csv deliberately ranks these #1-19 to show the wrong answer.
TITLE_DISQUALIFIER: frozenset[str] = frozenset(
    {
        "HR Manager",
        "Accountant",
        "Content Writer",
        "Graphic Designer",
        "Business Analyst",
        "Sales Executive",
        "Customer Support",
        "Operations Manager",
        "Mechanical Engineer",
        "Civil Engineer",
        "Marketing Manager",
        "Project Manager",
    }
)

# ---------------------------------------------------------------------------
# Services-company disqualifier (JD: "entire career at consulting firms")
# Applied only when the candidate has NO product-company experience.
# ---------------------------------------------------------------------------

SERVICES_COMPANIES: frozenset[str] = frozenset(
    {
        "tcs",
        "tata consultancy",
        "infosys",
        "wipro",
        "accenture",
        "cognizant",
        "capgemini",
        "mindtree",
        "hcl",
        "mphasis",
        "tech mahindra",
        "hexaware",
        "ltimindtree",
        "birlasoft",
    }
)

# ---------------------------------------------------------------------------
# Career-history keyword signals — what "actually built" looks like
# ---------------------------------------------------------------------------

# Terms in job descriptions that indicate genuine retrieval/ranking production work.
# Presence (especially multiple of these) promotes plain-language tier-5s.
CAREER_RETRIEVAL_KEYWORDS: tuple[str, ...] = (
    "retrieval",
    "ranking",
    "recommendation",
    "embedding",
    "vector search",
    "semantic search",
    "hybrid search",
    "information retrieval",
    "learning to rank",
    "learning-to-rank",
    "bm25",
    "tf-idf",
    "faiss",
    "pinecone",
    "weaviate",
    "qdrant",
    "milvus",
    "opensearch",
    "elasticsearch",
    "rag",
    "retrieval augmented",
    "fine-tuning",
    "fine tuning",
    "ndcg",
    "mrr",
    "a/b test",
    "a/b testing",
    "offline-online",
    "offline evaluation",
    "relevance",
    "reranking",
    "rerank",
    "sentence-transformer",
    "sentence transformer",
    "dense retrieval",
    "sparse retrieval",
    "ann index",
    "approximate nearest",
)

# Terms that indicate production delivery (not just research).
CAREER_PRODUCTION_KEYWORDS: tuple[str, ...] = (
    "shipped",
    "deployed",
    "production",
    "launched",
    "at scale",
    "real users",
    "live system",
    "millions",
    "revenue",
    "engagement",
    "a/b test",
    "a/b testing",
    "latency",
    "throughput",
)

# Red-flag terms in descriptions that indicate non-technical work.
CAREER_NONTTECH_KEYWORDS: tuple[str, ...] = (
    "marketing",
    "branding",
    "social media",
    "accounting",
    "hr process",
    "talent acquisition",
    "graphic design",
    "illustrat",
    "mechanical",
    "civil engineering",
    "construction",
    "sales pipeline",
)

# LangChain-only penalty terms — recent (under 12 months) LangChain w/o prior ML.
LANGCHAIN_ONLY_KEYWORDS: tuple[str, ...] = (
    "langchain",
    "llamaindex",
    "llama index",
    "llama_index",
    "flowise",
    "chainlit",
)

# ---------------------------------------------------------------------------
# Skill name lists — for skill-trust feature computation
# ---------------------------------------------------------------------------

# Must-have skills (from JD "things you absolutely need").
SKILLS_MUST_HAVE: frozenset[str] = frozenset(
    {
        "embeddings",
        "sentence transformers",
        "sentence-transformers",
        "vector database",
        "vector search",
        "faiss",
        "pinecone",
        "weaviate",
        "qdrant",
        "milvus",
        "opensearch",
        "elasticsearch",
        "information retrieval",
        "python",
        "ndcg",
        "mrr",
        "learning to rank",
        "hybrid search",
        "bm25",
        "dense retrieval",
        "rag",
        "retrieval augmented generation",
    }
)

# Nice-to-have skills.
SKILLS_NICE_TO_HAVE: frozenset[str] = frozenset(
    {
        "lora",
        "qlora",
        "peft",
        "fine-tuning llms",
        "fine tuning",
        "xgboost",
        "lightgbm",
        "recommendation systems",
        "mlops",
        "mlflow",
        "weights & biases",
        "distributed systems",
        "kafka",
        "spark",
        "transformer",
        "hugging face",
        "hugging face transformers",
        "pgvector",
        "redis",
        "open-source",
    }
)

# Skills that are CV/speech/robotics — penalise if they dominate the profile.
SKILLS_CV_SPEECH: frozenset[str] = frozenset(
    {
        "computer vision",
        "image classification",
        "object detection",
        "speech recognition",
        "text to speech",
        "ocr",
        "robotics",
        "autonomous driving",
        "yolo",
        "cnn",
        "openCV",
        "mediapipe",
        "gans",
        "diffusion models",
    }
)

# ---------------------------------------------------------------------------
# Location preferences (JD: Pune, Noida preferred; Hyd/Mum/Delhi-NCR OK)
# ---------------------------------------------------------------------------

LOCATIONS_PREFERRED: frozenset[str] = frozenset(
    {
        "pune",
        "noida",
        "hyderabad",
        "mumbai",
        "delhi",
        "ncr",
        "gurugram",
        "gurgaon",
        "bangalore",
        "bengaluru",
        "chennai",
    }
)

# Non-India locations: no visa sponsorship, down-weight unless willing to relocate.
LOCATIONS_NON_INDIA: frozenset[str] = frozenset(
    {
        "toronto",
        "london",
        "berlin",
        "singapore",
        "dubai",
        "new york",
        "san francisco",
        "seattle",
        "amsterdam",
        "sydney",
        "melbourne",
        "paris",
        "tokyo",
        "stockholm",
        "zurich",
    }
)

# ---------------------------------------------------------------------------
# Experience range (JD: "5-9 years"; ideal "6-8 years")
# ---------------------------------------------------------------------------

EXPERIENCE_IDEAL_MIN: float = 6.0
EXPERIENCE_IDEAL_MAX: float = 8.0
EXPERIENCE_SOFT_MIN: float = 4.0   # below this starts losing points
EXPERIENCE_SOFT_MAX: float = 10.0  # above this starts losing points (over-seniorised)
EXPERIENCE_HARD_MIN: float = 2.0   # hard floor
