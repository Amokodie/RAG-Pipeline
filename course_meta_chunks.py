"""
Course / site FAQ chunks — prepended to the IT KB so retrieval can answer
“what is this site”, AeroFleet vs Session 7, RAG overview, etc.
"""

from __future__ import annotations

# (title, category, body, source)
COURSE_META_DOCUMENTS: list[tuple[str, str, str, str]] = [
    (
        "What this Streamlit application is (Session 7)",
        "Course_meta",
        "This **main** Streamlit app (`app.py`) is the **Session 6–8 / Assignment 3 RAG concept demo**. "
        "It loads a small **alignment audit CSV** (cases like H01, O01, S04, B01) about helpfulness, safety, honesty, and bias. "
        "It is **not** the AeroFleet drone manual. Retrieval uses TF‑IDF and optional hybrid BM25+dense embeddings; tabs show analysis, live retrieval, Ask AI, and Advanced settings. "
        "If you ask about **AeroFleet** here, retrieval may still pick an unrelated audit row unless you use the dedicated AeroFleet app (see separate entry).",
        "curated",
    ),
    (
        "AeroFleet X200 — where that content lives",
        "Course_meta",
        "**AeroFleet X200** is a **fictional** battery cooling / maintenance **engineering corpus** (documents D01–D10) used in **Session 8** exercises. "
        "It lives in this repo under `rag_session8_classroom_exercise/` and has its **own** Streamlit entry: run `streamlit run rag_session8_classroom_exercise/aerofleet_rag_app.py`. "
        "That app uses **FAISS** + sentence-transformers over markdown manuals — not the Session 7 CSV. "
        "Do not confuse AeroFleet procedures with the alignment-audit case IDs in this app.",
        "curated",
    ),
    (
        "What is RAG (in this course)",
        "Course_meta",
        "**Retrieval-Augmented Generation (RAG)** means: (1) **chunk** documents, (2) **embed** chunks into vectors, (3) **retrieve** top‑k chunks for a user query, (4) **condition** an LLM (or show grounded text) so answers cite retrieved facts instead of guessing. "
        "This reduces **hallucination** compared to using the model alone. This demo compares TF‑IDF, dense/hybrid retrieval, and optional OpenAI with strict grounding.",
        "curated",
    ),
    (
        "Are you a bot",
        "Course_meta",
        "The **Ask AI** area is a **retrieval + template / API** interface: it is not a human. "
        "Offline mode stitches instructor notes and a local SQLite **IT knowledge base**; with an OpenAI key it calls the API with **grounding rules**. "
        "There is no persistent consciousness — only your query, retrieved rows, and the prompt.",
        "curated",
    ),
    (
        "Session 7 vs Session 8 in this repository",
        "Course_meta",
        "**Session 7** material is the alignment-audit dataset and this **RAG Concept Demo** (`app.py`). "
        "**Session 8** adds the **AeroFleet** classroom exercise folder with corpus D01–D10, eval queries, and `aerofleet_rag_app.py`. "
        "Use the correct app for each assignment part.",
        "curated",
    ),
    (
        "Why a wrong audit row sometimes appears",
        "Course_meta",
        "If your question does not match any row in the CSV well, **semantic retrieval** still returns the *least bad* chunk — which can look unrelated (e.g. a Safety case when you asked about AeroFleet). "
        "Check **audit relevance score** and the **Course_meta** / IT KB passages. For AeroFleet-specific answers, open the **AeroFleet** Streamlit app instead.",
        "curated",
    ),
    (
        "3D plots and LSA in this demo",
        "Course_meta",
        "The **Analysis & 3D** tab projects TF‑IDF chunks with **LSA** (truncated SVD) for visualization. "
        "**Ask AI** can show a compact **3D LSA** view of the query vs chunks after each question. "
        "3D is for intuition — actual retrieval still uses full TF‑IDF cosine (or hybrid) in high dimensions, not 3D distance.",
        "curated",
    ),
    (
        "OpenAI API key in Ask AI — privacy and safety",
        "Course_meta",
        "If you paste an **OpenAI API key** into this Streamlit app, treat it like a password: **anyone with access to the machine or screen** could see it. "
        "Prefer **environment variables** or **Streamlit Secrets** on deployment. **Rotate** the key if you suspect exposure. "
        "This course demo does not store your key on a server you do not control — but **never commit keys to Git**. "
        "Costs depend on your OpenAI billing; use small models and limits for class experiments.",
        "curated",
    ),
    (
        "Ask AI — what it can and cannot do",
        "Course_meta",
        "**Ask AI** combines (1) the **closest alignment-audit row**, (2) the **local SQLite knowledge base** (IT + course FAQ), and optionally (3) **OpenAI**. "
        "It is **not** a substitute for your instructor, official grades, or mental-health crisis services. "
        "It can explain RAG concepts, IT topics, and how this repo is structured — with **retrieval-grounded** text when matches are good.",
        "curated",
    ),
    (
        "Troubleshooting: app or video not loading",
        "Course_meta",
        "If Streamlit **won’t start**, check Python version, run `pip install -r requirements.txt`, and run from the repo root: `streamlit run app.py`. "
        "If the **explainer video** is missing on Streamlit Cloud, ensure `rag_session8_classroom_exercise/assets/rag_explainer.mp4` is in the repo or set **`RAG_EXPLAINER_VIDEO_URL`** to a direct MP4 URL in Secrets. "
        "Hard-refresh the browser after redeploy.",
        "curated",
    ),
    (
        "Academic integrity and using AI for this assignment",
        "Course_meta",
        "Follow your **course syllabus** and instructor rules on generative AI. Generally: **disclose** AI assistance where required, **cite** tools and repos you used, and **understand** every line you submit. "
        "Using this demo to **learn** RAG and alignment is the point; **copy-pasting** answers without understanding violates typical integrity expectations.",
        "curated",
    ),
    (
        "Citing this project or repository in a report",
        "Course_meta",
        "Cite the **repository URL**, **commit** or **release** if applicable, **authors / team names** from the app banner, and the **course name**. "
        "Describe **RAG** as retrieval-augmented generation and name the **datasets** (alignment audit CSV, optional AeroFleet corpus). "
        "Screenshots of Streamlit tabs can illustrate your pipeline — label figures clearly.",
        "curated",
    ),
    (
        "Grades and assessment — what this app does not know",
        "Course_meta",
        "This assistant has **no access** to your LMS, marks, or instructor decisions. "
        "For **rubrics, weighting, extensions, and appeals**, ask your **instructor or TA** through official channels. "
        "The app can only help you **study concepts** (RAG, alignment, IT topics) grounded in this repo’s materials.",
        "curated",
    ),
    (
        "Group work and team coordination for the assignment",
        "Course_meta",
        "Clarify **roles** (who runs the app, who records the video, who writes the report), use **version control** (Git branches, pull requests), and **merge** early to avoid last-minute conflicts. "
        "One teammate can deploy Streamlit Cloud while another extends the corpus — document who did what for your report.",
        "curated",
    ),
    (
        "English / ESL — using Ask AI with non-native English",
        "Course_meta",
        "You may ask in **simple English** or mixed wording; retrieval uses **lexical overlap**, so adding **keywords** from the course (RAG, alignment, TF‑IDF, AeroFleet) improves hits. "
        "If retrieval is weak, rephrase with terms from the **Overview** or **Concepts** tabs.",
        "curated",
    ),
    (
        "Accessibility and inclusive use of the demo",
        "Course_meta",
        "Streamlit supports browser **zoom** and OS **screen readers** to varying degrees. "
        "If you need **academic accommodations**, contact your institution’s disability office — this app cannot grant exam or deadline changes.",
        "curated",
    ),
    (
        "Time management and deadlines (study tips)",
        "Course_meta",
        "Break the assignment into **indexing**, **retrieval demo**, **write-up**, and **video**. "
        "Run the app **locally** first; deploy when stable. **Buffer** time for GPU/model downloads (sentence-transformers) on first run.",
        "curated",
    ),
    (
        "Career relevance — RAG and MLOps keywords",
        "Course_meta",
        "Employers in **ML platform** and **search** roles care about **embeddings**, **vector DBs**, **evaluation** of retrieval, and **guardrails**. "
        "This project demonstrates **end-to-end** thinking: chunking, indexing, grounding, and honest failure modes — good interview stories if you can explain **your** design choices.",
        "curated",
    ),
    (
        "Common fears about hallucinations and this lab",
        "Course_meta",
        "It is normal to worry that **LLMs lie**. This course uses **RAG** precisely to **reduce** unsupported claims by conditioning on retrieved text. "
        "Your report can contrast **parametric** guesses vs **non-parametric** manual chunks (see alignment cases O01, H04).",
        "curated",
    ),
    (
        "Data and privacy — what leaves your computer",
        "Course_meta",
        "Running **locally**, queries stay on your machine unless you enable **OpenAI** (then user text is sent to OpenAI per their API policy). "
        "**Streamlit Cloud** runs on Streamlit’s hosting; read their privacy terms. **Do not** paste real personal secrets or classified data into class demos.",
        "curated",
    ),
    (
        "If you feel overwhelmed or in crisis",
        "Course_meta",
        "This app is **educational**, not therapy. If you are in **crisis** or need mental-health support, contact **local emergency services**, your **university counseling center**, or a **trusted professional** in your country. "
        "It is **strength** to ask for help early.",
        "curated",
    ),
    (
        "What is hallucination (definition for this course)",
        "Course_meta",
        "**Hallucination** (in LLMs) means the model generates **plausible-sounding content that is false, unsupported, or not grounded** in the evidence it was given—e.g. **invented citations**, wrong dates, or details that **look** factual but are not. "
        "It is different from **I don’t know** (refusal) or clearly hedged uncertainty. "
        "**Why it happens:** models learn statistical patterns over text; without **external grounding**, they can **confabulate** fluent answers that mix truth-like form with incorrect substance (**parametric** memory is not a verified database). "
        "**Mitigations** include **RAG** (retrieve real documents before answering), **tools** (calculators, APIs), **citations to verified sources**, **uncertainty calibration** (“I’m not sure”), and **human review** for high-stakes use. "
        "Poor retrieval (wrong chunks) can still produce bad answers—**retrieval quality** and **attribution** matter. "
        "In this CSV lab, case **O01** illustrates **citation-style** hallucination; case **H04** is about **sycophancy** (agreeing with a false premise)—related to alignment failures, **not** the textbook definition of hallucination.",
        "curated",
    ),
    (
        "Main navigation — tabs at the top of this app",
        "Course_meta",
        "The **horizontal tab bar** (under the title and explainer video) switches **pages** of the app. "
        "Order: **Overview & corpus** → **Analysis & 3D embedding** → **Case lab (failure vs RAG)** → **Live retrieval inspector** → **Ask AI** → **Advanced: hybrid + LLM** → **Concepts & checklist**. "
        "Click a tab to change the main content; the **left sidebar** (Appearance, Engineering storyboard) stays visible across tabs. "
        "This mirrors the **Session 6–8 / Assignment 3** demo flow: corpus → analysis → examples → interactive retrieval → chat → advanced models → concepts.",
        "curated",
    ),
    (
        "Overview & corpus tab — what it shows",
        "Course_meta",
        "**Overview & corpus** is the **first tab**: dataset **metrics** (chunk count, vocabulary size), **bar chart** of rows by `category` (H/O/S/B), **preview** of the CSV, **full sortable table**, and **chunk length** analysis with a Plotly chart. "
        "Use it to understand **what is in the alignment-audit file** before retrieval. **ASCII pipeline** expander shows Index → Retrieve → Generate in text form.",
        "curated",
    ),
    (
        "Analysis & 3D embedding tab — meaning and plots",
        "Course_meta",
        "**Analysis & 3D embedding** is the **quantitative** tab: **TF‑IDF** geometry of chunks (intra- vs inter-category cosine, **sparsity**), **LSA** variance bars, **self-query rank** (each row’s prompt as query), **margin** (top1−top2 similarity), **heatmap** of pairwise chunk cosine, **softmax** mass over chunks, and an **interactive 3D** LSA plot with a **query beam** to top‑k neighbors. "
        "**Analysis** here means **statistical / geometric analysis of the index**, not “grading your assignment.” 3D is **visualization only**; real retrieval uses full TF‑IDF or hybrid scores in high dimensions.",
        "curated",
    ),
    (
        "Case lab tab — failure vs RAG side-by-side",
        "Course_meta",
        "**Case lab (failure vs RAG)** lets you pick a **case_id** (e.g. O01, H04). **Left column:** original **user_prompt** and **model_response** from the CSV. **Right column:** **RAG-grounded** revised text from `grounded_responses.py`. "
        "Expander shows a **simulated prompt** after retrieval. **Auto-check** reports whether TF‑IDF rank‑1 for that prompt is the same row—short prompts may not rank #1.",
        "curated",
    ),
    (
        "Live retrieval inspector tab — how it works",
        "Course_meta",
        "**Live retrieval inspector** lets you type a **free query** and see: **keyword** fallback hits, **TF‑IDF query term weights**, **cosine similarity** bar chart and **softmax** mass, **top‑k chunk text**, and optional **dominant terms** for the top hit. "
        "Use it to **debug** why a certain row matches—lexical overlap vs semantic intent.",
        "curated",
    ),
    (
        "Ask AI tab — chat and knowledge base",
        "Course_meta",
        "**Ask AI** combines (1) the **closest alignment-audit row**, (2) **SQLite** passages (IT + **Course_meta** FAQ + **Student_support**), and (3) optional **OpenAI**. "
        "Enter a question, optionally enable **OpenAI** and set **model**; **Clear chat** resets history. **3D LSA** view after a query is optional. "
        "Questions about **this app’s tabs**, **definitions**, or **wellbeing** are steered to the KB—see other Course_meta entries.",
        "curated",
    ),
    (
        "Advanced tab — hybrid retrieval and LLM grounding",
        "Course_meta",
        "**Advanced: hybrid + LLM** adds **BM25 + dense** sentence-transformer fusion, **sentence-level attributions**, **strict OpenAI** grounding on retrieved context, and related controls. "
        "If the dense model fails to load, the UI falls back to **BM25-heavy** hybrid. **Requires** more dependencies and may download models on first run.",
        "curated",
    ),
    (
        "Concepts & checklist tab",
        "Course_meta",
        "**Concepts & checklist** maps **failure modes** (helpfulness, safety, honesty, bias) to **RAG mitigations**, lists **assignment** session mapping, and **optional upgrades** implemented in the repo. "
        "Use it for **report writing** and **exam-style** explanations of why retrieval helps.",
        "curated",
    ),
    (
        "Left sidebar — Appearance and Engineering storyboard",
        "Course_meta",
        "The **left sidebar** has **Appearance** (radio **light** / **dark** theme — applies `ui_theme.py` CSS) and **Engineering storyboard** (short notes on chunking, TF‑IDF, retrieval + augmentation, **spotlight cases**). "
        "Bottom shows **corpus stats** (chunk count, vocabulary size, CSV filename). Sidebar content is **not** a separate Streamlit page; it complements whichever **main tab** is active.",
        "curated",
    ),
]
