import io
import re
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st
import fitz  # PyMuPDF
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Employer ATS Resume Ranker", page_icon="📋", layout="wide")

st.title("📋 Employer ATS Resume Ranker")
st.caption("Bulk resume screening • Transparent job-specific scores • Recruiter remains the decision-maker")

st.warning(
    "Decision-support only. Scores are approximate indicators of resume-to-job alignment, not measures of a person's ability. "
    "Review the evidence and original resume before making hiring decisions. Do not use protected characteristics."
)

def read_pdf(data: bytes) -> str:
    text_parts = []
    with fitz.open(stream=data, filetype="pdf") as pdf:
        for page in pdf:
            text_parts.append(page.get_text("text"))
    return "\n".join(text_parts).strip()

def read_docx(data: bytes) -> str:
    doc = Document(io.BytesIO(data))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text.strip() for cell in row.cells))
    return "\n".join(parts).strip()

def extract_text(name: str, data: bytes) -> str:
    ext = Path(name).suffix.lower()
    if ext == ".pdf":
        return read_pdf(data)
    if ext == ".docx":
        return read_docx(data)
    raise ValueError("Unsupported file type. Use PDF or DOCX.")

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()

def parse_skills(raw: str):
    return list(dict.fromkeys(
        item.strip() for item in re.split(r"[,;\n]", raw or "") if item.strip()
    ))

def find_evidence(text: str, term: str, window: int = 90) -> str:
    """Return a short text excerpt around the first literal skill occurrence."""
    low = text.lower()
    needle = term.lower().strip()
    idx = low.find(needle)
    if idx < 0:
        # Handle common punctuation/spacing differences without inventing evidence.
        pattern = re.escape(needle).replace(r"\ ", r"\s+")
        match = re.search(pattern, low)
        if not match:
            return ""
        idx = match.start()
    start = max(0, idx - window // 2)
    end = min(len(text), idx + len(term) + window // 2)
    return re.sub(r"\s+", " ", text[start:end]).strip()

def split_sections(text: str):
    """Lightweight heuristic; section headings vary by resume."""
    lines = [re.sub(r"\s+", " ", x).strip() for x in text.splitlines()]
    project_lines, experience_lines = [], []
    mode = ""
    project_heads = ("project", "academic project", "personal project", "projects")
    exp_heads = ("experience", "work experience", "employment", "internship", "professional experience")
    for line in lines:
        low = line.lower().strip(" :")
        if any(low == h or low.startswith(h + " ") for h in project_heads):
            mode = "project"
            continue
        if any(low == h or low.startswith(h + " ") for h in exp_heads):
            mode = "experience"
            continue
        if low in ("education", "skills", "certifications", "achievements", "summary", "objective"):
            mode = ""
        if mode == "project":
            project_lines.append(line)
        elif mode == "experience":
            experience_lines.append(line)
    return "\n".join(project_lines), "\n".join(experience_lines)

def tfidf_similarity(job_text: str, resume_text: str) -> float:
    if not job_text.strip() or not resume_text.strip():
        return 0.0
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=8000)
        matrix = vec.fit_transform([job_text, resume_text])
        return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
    except ValueError:
        return 0.0

def score_resume(name, text, job_description, required, preferred, must_have, degree_terms,
                 min_exp_years, weights):
    clean = normalize(text)
    project_text, experience_text = split_sections(text)
    project_clean, experience_clean = normalize(project_text), normalize(experience_text)

    req_found = [s for s in required if normalize(s) in clean]
    pref_found = [s for s in preferred if normalize(s) in clean]
    missing_req = [s for s in required if s not in req_found]
    missing_must = [s for s in must_have if normalize(s) not in clean]

    # Skills score: required skills count more than preferred skills.
    req_pct = len(req_found) / len(required) if required else None
    pref_pct = len(pref_found) / len(preferred) if preferred else None
    if req_pct is not None and pref_pct is not None:
        skill_score = 100 * (0.8 * req_pct + 0.2 * pref_pct)
    elif req_pct is not None:
        skill_score = 100 * req_pct
    elif pref_pct is not None:
        skill_score = 100 * pref_pct
    else:
        skill_score = 0.0

    project_terms = required + preferred
    project_hits = [s for s in project_terms if normalize(s) in project_clean]
    project_score = 100 * len(project_hits) / len(project_terms) if project_terms else 0.0

    exp_hits = [s for s in project_terms if normalize(s) in experience_clean]
    experience_score = 100 * len(exp_hits) / len(project_terms) if project_terms else 0.0
    # Do not infer years of experience from dates; this is an evidence signal, not a duration estimate.

    education_hits = [d for d in degree_terms if normalize(d) in clean]
    education_score = 100.0 if not degree_terms else (100.0 if education_hits else 0.0)

    align = 100 * tfidf_similarity(job_description, text)

    # Use the configured weights, renormalized if a component has no usable input.
    components = {
        "Skills score": (skill_score, weights["skills"], bool(required or preferred)),
        "Project evidence": (project_score, weights["projects"], bool(project_terms)),
        "Experience evidence": (experience_score, weights["experience"], bool(project_terms)),
        "Education match": (education_score, weights["education"], bool(degree_terms)),
        "Job description similarity": (align, weights["alignment"], bool(job_description.strip())),
    }
    active_weight = sum(w for _, w, active in components.values() if active)
    total = sum(score * w for score, w, active in components.values() if active) / active_weight if active_weight else 0.0

    eligible = not missing_must
    if min_exp_years > 0:
        # Experience duration is not reliably parsed in this baseline. Require manual recruiter verification.
        exp_check = "Manual verification required"
    else:
        exp_check = "Not configured"

    evidence = []
    for skill in required + preferred:
        excerpt = find_evidence(text, skill)
        if excerpt:
            evidence.append(f"{skill}: “{excerpt}”")
    return {
        "Candidate": name,
        "Match score": round(total, 1),
        "Skills score": round(skill_score, 1),
        "Project evidence score": round(project_score, 1),
        "Experience evidence score": round(experience_score, 1),
        "Education match score": round(education_score, 1),
        "Job alignment score": round(align * 100, 1),
        "Required skills found": ", ".join(req_found) or "None detected",
        "Missing required skills": ", ".join(missing_req) or "None detected",
        "Must-have check": "Pass (text match)" if eligible else "Review: missing must-have",
        "Missing must-have criteria": ", ".join(missing_must) or "None detected",
        "Degree evidence": ", ".join(education_hits) or ("Not detected" if degree_terms else "Not configured"),
        "Experience check": exp_check,
        "Evidence excerpts": "\n".join(evidence[:12]) or "No configured skill evidence found",
        "Recruiter decision": "Not reviewed",
        "Recruiter notes": "",
        "_eligible": eligible,
        "_text": text,
    }

with st.sidebar:
    st.header("1. Job requirements")
    job_title = st.text_input("Job title", "Python Developer — Fresher")
    job_description = st.text_area(
        "Job description / responsibilities",
        "Build and maintain Python applications and REST APIs. Work with SQL databases, Git, testing, debugging, and backend frameworks such as FastAPI or Django.",
        height=180,
    )
    required_raw = st.text_area("Required skills (comma-separated)", "Python, SQL, Git, REST API")
    preferred_raw = st.text_area("Preferred skills (comma-separated)", "FastAPI, Django, pytest, PostgreSQL")
    must_raw = st.text_area("Mandatory criteria to check (comma-separated)", "Python")
    degree_raw = st.text_area("Accepted degree terms (optional)", "BCA, MCA, B.Tech, B.Sc")
    min_exp_years = st.number_input(
        "Minimum experience in years (manual verification)", min_value=0, max_value=30, value=0,
        help="This version does not automatically infer years from resume dates. If set above zero, it flags experience for recruiter verification rather than rejecting candidates."
    )
    st.header("2. Score weights")
    w_skills = st.slider("Technical skills", 0, 100, 40)
    w_projects = st.slider("Project evidence", 0, 100, 25)
    w_experience = st.slider("Experience evidence", 0, 100, 15)
    w_education = st.slider("Education", 0, 100, 10)
    w_alignment = st.slider("Job description alignment", 0, 100, 10)
    st.caption("Weights are normalized automatically. Set a component's weight to 0 to ignore it.")

required = parse_skills(required_raw)
preferred = parse_skills(preferred_raw)
must_have = parse_skills(must_raw)
degree_terms = parse_skills(degree_raw)
weights = {
    "skills": w_skills,
    "projects": w_projects,
    "experience": w_experience,
    "education": w_education,
    "alignment": w_alignment,
}

st.subheader("Upload resumes")
st.write("Select multiple PDF/DOCX files, or upload one ZIP containing resumes. Password-protected, scanned-image-only, or unreadable PDFs may need OCR or manual review.")
uploads = st.file_uploader(
    "Resume files",
    type=["pdf", "docx", "zip"],
    accept_multiple_files=True,
)

def unpack_uploads(uploaded):
    items = []
    errors = []
    for file in uploaded or []:
        name = file.name
        data = file.getvalue()
        if name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    for member in zf.infolist():
                        if member.is_dir():
                            continue
                        ext = Path(member.filename).suffix.lower()
                        if ext in (".pdf", ".docx"):
                            try:
                                items.append((Path(member.filename).name, zf.read(member)))
                            except Exception as e:
                                errors.append(f"{member.filename}: {e}")
            except Exception as e:
                errors.append(f"{name}: Could not read ZIP ({e})")
        else:
            items.append((name, data))
    # Avoid duplicate names silently overwriting one another.
    seen = {}
    final = []
    for name, data in items:
        seen[name] = seen.get(name, 0) + 1
        label = name if seen[name] == 1 else f"{Path(name).stem}_{seen[name]}{Path(name).suffix}"
        final.append((label, data))
    return final, errors

if uploads:
    files, unpack_errors = unpack_uploads(uploads)
    for error in unpack_errors:
        st.error(error)

    rows, parse_errors = [], []
    progress = st.progress(0)
    status = st.empty()
    for i, (name, data) in enumerate(files):
        status.write(f"Processing {i+1} of {len(files)}: {name}")
        try:
            text = extract_text(name, data)
            if len(text.strip()) < 40:
                parse_errors.append(f"{name}: Very little text extracted. It may be scanned, image-only, or unreadable.")
                continue
            row = score_resume(name, text, job_description, required, preferred, must_have, degree_terms, min_exp_years, weights)
            rows.append(row)
        except Exception as e:
            parse_errors.append(f"{name}: Could not parse file ({e})")
        progress.progress((i + 1) / max(len(files), 1))
    status.empty()
    progress.empty()

    for error in parse_errors:
        st.warning(error)

    if rows:
        df = pd.DataFrame(rows).sort_values(["Match score", "Candidate"], ascending=[False, True]).reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df) + 1))
        df["Shortlist status"] = [
            "Review required" if not eligible else ("Top 20 — review" if rank <= 20 else "Not in top 20")
            for rank, eligible in zip(df["Rank"], df["_eligible"])
        ]
        # Do not expose extracted full resume text in the main table or exports.
        visible_cols = [
            "Rank", "Candidate", "Match score", "Skills score", "Project evidence score",
            "Experience evidence score", "Education match score", "Job alignment score",
            "Required skills found", "Missing required skills", "Must-have check",
            "Missing must-have criteria", "Degree evidence", "Experience check",
            "Shortlist status", "Recruiter decision", "Recruiter notes", "Evidence excerpts"
        ]

        st.subheader("Results")
        c1, c2, c3 = st.columns(3)
        c1.metric("Resumes processed", len(rows))
        c2.metric("Top 20 shortlist", min(20, len(rows)))
        c3.metric("Flagged for eligibility review", int((~df["_eligible"]).sum()))

        st.info("The top 20 is a score-based review queue, not an automatic hiring decision. Candidates with missing mandatory criteria remain visible and are flagged for review.")
        st.markdown("### Top 20 candidates")
        st.dataframe(df.head(20)[visible_cols], use_container_width=True, hide_index=True)

        st.markdown("### Full ranked list")
        st.caption("Edit the recruiter decision and notes in the review table. Decisions are saved in this session and included in the export.")
        review_cols = ["Rank", "Candidate", "Match score", "Must-have check", "Shortlist status", "Recruiter decision", "Recruiter notes"]
        edited = st.data_editor(
            df[review_cols],
            use_container_width=True,
            hide_index=True,
            disabled=["Rank", "Candidate", "Match score", "Must-have check", "Shortlist status"],
            column_config={
                "Recruiter decision": st.column_config.SelectboxColumn(
                    "Recruiter decision",
                    options=["Not reviewed", "Advance", "Hold", "Reject", "Needs clarification"],
                    required=True,
                ),
                "Recruiter notes": st.column_config.TextColumn("Recruiter notes", width="large"),
            },
            key="review_editor",
        )
        decision_map = edited.set_index("Candidate")[["Recruiter decision", "Recruiter notes"]]
        for idx, row in df.iterrows():
            candidate = row["Candidate"]
            if candidate in decision_map.index:
                df.at[idx, "Recruiter decision"] = decision_map.at[candidate, "Recruiter decision"]
                df.at[idx, "Recruiter notes"] = decision_map.at[candidate, "Recruiter notes"]

        st.markdown("### Candidate evidence")
        selected = st.selectbox("Choose a candidate to inspect", df["Candidate"].tolist())
        selected_row = df[df["Candidate"] == selected].iloc[0]
        st.write(f"**Rank:** {selected_row['Rank']}  ·  **Match score:** {selected_row['Match score']}/100")
        st.write(f"**Must-have check:** {selected_row['Must-have check']}")
        st.write(f"**Missing required skills:** {selected_row['Missing required skills']}")
        st.write("**Evidence excerpts from the resume**")
        st.text(selected_row["Evidence excerpts"])
        with st.expander("View extracted resume text (check for extraction errors)"):
            st.text(selected_row["_text"])

        export_df = df[visible_cols].copy()
        csv = export_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Download full ranked report (CSV)",
            data=csv,
            file_name="ats_ranked_candidates.csv",
            mime="text/csv",
            type="primary",
        )
        top_csv = df.head(20)[visible_cols].to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Download top 20 shortlist (CSV)",
            data=top_csv,
            file_name="ats_top_20_shortlist.csv",
            mime="text/csv",
        )
    else:
        st.error("No readable resumes were processed. Try text-based PDFs or DOCX files. Scanned PDFs need OCR support.")
else:
    st.markdown("### What you'll get")
    st.markdown(
        "- Ranked list of all readable candidates\n"
        "- Top 20 review queue\n"
        "- Per-criterion scores and missing-skill flags\n"
        "- Short evidence excerpts from each resume\n"
        "- Recruiter decision and notes\n"
        "- CSV exports for the full ranking and shortlist"
    )
    st.caption("For a first test, upload 3–5 resumes you have permission to process and verify the extracted text and scores before using a larger batch.")
