# Employer ATS Resume Ranker

A local, beginner-friendly Streamlit app that extracts text from PDF/DOCX resumes, scores alignment against a job description, ranks candidates, provides evidence snippets, supports recruiter review, and exports CSV reports.

## Requirements
- Windows 10/11
- Python 3.10–3.12 recommended
- Internet only for installing packages; resume processing runs locally after setup.

## Setup (Windows / VS Code)

1. Extract this ZIP to a folder, e.g. `C:\ATS-Resume-Ranker`.
2. Open that folder in VS Code.
3. Open Terminal → New Terminal.
4. Run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

If PowerShell blocks activation, use Command Prompt or run:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

5. The app opens in your browser, usually at `http://localhost:8501`.
6. Enter the job description and skill criteria in the sidebar.
7. Upload multiple resumes or a ZIP containing PDF/DOCX resumes.

## How scoring works
Default weights:
- Skills: 40%
- Project evidence: 25%
- Experience evidence: 15%
- Education: 10%
- Job description similarity: 10%

Weights are normalized when some criteria are not configured. These are illustrative defaults, not validated hiring science. Exact skill matches and TF-IDF text similarity are simple baseline techniques; they can miss synonyms, context, formatting differences, and genuine skill depth.

## Eligibility and limitations
- Must-have checks are literal text checks. A missing match means "not detected", not proof that the person lacks the skill.
- Minimum years of experience is not automatically calculated; it is flagged for human verification.
- Degree terms are searched in resume text; unusual abbreviations and equivalent qualifications may be missed.
- Scanned/image-only PDFs need OCR, which is not included in this version.
- Ranking is not an automated hiring decision. Recruiters should inspect evidence and the original resume, correct parsing errors, and provide an accessible review route.
- Do not score or filter based on protected characteristics or proxies. Limit access to resumes, define a retention policy, and obtain appropriate authorization to process candidate data.

## Next upgrades
1. OCR for scanned PDFs.
2. Better skill synonym dictionary and semantic embeddings.
3. Database-backed job and candidate records.
4. Authentication, access control, audit logs, and secure deployment.
5. Recruiter feedback and bias/validity evaluation before any real hiring use.
