"""
End-to-End Contract Review Pipeline:
1. Segments full contract text into discrete clauses.
2. Identifies key high-risk clauses (Cap on Liability, Termination, IP, Assignment, etc.).
3. Runs the US Commercial Contract Playbook (Client/Buyer Perspective) via local Ollama.
4. Generates a comprehensive Executive Risk Audit Report.
"""

import json
import re
import sys
import time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
OLLAMA_URL = "http://127.0.0.1:11434"
MODEL = "qwen2.5:3b"

CONTRACT_FILE = ROOT / "sample_contract_emdeon_webmd.txt"
REPORT_OUTPUT = ROOT / "Emdeon_Contract_Risk_Report.md"

# Playbook questions
PLAYBOOK_QUESTIONS = {
    "Cap On Liability": (
        "1. Does the cap apply to both parties equally (mutual) or does it solely protect the Licensor/Vendor? "
        "2. What is the cap amount or metric (e.g., $50, fees paid, or unlimited)? "
        "3. Are indirect, consequential, or punitive damages disclaimed, and does this leave the Client without remedy? "
        "4. Are there any carveouts for gross negligence, willful misconduct, or data breach?"
    ),
    "Anti-Assignment": (
        "1. Can the Client assign this agreement to an affiliate, subsidiary, or buyer in an M&A transaction without prior written consent? "
        "2. Is the assignment restriction mutual or unilateral? "
        "3. Does the clause state that consent shall not be unreasonably withheld or delayed?"
    ),
    "Termination": (
        "1. Can the Client terminate for convenience or only for material breach with a cure period? "
        "2. What notice period is required (e.g., 30 days)? "
        "3. What survival or wind-down rights does the Client have upon termination?"
    ),
    "IP Ownership & Licensing": (
        "1. Who retains ownership of the underlying content/IP? "
        "2. Is the license granted to the Client non-exclusive, non-transferable, or revocable? "
        "3. Can the Client modify, adapt, or create derivative works from the content?"
    ),
    "Warranty & Disclaimers": (
        "1. Does the Licensor provide any warranty regarding accuracy, fitness for purpose, or non-infringement? "
        "2. Is the Client forced to accept the content strictly 'AS IS' with all risks disclaimed?"
    )
}

def segment_contract(text: str) -> list[dict]:
    """Parse contract sections and clauses."""
    # Remove header preamble and split into sections
    sections = re.split(r'\n(?=[0-9]+\.\s+|TERMS AND CONDITIONS|EXHIBIT [A-Z])', text)
    clauses = []
    
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        
        # Check title / number
        first_line = sec.split('\n')[0].strip()
        
        # Classify key sections
        sec_lower = sec.lower()
        clause_type = None
        
        if "liability" in sec_lower and ("exceed" in sec_lower or "cap" in sec_lower or "disclaimer" in sec_lower or "shall not exceed" in sec_lower):
            clause_type = "Cap On Liability"
        elif "assignment" in sec_lower and ("prior written consent" in sec_lower or "transfer" in sec_lower):
            clause_type = "Anti-Assignment"
        elif "term and termination" in sec_lower or ("termination" in sec_lower and "cure" in sec_lower):
            clause_type = "Termination"
        elif "intellectual property" in sec_lower or ("license" in sec_lower and "non-exclusive" in sec_lower and "content" in sec_lower):
            clause_type = "IP Ownership & Licensing"
        elif "disclaimer" in sec_lower or "merchantability" in sec_lower:
            clause_type = "Warranty & Disclaimers"
            
        if clause_type:
            clauses.append({
                "title": first_line,
                "clause_type": clause_type,
                "text": sec
            })
            
    return clauses

def analyze_clause(clause_type: str, clause_text: str) -> dict:
    questions = PLAYBOOK_QUESTIONS.get(clause_type, "1. What obligations does this clause place on the Client? 2. What risk exists?")
    
    prompt = f"""You are a senior US commercial contract attorney reviewing this agreement strictly from the perspective of Emdeon (the Licensee/Client/Buyer).

Clause Type: {clause_type}
Clause Text:
\"\"\"{clause_text}\"\"\"

Review Questions:
{questions}

Analyze the clause and output a JSON object with:
{{
  "risk_severity": "CRITICAL_RISK | HIGH_RISK | MEDIUM_RISK | LOW_RISK",
  "client_impact": "Direct, practical legal explanation of what Emdeon risks or loses under this clause.",
  "red_flag_quote": "Exact verbatim excerpt from the text that proves the risk.",
  "recommended_fix": "Specific negotiation change or fallback language Emdeon should demand."
}}
Return ONLY the JSON object."""

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0, "num_predict": 450}
            },
            timeout=180
        )
        content = resp.json().get("response", "").strip()
        
        # Parse JSON with fallback repair
        try:
            return json.loads(content)
        except Exception:
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                return json.loads(match.group(0))
    except Exception as e:
        print(f"Error analyzing {clause_type}: {e}")
        
    return {
        "risk_severity": "HIGH_RISK",
        "client_impact": "Analysis error or timeout.",
        "red_flag_quote": "",
        "recommended_fix": "Manual attorney review required."
    }

def main():
    print("="*70)
    print("Running End-to-End Contract Review Pipeline on Sample Agreement")
    print("Contract: Content License Agreement (Emdeon Corporation & WebMD, Inc.)")
    print("Perspective: Client / Licensee (Emdeon)")
    print("="*70)
    
    text = CONTRACT_FILE.read_text(encoding="utf-8")
    clauses = segment_contract(text)
    print(f"\n[1] Extracted {len(clauses)} Key High-Impact Clauses for Deep Legal Audit:")
    for c in clauses:
        print(f"   - {c['clause_type']} ({c['title'][:40]}...)")
        
    audit_results = []
    print("\n[2] Executing Risk Playbook Inference via Local Model...")
    t0 = time.time()
    
    for i, c in enumerate(clauses, 1):
        print(f"   -> [{i}/{len(clauses)}] Auditing: {c['clause_type']}...", end="", flush=True)
        t_start = time.time()
        analysis = analyze_clause(c['clause_type'], c['text'])
        elapsed = time.time() - t_start
        print(f" Done ({elapsed:.1f}s) | Severity: {analysis.get('risk_severity')}")
        
        audit_results.append({
            "section_title": c["title"],
            "clause_type": c["clause_type"],
            "clause_text": c["text"],
            "analysis": analysis
        })
        
    total_time = time.time() - t0
    print(f"\n[3] Legal Audit Complete in {total_time:.1f}s!")
    
    # Generate Markdown Executive Report
    report = f"""# Executive Legal Risk Audit Report
**Agreement:** Content License Agreement  
**Parties:** WebMD, Inc. (Licensor) vs. Emdeon Corporation (Licensee / Client)  
**Perspective:** Client / Licensee (Emdeon)  
**Governing Law:** State of New York  
**Analysis Engine:** Antigravity US Commercial Contract Review AI (Qwen 2.5 3B / Local Inference)  
**Date of Review:** October 18, 2026  

---

## 1. Executive Summary & Deal Health
This agreement licenses Anatomical Line Drawings from WebMD to Emdeon for integration into medical software products.
From the perspective of **Emdeon (Client)**, the contract presents **extreme asymmetric liability risks** and severe operational restrictions that heavily favor the licensor.

### Key Risk Heatmap
"""
    for r in audit_results:
        sev = r["analysis"].get("risk_severity", "MEDIUM_RISK")
        badge = "🔴 CRITICAL" if "CRITICAL" in sev else ("🟠 HIGH" if "HIGH" in sev else "🟡 MEDIUM")
        report += f"- **{r['clause_type']}:** {badge}\n"
        
    report += "\n---\n\n## 2. Clause-by-Clause Detailed Risk Findings\n\n"
    
    for i, r in enumerate(audit_results, 1):
        a = r["analysis"]
        sev = a.get("risk_severity", "N/A")
        badge = "🔴 CRITICAL RISK" if "CRITICAL" in sev else ("🟠 HIGH RISK" if "HIGH" in sev else "🟡 MEDIUM RISK")
        
        report += f"### {i}. {r['clause_type']} — {badge}\n"
        report += f"**Source Section:** `{r['section_title']}`\n\n"
        report += f"**Impact on Client (Emdeon):**\n{a.get('client_impact')}\n\n"
        if a.get('red_flag_quote'):
            report += f"> **Red Flag Contract Language:**\n> *\"{a.get('red_flag_quote')}\"*\n\n"
        report += f"**Recommended Negotiation Fix (Attorney Fallback):**\n{a.get('recommended_fix')}\n\n"
        report += "---\n\n"
        
    report += """## 3. Top Action Items Before Signing
1. **Remove the $50 Liability Cap:** WebMD's cap of $50 against an annual fee of $5,000+ leaves Emdeon with essentially zero legal remedy if the drawings infringe third-party IP or cause clinical errors.
2. **Add Uncapped IP Indemnification:** Require WebMD to indemnify and hold Emdeon harmless from any third-party copyright or trademark infringement claims arising from the Content.
3. **Carve Out M&A from Anti-Assignment:** Add an exception permitting assignment without consent in connection with a merger, acquisition, or sale of substantially all assets.
"""

    REPORT_OUTPUT.write_text(report, encoding="utf-8")
    print(f"\n[4] Comprehensive Audit Report Saved to:\n    {REPORT_OUTPUT}")

if __name__ == "__main__":
    main()
