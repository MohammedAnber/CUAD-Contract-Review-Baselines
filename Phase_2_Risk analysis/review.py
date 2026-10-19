"""
POST /documents/review/stream

Accepts an uploaded contract (PDF, TXT, DOCX), segments candidate clauses
using broad legal signals, then uses Ollama to BOTH classify the clause type
AND perform risk analysis in a single combined prompt per section.

This avoids hard-coded keyword→category mapping and lets the model identify
the actual legal nature of each section.

Location: app/api/routes/review.py
"""

from __future__ import annotations

import asyncio
import json
import queue
import re
import shutil
import tempfile
import time
from pathlib import Path
from threading import Thread

import requests
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/documents", tags=["review"])

ALLOWED_SUFFIXES = {".pdf", ".txt", ".doc", ".docx"}

OLLAMA_URL = "http://127.0.0.1:11434"
MODEL = "qwen2.5:3b"

# ── Clause taxonomy ──────────────────────────────────────────────────────────
# All recognised legal clause types the model can pick from.
KNOWN_CLAUSE_TYPES: list[str] = [
    "Cap On Liability",
    "Anti-Assignment",
    "Termination",
    "IP Ownership & Licensing",
    "Warranty & Disclaimers",
    "Confidentiality",
    "Non-Compete",
    "Liquidated Damages",
    "Audit Rights",
    "Change Of Control",
    "Minimum Commitment",
    "Price Restrictions",
]

# Broad legal-signal keywords — used ONLY to score candidate sections.
# NOT used to assign clause types (that's the AI's job now).
_LEGAL_SIGNALS: list[str] = [
    "shall", "agree", "party", "parties", "licens", "assign",
    "terminat", "liabilit", "warrant", "confidential", "intellectu",
    "proprietary", "indemnif", "govern", "jurisdict", "remedy",
    "obligation", "right", "title", "interest", "damages",
    "represent", "covenant", "consent", "notice", "effect",
]


# ── Text extraction ──────────────────────────────────────────────────────────

def _extract_text(path: Path, suffix: str) -> str:
    if suffix == ".pdf":
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages)
        except Exception:
            pass
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            pass
        raise RuntimeError("Could not extract text from PDF.")

    if suffix in (".doc", ".docx"):
        try:
            import docx
            doc = docx.Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            raise RuntimeError("Could not extract .docx text.")

    return path.read_text(encoding="utf-8", errors="replace")


# ── Candidate section selection ──────────────────────────────────────────────

def _score_section(text_lower: str) -> int:
    """Count how many distinct legal signals appear in this section."""
    return sum(1 for sig in _LEGAL_SIGNALS if sig in text_lower)


def _get_candidate_sections(text: str, max_candidates: int = 8) -> list[str]:
    """
    Split the contract into structural sections and return the top N
    that look most like substantive legal clauses (by legal-signal score).

    No clause-type assignment happens here — classification is fully
    delegated to the AI in _classify_and_analyze().
    """
    # Split on common contract section boundaries
    raw_sections = re.split(
        r"\n(?=\d+[\.\)]\s+[A-Z]"            # "1. TERMINATION"
        r"|[A-Z][A-Z\s]{3,}(?:\.|:|\n)"      # "CONFIDENTIALITY."
        r"|EXHIBIT [A-Z]"                     # "EXHIBIT A"
        r"|SCHEDULE [A-Z0-9])",               # "SCHEDULE 1"
        text,
    )

    scored: list[tuple[int, str]] = []
    for sec in raw_sections:
        sec = sec.strip()
        # Skip too-short (headings) or too-long (entire agreement bodies)
        if not (80 <= len(sec) <= 4000):
            continue
        score = _score_section(sec.lower())
        if score >= 3:                        # Must have at least 3 signals
            scored.append((score, sec))

    # Sort descending by score, deduplicate near-identical sections
    scored.sort(key=lambda x: -x[0])
    seen_prefixes: set[str] = set()
    unique: list[str] = []
    for _, sec in scored:
        prefix = sec[:60].lower()
        if prefix not in seen_prefixes:
            seen_prefixes.add(prefix)
            unique.append(sec)
        if len(unique) >= max_candidates:
            break

    return unique


# ── JSON repair ──────────────────────────────────────────────────────────────

def _repair_json(raw: str) -> dict:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        candidate = m.group(0)
        open_braces = candidate.count("{") - candidate.count("}")
        candidate += "}" * max(open_braces, 0)
        try:
            return json.loads(candidate)
        except Exception:
            pass
    return {}


# ── AI classify + analyze (single combined call) ─────────────────────────────

def _classify_and_analyze(
    section_text: str,
    section_index: int,
    publish_heartbeat: callable = None,
) -> dict:
    """
    Single Ollama call that:
      1. Classifies the clause type (from KNOWN_CLAUSE_TYPES, or NONE)
      2. Performs full risk analysis from the Client perspective

    Returns a dict with keys:
        clause_type, risk_severity, client_impact, red_flag_quote, recommended_fix
    """
    clause_list = " | ".join(KNOWN_CLAUSE_TYPES) + " | NONE"

    prompt = f"""You are a senior US commercial contract attorney reviewing from the Client (Licensee/Buyer) perspective.

Step 1 — Identify the clause type. Pick EXACTLY one from:
{clause_list}

Step 2 — Analyze the legal risk for the Client.

Section Text:
\"\"\"{section_text[:2500]}\"\"\"

Return ONLY a JSON object with these five keys:
{{
  "clause_type": "<one of the types above, or NONE>",
  "risk_severity": "CRITICAL_RISK | HIGH_RISK | MEDIUM_RISK | LOW_RISK",
  "client_impact": "<direct practical explanation of what the Client risks or loses>",
  "red_flag_quote": "<verbatim excerpt from the text that proves the risk, or empty string>",
  "recommended_fix": "<specific negotiation change or fallback language the Client should demand>"
}}
If clause_type is NONE, set risk_severity to LOW_RISK and leave the other fields as empty strings.
Return ONLY the JSON, no other text."""

    try:
        t0 = time.time()
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": True,
                "format": "json",
                "options": {
                    "temperature": 0,
                    "num_predict": 350,
                    "num_thread": 8,
                },
            },
            timeout=200,
            stream=True,
        )

        accumulated: list[str] = []
        token_count = 0
        last_heartbeat = time.time()

        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line.decode("utf-8"))
            tok = chunk.get("response", "")
            accumulated.append(tok)
            token_count += 1

            now = time.time()
            if publish_heartbeat and (now - last_heartbeat >= 2.0):
                last_heartbeat = now
                publish_heartbeat({
                    "type": "heartbeat",
                    "clause": f"Section {section_index}",
                    "tokens": token_count,
                    "elapsed": round(now - t0, 1),
                })

            if chunk.get("done", False):
                break

        raw = "".join(accumulated).strip()
        parsed = _repair_json(raw)

        # Validate clause_type against known list (accept "NONE" too)
        ct = parsed.get("clause_type", "NONE")
        if ct not in KNOWN_CLAUSE_TYPES and ct != "NONE":
            # Try fuzzy match — model might add punctuation or extra words
            ct_lower = ct.lower()
            matched = next(
                (k for k in KNOWN_CLAUSE_TYPES if k.lower() in ct_lower or ct_lower in k.lower()),
                "NONE",
            )
            parsed["clause_type"] = matched

        if parsed.get("risk_severity") and parsed.get("clause_type"):
            return parsed

    except Exception as exc:
        print(f"Error in section {section_index}: {exc}")

    return {
        "clause_type": "NONE",
        "risk_severity": "LOW_RISK",
        "client_impact": "",
        "red_flag_quote": "",
        "recommended_fix": "",
    }


# ── Main review pipeline ─────────────────────────────────────────────────────

_SEV_ORDER = {"CRITICAL_RISK": 4, "HIGH_RISK": 3, "MEDIUM_RISK": 2, "LOW_RISK": 1}


def _run_review(path: Path, filename: str, publish: callable, events: queue.Queue) -> None:
    try:
        suffix = path.suffix.lower()
        publish({"type": "progress", "stage": "extract",
                 "message": f"Extracting text from {filename}…"})

        try:
            text = _extract_text(path, suffix)
        except RuntimeError as exc:
            publish({"type": "error", "status": 422, "message": str(exc)})
            return

        if not text or len(text.strip()) < 100:
            publish({"type": "error", "status": 422,
                     "message": "Uploaded file is empty or unreadable."})
            return

        publish({"type": "progress", "stage": "segment",
                 "message": "Selecting candidate clauses for AI classification…"})
        candidates = _get_candidate_sections(text, max_candidates=8)

        if not candidates:
            publish({"type": "error", "status": 422,
                     "message": "No recognizable legal sections found in this document."})
            return

        publish({
            "type": "progress",
            "stage": "segment",
            "message": f"Found {len(candidates)} candidate sections — AI will classify and analyze each.",
            "total": len(candidates),
        })

        audit_results: list[dict] = []
        seen_types: set[str] = set()
        MAX_OUTPUT_CLAUSES = 5

        for i, section_text in enumerate(candidates, 1):
            # Stop once we have enough valid classified clauses
            if len(audit_results) >= MAX_OUTPUT_CLAUSES:
                break

            first_line = section_text.split("\n")[0].strip()[:80]
            publish({
                "type": "progress",
                "stage": "audit",
                "message": f"Classifying & auditing section [{i}/{len(candidates)}]: {first_line}…",
                "current": i,
                "total": len(candidates),
                "clause_type": "Analyzing…",
            })

            t0 = time.time()
            result = _classify_and_analyze(section_text, i, publish_heartbeat=publish)
            elapsed = round(time.time() - t0, 1)

            ct = result.get("clause_type", "NONE")
            sev = result.get("risk_severity", "LOW_RISK")

            if ct == "NONE":
                publish({
                    "type": "progress",
                    "stage": "audit",
                    "message": f"Section [{i}] → not a tracked clause type, skipping.",
                    "current": i,
                    "total": len(candidates),
                })
                continue

            if ct in seen_types:
                # Keep the higher-severity version
                existing_idx = next(
                    (j for j, r in enumerate(audit_results) if r["clause_type"] == ct), None
                )
                if existing_idx is not None:
                    existing_sev = audit_results[existing_idx]["analysis"].get("risk_severity", "LOW_RISK")
                    if _SEV_ORDER.get(sev, 0) > _SEV_ORDER.get(existing_sev, 0):
                        audit_results[existing_idx] = {
                            "clause_index": existing_idx + 1,
                            "section_title": first_line,
                            "clause_type": ct,
                            "clause_text": section_text,
                            "analysis": result,
                        }
                publish({
                    "type": "progress",
                    "stage": "audit",
                    "message": f"Section [{i}] → {ct} already found, kept higher severity.",
                    "current": i,
                    "total": len(candidates),
                })
                continue

            seen_types.add(ct)
            audit_results.append({
                "clause_index": len(audit_results) + 1,
                "section_title": first_line,
                "clause_type": ct,
                "clause_text": section_text,
                "analysis": result,
            })

            publish({
                "type": "progress",
                "stage": "audit",
                "message": f"✓ [{i}/{len(candidates)}] {ct} → {sev} ({elapsed}s)",
                "current": i,
                "total": len(candidates),
                "clause_type": ct,
            })

        if not audit_results:
            publish({"type": "error", "status": 422,
                     "message": "AI could not classify any recognizable legal clauses in this contract."})
            return

        # Sort by severity descending
        audit_results.sort(
            key=lambda r: -_SEV_ORDER.get(r["analysis"].get("risk_severity", "LOW_RISK"), 0)
        )
        # Re-number after sort
        for idx, r in enumerate(audit_results, 1):
            r["clause_index"] = idx

        top_sev = audit_results[0]["analysis"].get("risk_severity", "HIGH_RISK")
        overall_map = {
            "CRITICAL_RISK": "CRITICAL", "HIGH_RISK": "HIGH",
            "MEDIUM_RISK": "MEDIUM", "LOW_RISK": "LOW",
        }
        overall_risk = overall_map.get(top_sev, "HIGH")

        critical_count = sum(1 for r in audit_results if "CRITICAL" in r["analysis"].get("risk_severity", ""))
        high_count = sum(1 for r in audit_results if r["analysis"].get("risk_severity") == "HIGH_RISK")

        publish({
            "type": "result",
            "data": {
                "filename": filename,
                "total_clauses": len(audit_results),
                "overall_risk": overall_risk,
                "critical_count": critical_count,
                "high_count": high_count,
                "clauses": audit_results,
            },
        })

    except Exception as exc:
        publish({"type": "error", "status": 500, "message": f"Server processing error: {exc}"})
    finally:
        path.unlink(missing_ok=True)
        events.put(None)


# ── FastAPI endpoint ─────────────────────────────────────────────────────────

@router.post("/review/stream", include_in_schema=True)
async def review_contract_stream(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Upload a .pdf, .txt, or .docx file.",
        )

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    filename = file.filename or tmp_path.name
    events: queue.Queue[dict | None] = queue.Queue()

    def publish(event: dict) -> None:
        events.put(event)

    Thread(target=_run_review, args=(tmp_path, filename, publish, events), daemon=True).start()

    async def stream():
        yield "data: " + json.dumps({
            "type": "progress", "stage": "upload",
            "message": "Upload received — AI classification starting…",
        }) + "\n\n"
        while True:
            event = await asyncio.to_thread(events.get)
            if event is None:
                break
            yield "data: " + json.dumps(event) + "\n\n"
            if event.get("type") in ("result", "error"):
                break

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
