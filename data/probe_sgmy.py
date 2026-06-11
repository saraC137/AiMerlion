"""
probe_sgmy.py — Act 2 DIAGNOSTIC probe (Step 1).

Scores the CURRENT model's CONTENT accuracy on the frozen golden set for the
SG/MY-specific patterns Act 2 must reinforce. The point is to find which
patterns are WEAK *before* generating Act 2 data, so the data targets real
gaps instead of guesses. This never trains and never touches Act 1's grade.

Patterns scored against golden ground truth:
  1. Patronymic retention  — does pred Name keep bin/binti/bte/s/o/d/o/a/l/a/p?
  2. Phone capture         — does pred Phone match the golden number (digit core)?
  3. Company suffix         — does pred Current Company keep Pte Ltd/Sdn Bhd/Berhad?
  4. Email exact            — sanity content check.

Certifications is reported as a RECALL PROXY only: golden has 0/25 certs
annotated, so true cert accuracy cannot be scored here. The proxy asks "when the
resume text mentions CMFAS/modules, did the model put anything in Certifications?"
If you want a real cert score for Act 2, the golden set needs cert annotations.

Usage:
    python probe_sgmy.py
Outputs:
    probe_sgmy.csv  — per-record pass/fail for each pattern
"""

import json, re, urllib.request

# ════════════════ EDIT HERE (match your eval_act1_format.py) ════════════════
OLLAMA_URL  = "http://localhost:11434/api/chat"
MODEL       = "aimerlion_act1_v2"     # your current Ollama model
GOLDEN      = "golden.jsonl"
OUT_CSV     = "probe_sgmy.csv"
TEMPERATURE = 0.0
NUM_CTX     = 16384

# MUST match the INSTRUCTION you trained / eval with
INSTRUCTION = """You are a resume extraction model specialized in Singapore and Malaysia resumes. Extract the candidate's information from the resume text below into a JSON object using this exact schema.

Schema (use these keys exactly; empty string "" or empty list [] when a field is absent — never invent values):
- Name, Phone, Email, Current Location, Current Company, Current Title, Summary, Function (string)
- Industry, Language Skills, Certifications, hard_skills/tags, soft_skills/skills, Achievements (lists)
- Work Experience: list of {company, title, from, to, responsibility (list)}
- Project Experience: list
- Education: list of {school, major, degree, dates}

Pay special attention to:
- Patronymics: bin/binti (Malay), s/o or d/o (Indian)
- Singapore 8-digit phones (+65 8XXX XXXX / 9XXX XXXX); Malaysia phones (+60 1X-XXX XXXX)
- Company suffixes: Pte Ltd, Sdn Bhd, Berhad
- CMFAS module numbers (Module 1A, 5, 6, 6A, 8, 8A, 9, 9A, HI) -> Certifications
- Singapore/Malaysia polytechnics and universities

Output ONLY the JSON object, no commentary."""

# ════════════════ pattern detectors ════════════════
PATRO_MARKERS = ["binti", "binte", "bin", "bte", "s/o", "d/o", "a/l", "a/p"]
CMFAS_TEXT = re.compile(r"(cmfas|module\s*\d|capital markets and financial advisory)", re.I)


def has_patronymic(s):
    t = str(s or "").lower()
    for p in PATRO_MARKERS:
        if re.search(r"(?<![a-z])" + re.escape(p) + r"(?![a-z])", t):
            return True
    return False


def company_suffixes(s):
    t = str(s or "").lower()
    found = set()
    if re.search(r"pte\.?\s*ltd", t):  found.add("Pte Ltd")
    if re.search(r"sdn\.?\s*bhd", t):  found.add("Sdn Bhd")
    if re.search(r"\bberhad\b", t):    found.add("Berhad")
    return found


def digits(s):
    return re.sub(r"\D", "", str(s or ""))


def phone_match(pred, gt):
    dp, dg = digits(pred), digits(gt)
    if not dg:
        return None
    if not dp:
        return False
    # lenient on +65/+60 prefix: match if the 8-digit core lines up
    return dp == dg or dp.endswith(dg) or dg.endswith(dp) or dp[-8:] == dg[-8:]


def norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


# ════════════════ model call (same as your eval) ════════════════
def call_model(raw_text):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": raw_text},
        ],
        "stream": False,
        "options": {"temperature": TEMPERATURE, "num_ctx": NUM_CTX},
    }
    req = urllib.request.Request(OLLAMA_URL, data=json.dumps(payload).encode("utf-8"),
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read()).get("message", {}).get("content", "")


def extract_json(text):
    text = re.sub(r"```(?:json)?", "", text or "")
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def main():
    rows = [json.loads(l) for l in open(GOLDEN, encoding="utf-8") if l.strip()]

    patro = {"hit": 0, "tot": 0, "fail": []}
    phone = {"hit": 0, "tot": 0, "fail": []}
    suffix = {"hit": 0, "tot": 0, "fail": []}
    email = {"hit": 0, "tot": 0, "fail": []}
    cert_proxy = {"hit": 0, "tot": 0, "miss": []}

    with open(OUT_CSV, "w", encoding="utf-8") as f:
        f.write("id,patronymic,phone,company_suffix,email,cert_proxy\n")
        for r in rows:
            gt = r["ground_truth"]
            pred = extract_json(call_model(r["raw_text"])) or {}

            # 1. patronymic retention (only on golden names that have one)
            p_res = ""
            if has_patronymic(gt.get("Name")):
                patro["tot"] += 1
                if has_patronymic(pred.get("Name")):
                    patro["hit"] += 1; p_res = "ok"
                else:
                    p_res = "DROPPED"
                    patro["fail"].append((r["id"], gt.get("Name"), pred.get("Name")))

            # 2. phone capture
            ph_res = ""
            pm = phone_match(pred.get("Phone"), gt.get("Phone"))
            if pm is not None:
                phone["tot"] += 1
                if pm:
                    phone["hit"] += 1; ph_res = "ok"
                else:
                    ph_res = "WRONG"
                    phone["fail"].append((r["id"], gt.get("Phone"), pred.get("Phone")))

            # 3. company suffix retention
            s_res = ""
            gt_suf = company_suffixes(gt.get("Current Company"))
            if gt_suf:
                suffix["tot"] += 1
                pred_suf = company_suffixes(pred.get("Current Company"))
                if gt_suf <= pred_suf:
                    suffix["hit"] += 1; s_res = "ok"
                else:
                    s_res = "DROPPED"
                    suffix["fail"].append((r["id"], gt.get("Current Company"), pred.get("Current Company")))

            # 4. email exact
            e_res = ""
            if gt.get("Email"):
                email["tot"] += 1
                if norm(pred.get("Email")) == norm(gt.get("Email")):
                    email["hit"] += 1; e_res = "ok"
                else:
                    e_res = "WRONG"
                    email["fail"].append((r["id"], gt.get("Email"), pred.get("Email")))

            # cert recall proxy (no golden ground truth — text-mention based)
            c_res = ""
            if CMFAS_TEXT.search(r["raw_text"]):
                cert_proxy["tot"] += 1
                got = pred.get("Certifications")
                if isinstance(got, list) and got:
                    cert_proxy["hit"] += 1; c_res = "populated"
                else:
                    c_res = "EMPTY"
                    cert_proxy["miss"].append((r["id"],))

            f.write(f'{r["id"]},{p_res},{ph_res},{s_res},{e_res},{c_res}\n')
            print(f'{r["id"]}: patro={p_res or "-"} phone={ph_res or "-"} '
                  f'suffix={s_res or "-"} email={e_res or "-"} cert_proxy={c_res or "-"}')

    def pct(d): return f'{d["hit"]}/{d["tot"]}' + (f' = {100*d["hit"]/d["tot"]:.0f}%' if d["tot"] else ' (none in golden)')

    print("\n========== ACT 2 SG/MY PROBE ==========")
    print(f"Patronymic retained : {pct(patro)}")
    print(f"Phone captured      : {pct(phone)}")
    print(f"Company suffix kept : {pct(suffix)}")
    print(f"Email exact         : {pct(email)}")
    print(f"Cert recall (proxy) : {pct(cert_proxy)}   <-- proxy only; golden has no cert annotations")

    def show(title, fails):
        if fails:
            print(f"\n  {title}:")
            for row in fails:
                if len(row) == 3:
                    print(f"    [{row[0]}]  golden={row[1]!r}  ->  pred={row[2]!r}")
                else:
                    print(f"    [{row[0]}]")

    show("Patronymics DROPPED", patro["fail"])
    show("Phones WRONG", phone["fail"])
    show("Company suffixes DROPPED", suffix["fail"])
    show("Emails WRONG", email["fail"])
    show("Cert mentions in text but Certifications EMPTY", cert_proxy["miss"])

    print("\nRead this as your Act 2 target list: whichever rows are weakest are what")
    print("the Act 2 synthetic data should over-represent.")


if __name__ == "__main__":
    main()