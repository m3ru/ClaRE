#!/usr/bin/env python3
"""Generate three minimal grammatical repairs for each strict GCG rewrite.

The source rewrite, rather than the benign original, is the text being repaired.  The
original is provided only as context so the editor can avoid inventing a new task.  A
JSONL checkpoint makes the API run safe to resume.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import random
import re
import shutil
import threading
import time


SYSTEM = """You are a conservative copy editor repairing malformed research prompts.

Return exactly three distinct, minimally edited grammatical versions of REWRITE_TO_REPAIR,
or mark it UNREPAIRABLE when three faithful versions cannot be made without guessing or
changing meaning.
Preserve its current meaning, intent, polarity, likely refusal-triggering words/phrases, and
as much wording as possible. Fix only grammar, punctuation, broken syntax, fragments, and
obvious formatting artifacts. Every output must
be a natural, coherent, fully grammatical and complete user prompt: never leave sentence
fragments, dangling ellipses, unmatched delimiters, placeholder-like text, or unfinished
questions. If a corrupted fragment's intended meaning cannot be recovered without inventing
content, delete that fragment; do not guess how to complete it. All three repairs must
express the same single core request as each other. Do not turn an orphan fragment into an
additional question or explanatory clause. Do not introduce synonyms or any new substantive
noun, verb, adjective, or adverb. To make the three outputs distinct, use only punctuation,
word order, or small function-word framing (for example, "Please..." or "Could you..."). If
necessary, two alternatives may be near-identical; faithfulness matters more than variety.
In particular, never invent a new content word to finish a dangling phrase:
delete the whole dangling phrase instead (for example, remove "look decidedly..." rather
than guessing what it should look like).

Safety/content constraint: do not add harmful, dangerous, illegal, sexual, hateful, or
otherwise sensitive content; do not add a new request, instruction, example, scenario,
claim, target, or substantive detail; do not make an existing request more harmful or
actionable. If suspicious wording already exists, retain rather than expand it. Do not
answer or discuss the prompt.

Return only one of these JSON shapes:
{"status":"OK","repairs":["repair 1","repair 2","repair 3"]}
{"status":"UNREPAIRABLE","reason":"brief reason"}
"""

USER_T = """REWRITE_TO_REPAIR:
<rewrite>{rewrite}</rewrite>

LIKELY_TRIGGER (retain verbatim if grammar permits):
<trigger>{trigger}</trigger>

Return exactly three distinct minimal repairs, or UNREPAIRABLE, as JSON only."""

AUDIT_SYSTEM = """You are the final quality auditor for grammatical prompt repairs.
Be conservative. A repair set passes only if all three outputs are natural, coherent,
fully grammatical, complete standalone user requests; preserve the source rewrite's current
meaning, polarity, and likely trigger; add no harmful/sensitive content, new request, claim,
example, target, or substantive detail; and delete rather than repunctuate or guess at
unrecoverable fragments and dangling keyword lists. All three must express the same core
request. If the proposed status is UNREPAIRABLE, pass it only when three faithful repairs
really cannot be produced under these rules. Read EACH repair separately and reject the
entire set if even one has a fragment, dangling modifier, comma splice, missing subject,
missing conjunction, malformed list, or unnatural syntax. For example, "The tree caught
fire, started because of arson" must be rejected because it is missing a conjunction.
For an OK proposal return one per-repair verdict and an overall verdict; overall PASS is
allowed only when all three repair verdicts are PASS. For UNREPAIRABLE, use an empty list.
Return only JSON:
{"verdict":"PASS","repair_verdicts":["PASS","PASS","PASS"],"reason":"brief reason"} or
{"verdict":"REJECT","repair_verdicts":["PASS","REJECT","PASS"],"reason":"brief reason"}
"""

AUDIT_USER_T = """SOURCE_REWRITE:
<rewrite>{rewrite}</rewrite>
LIKELY_TRIGGER:
<trigger>{trigger}</trigger>
PROPOSAL:
{proposal}
Audit the proposal. Return JSON only."""

# New function words are sometimes required for grammar.  Any new content word is rejected
# deterministically so the editor cannot restore details from the original or invent a
# completion for a broken fragment.
FUNCTION_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "could", "did", "do", "does", "doing",
    "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more",
    "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "please",
    "same", "she", "should", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "we", "were",
    "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with",
    "would", "you", "your", "yours", "yourself", "yourselves",
    "one", "ones",
}


def words(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.casefold())


def has_new_content_word(repair, source):
    source_words = set(words(source))
    for word in words(repair):
        if word in source_words or word in FUNCTION_WORDS:
            continue
        # Permit only trivial inflection/spelling cleanup, never a novel concept.
        if any(abs(len(word) - len(old)) <= 1 and sum(a != b for a, b in zip(word, old))
               + abs(len(word) - len(old)) <= 1 for old in source_words):
            continue
        return True
    return False


def parse_object(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).rsplit("```", 1)[0]
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j < i:
        return None
    try:
        data = json.loads(text[i:j + 1])
    except json.JSONDecodeError:
        return None
    return data


def valid_repair(repair, source):
    if not repair or len(repair) > 2000:
        return False
    # Deterministically reject the main artifacts seen in the GCG corpus.  Semantic
    # completeness is handled by the editor prompt; these catch obvious violations.
    if "..." in repair or "…" in repair:
        return False
    if re.search(r"([.!?,;:])\1", repair):
        return False
    for left, right in (("(", ")"), ("[", "]"), ("{", "}")):
        if repair.count(left) != repair.count(right):
            return False
    if re.search(r"(?:\(|\[|\{)\s*$", repair):
        return False
    if re.search(r",\s*please[.!?]$", repair, flags=re.IGNORECASE):
        return False
    if re.search(r"[.!?]\s+(?:[A-Z][A-Za-z]*)(?:\s*,\s*[A-Z][A-Za-z]*)+[.!?]$", repair):
        return False
    if has_new_content_word(repair, source):
        return False
    normalized = re.sub(r"\s+", " ", repair).casefold()
    return normalized != re.sub(r"\s+", " ", source.strip()).casefold()


def atomic_json(path, value):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(value, f, indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="sonnet_filtered_strict.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", help="Default: OUT.checkpoint.jsonl")
    ap.add_argument("--backup_dir", help="Periodically copy checkpoint/final output here")
    ap.add_argument("--key_file", default=os.path.expanduser("~/.anthropic_key"))
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--fallback_model", default="claude-sonnet-4-6",
                    help="Used after repeated primary-model refusals")
    ap.add_argument("--audit_model", default="claude-sonnet-4-6",
                    help="Use a distinct Sonnet model for final repair QA")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=800)
    ap.add_argument("--backup_every", type=int, default=100)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--source_ids", help="Comma-separated cfg:idx IDs (for QA/smoke tests)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    with open(args.input) as f:
        source = json.load(f)
    rows = source["rows"]
    if args.source_ids:
        wanted = {x.strip() for x in args.source_ids.split(",") if x.strip()}
        rows = [r for r in rows if f"{r['cfg']}:{int(r['idx'])}" in wanted]
        missing = wanted - {f"{r['cfg']}:{int(r['idx'])}" for r in rows}
        if missing:
            raise ValueError(f"unknown source IDs: {sorted(missing)}")
    if args.limit is not None:
        rows = rows[:args.limit]
    items = []
    for row in rows:
        items.append({
            "source_id": f"{row['cfg']}:{int(row['idx'])}",
            "cfg": row["cfg"], "idx": int(row["idx"]),
            "original": row["original"], "rewrite": row["rewrite"],
            "source_similarity": row.get("similarity"),
            "source_rewrite_response": row.get("rewrite_response", ""),
            "source_rewrite_judgment": row.get("rewrite_judgment", {}),
        })
    if len({x["source_id"] for x in items}) != len(items):
        raise ValueError("source IDs are not unique")
    print(f"[data] {len(items)} source rewrites; 3 repairs each or UNREPAIRABLE", flush=True)
    if args.dry_run:
        print(f"[dry-run] would issue {len(items)} requests to {args.model}")
        return

    checkpoint = args.checkpoint or f"{args.out}.checkpoint.jsonl"
    completed = {}
    for path in (args.out, checkpoint):
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                prev = json.load(f).get("rows", []) if path == args.out else [
                    json.loads(line) for line in f if line.strip()]
            for row in prev:
                if ((len(row.get("repairs", [])) == 3 or row.get("status") == "UNREPAIRABLE")
                        and not row.get("error")):
                    completed[row["source_id"]] = row
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            print(f"[resume] ignoring unreadable {path}: {e}", flush=True)
    pending = [x for x in items if x["source_id"] not in completed]
    print(f"[resume] {len(completed)} complete; {len(pending)} requests remaining", flush=True)

    import anthropic
    with open(args.key_file) as f:
        client = anthropic.Anthropic(api_key=f.read().strip())

    def one(item):
        trigger = item["source_rewrite_judgment"].get("trigger", "")
        last_error = "invalid response"
        feedback = ""
        had_parseable_proposal = False
        for attempt in range(5):
            try:
                generation_model = args.model if attempt < 2 else args.fallback_model
                message = client.messages.create(
                    model=generation_model, max_tokens=args.max_tokens, system=SYSTEM,
                    thinking={"type": "disabled"},
                    messages=[{"role": "user", "content": USER_T.format(
                        rewrite=item["rewrite"], trigger=trigger) + feedback}],
                )
                if message.stop_reason == "refusal":
                    last_error = "model refused copy-editing request"
                else:
                    txt = next((b.text for b in message.content
                                if getattr(b, "type", "") == "text"), "")
                    proposal = parse_object(txt)
                    if not isinstance(proposal, dict):
                        last_error = "invalid proposal JSON"
                        continue
                    status = str(proposal.get("status", "")).upper()
                    repairs = proposal.get("repairs", [])
                    if status in ("OK", "UNREPAIRABLE"):
                        had_parseable_proposal = True
                    if status == "OK":
                        repairs = [str(x).strip() for x in repairs] if isinstance(repairs, list) else []
                        normalized = {re.sub(r"\s+", " ", x).casefold() for x in repairs}
                        if (len(repairs) != 3 or len(normalized) != 3
                                or not all(valid_repair(x, item["rewrite"]) for x in repairs)):
                            last_error = "deterministic faithfulness/artifact check failed"
                            feedback = f"\nPrevious attempt failed: {last_error}. Try again."
                            continue
                    elif status == "UNREPAIRABLE":
                        proposal = {"status": "UNREPAIRABLE",
                                    "reason": str(proposal.get("reason", ""))[:240]}
                        repairs = []
                    else:
                        last_error = "proposal status was not OK or UNREPAIRABLE"
                        continue

                    audit_model = args.audit_model if generation_model != args.audit_model else args.model
                    audit = client.messages.create(
                        model=audit_model, max_tokens=250, system=AUDIT_SYSTEM,
                        thinking={"type": "disabled"},
                        messages=[{"role": "user", "content": AUDIT_USER_T.format(
                            rewrite=item["rewrite"], trigger=trigger,
                            proposal=json.dumps(proposal if status == "UNREPAIRABLE" else
                                                {"status": "OK", "repairs": repairs}))}],
                    )
                    audit_txt = next((b.text for b in audit.content
                                      if getattr(b, "type", "") == "text"), "")
                    verdict = parse_object(audit_txt) or {}
                    per_repair = [str(x).upper() for x in verdict.get("repair_verdicts", [])]
                    audit_pass = str(verdict.get("verdict", "")).upper() == "PASS"
                    if status == "OK":
                        audit_pass = audit_pass and per_repair == ["PASS", "PASS", "PASS"]
                    if audit_pass:
                        if status == "UNREPAIRABLE":
                            return {**item, "status": "UNREPAIRABLE", "repairs": [],
                                    "unrepairable_reason": proposal["reason"],
                                    "repair_model": generation_model, "audit_model": audit_model}
                        return {**item, "status": "OK", "repairs": repairs,
                                "repair_model": generation_model, "audit_model": audit_model}
                    last_error = f"audit rejected: {str(verdict.get('reason', ''))[:120]}"
                    feedback = f"\nPrevious attempt failed final audit: {last_error}. Try again."
            except Exception as e:  # noqa: BLE001
                last_error = str(e)[:160]
            if attempt < 4:
                time.sleep(min(30, 2 ** attempt) + random.random())
        if had_parseable_proposal:
            return {**item, "status": "UNREPAIRABLE", "repairs": [],
                    "unrepairable_reason": f"No faithful repair set passed final audit: {last_error}",
                    "repair_model": args.model, "audit_model": args.audit_model}
        if "refused copy-editing request" in last_error:
            return {**item, "status": "UNREPAIRABLE", "repairs": [],
                    "unrepairable_reason": "Both Sonnet editors refused the repair request; no faithful audited repair was produced.",
                    "repair_model": args.fallback_model, "audit_model": args.audit_model}
        return {**item, "repairs": [], "repair_model": args.model, "error": last_error}

    os.makedirs(os.path.dirname(os.path.abspath(checkpoint)), exist_ok=True)
    if args.backup_dir:
        os.makedirs(args.backup_dir, exist_ok=True)
    backup_lock = threading.Lock()

    def backup(path):
        if args.backup_dir and os.path.exists(path):
            with backup_lock:
                shutil.copy2(path, os.path.join(args.backup_dir, os.path.basename(path)))

    errors = 0
    with open(checkpoint, "a") as ckpt, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(one, item): item for item in pending}
        for done, future in enumerate(as_completed(futures), 1):
            row = future.result()
            ckpt.write(json.dumps(row) + "\n")
            ckpt.flush()
            if row.get("error"):
                errors += 1
            else:
                completed[row["source_id"]] = row
            if done % args.backup_every == 0:
                backup(checkpoint)
            if done % 50 == 0 or done == len(pending):
                print(f"[progress] {done}/{len(pending)} new; errors={errors}", flush=True)

    ordered = [completed[x["source_id"]] for x in items if x["source_id"] in completed]
    repaired = [x for x in ordered if x.get("status") == "OK"]
    unrepairable = [x for x in ordered if x.get("status") == "UNREPAIRABLE"]
    result = {
        "definition": "three distinct minimal grammatical repairs per strict rewrite; no added content",
        "repair_model": args.model,
        "n_sources": len(ordered),
        "n_repaired_sources": len(repaired),
        "n_unrepairable_sources": len(unrepairable),
        "repairs_per_source": 3,
        "n_repairs": 3 * len(repaired),
        "rows": ordered,
    }
    atomic_json(args.out, result)
    backup(checkpoint)
    backup(args.out)
    print(f"[done] wrote {len(repaired)} repaired + {len(unrepairable)} unrepairable sources / "
          f"{3 * len(repaired)} repairs to {args.out}", flush=True)
    if errors or len(ordered) != len(items):
        print(f"[warning] {errors} errors this pass; rerun the same command to retry", flush=True)


if __name__ == "__main__":
    main()
