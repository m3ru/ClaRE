# ClaRE repository organization guide

This doc explains **which branch to use**, **whether you need `sample-data`**, and **how to rename folders** so the repo clearly shows what the team has worked on instead of "Alec's Stuff" / "Andrew's stuff".

---

## 1. Which branch has all the info?

| Branch | What it has | Relation to others |
|--------|-------------|--------------------|
| **sample-data** | Alec's Stuff, Andrew Refusal Classification, SFT_Scoring, scripts, src, assets. **No** data/ sample files (they were removed). | Default branch. **This is the most complete.** |
| **refusal-and-regex** | Same as sample-data + 1 merge commit. Effectively the same content. | 1 commit ahead of sample-data. |
| **main** | Only: Alec's Stuff, data/, src. **Missing** Andrew Refusal Classification, SFT_Scoring, scripts, assets. | 60 commits behind sample-data. **Out of date.** |
| **refusal-classifier-andrew** | Alec's Stuff, Andrew Refusal Classification, notebooks, src. **Missing** SFT_Scoring, scripts, assets. | 67 commits behind sample-data. **Older.** |

**Conclusion:** **sample-data** (or refusal-and-regex) has all the current work. **You do need the sample-data branch** — it *is* your main development branch; the name is just misleading now that sample data was removed.

**Recommendation:** Treat **sample-data** as the single source of truth. Optionally make **main** match it (e.g. merge sample-data into main and use main as default) so new visitors see the full repo on main.

---

## 2. Branch consolidation (optional)

To simplify:

1. **Merge sample-data into main** so main has the full repo.
2. **Set main as the default branch** on GitHub (Settings → General → Default branch).
3. Keep **sample-data** as a legacy name or delete it after main is updated.
4. Use **refusal-and-regex** and **refusal-classifier-andrew** only for historical reference or delete them once merged.

---

## 3. Folder renames: from "Alec's Stuff" / "Andrew Refusal Classification" to what we actually built

Goal: **Rename by research area** so the repo is self-explanatory and usable.

### Current → proposed mapping

| Current name | Proposed name | What it is |
|--------------|----------------|------------|
| **Alec's Stuff** | *(split into the rows below)* | — |
| ↳ EPO_Dreams | **research/epo_dreams** | EPO refusal “dream” optimization |
| ↳ Getting_Generations | **research/prompt_generations** | LLaMA/Reddit prompt generation, refusal filter |
| ↳ Getting_Refusal_Vector | **research/refusal_vector** | Refusal vector extraction, PEZ, steering demo |
| ↳ OR_Fine_Tuning | **research/overrefusal_finetuning** | PPO overrefusal, prompt rewriter, sanitization |
| ↳ Sampling | **research/overrefusal_sampling** | Rejection sampling for (benign, over-refusal) pairs |
| **Andrew Refusal Classification** | **research/refusal_classification** | M3 refusal classifier, regex refusals, classification pipelines |
| **SFT_Scoring** | **SFT_Scoring** *(no change)* | Scoring pipeline for SFT candidates (already clear) |

All research lives under **research/** so the root stays clean and each subfolder name describes the work.

### Alternative: flat structure

If you prefer no `research/` and want everything at top level:

- **refusal_classification** (from Andrew Refusal Classification)
- **refusal_vector** (from Alec's Stuff/Getting_Refusal_Vector)
- **prompt_generations** (from Alec's Stuff/Getting_Generations)
- **epo_dreams** (from Alec's Stuff/EPO_Dreams)
- **overrefusal_finetuning** (from Alec's Stuff/OR_Fine_Tuning)
- **overrefusal_sampling** (from Alec's Stuff/Sampling)
- **SFT_Scoring** (unchanged)
- **scripts**, **src**, **assets** (unchanged)

---

## 4. How to do the renames (research/ layout)

Run from repo root on **sample-data** (or main after merging):

```bash
# Create research and move Alec's Stuff subdirs into it with new names
mkdir -p research
git mv "Alec's Stuff/EPO_Dreams" research/epo_dreams
git mv "Alec's Stuff/Getting_Generations" research/prompt_generations
git mv "Alec's Stuff/Getting_Refusal_Vector" research/refusal_vector
git mv "Alec's Stuff/OR_Fine_Tuning" research/overrefusal_finetuning
git mv "Alec's Stuff/Sampling" research/overrefusal_sampling

# Move loose files from Alec's Stuff into one of the above or into research/
# (e.g. run_refusal_filter.py, run_llama_prompts_*.py/slurm, CSVs)
# Suggested: put run_* and Reddit CSVs into research/prompt_generations
git mv "Alec's Stuff/run_refusal_filter.py" research/prompt_generations/
git mv "Alec's Stuff/run_refusal_filter.slurm" research/prompt_generations/
git mv "Alec's Stuff/run_llama_prompts_lines_sharded.py" research/prompt_generations/
git mv "Alec's Stuff/run_llama_prompts_filtered.slurm" research/prompt_generations/
git mv "Alec's Stuff/run_llama_prompts_reddit.slurm" research/prompt_generations/
git mv "Alec's Stuff/llama8b_outputs_filtered.csv" research/prompt_generations/ 2>/dev/null || true
git mv "Alec's Stuff/outputs_reddit.csv" research/prompt_generations/ 2>/dev/null || true
git mv "Alec's Stuff/Final-People-Reddit-Prompts-"*.csv research/prompt_generations/ 2>/dev/null || true

# Andrew's folder → research/refusal_classification
git mv "Andrew Refusal Classification" research/refusal_classification

# Remove empty Alec's Stuff (if any files remain, move them into research/prompt_generations first)
rmdir "Alec's Stuff" 2>/dev/null || true
git add -A
git status
```

**If `Alec's Stuff` still has other files** (e.g. large CSVs), either move them into `research/prompt_generations` or add them to `.gitignore` and leave them out of the repo.

Then:

1. **Update any paths** in READMEs and scripts that reference `Alec's Stuff` or `Andrew Refusal Classification` (e.g. in Sampling README: `../OR_Fine_Tuning/` → `../overrefusal_finetuning/`).
2. **Commit:**  
   `git commit -m "Reorganize repo: research/ layout and descriptive folder names"`

---

## 5. After reorganizing: root layout

You’ll get something like:

```
ClaRE/
├── research/
│   ├── epo_dreams/
│   ├── prompt_generations/
│   ├── refusal_vector/
│   ├── overrefusal_finetuning/
│   ├── overrefusal_sampling/
│   └── refusal_classification/
├── SFT_Scoring/
├── scripts/
├── src/
├── assets/
├── .gitignore
└── README.md
```

This makes it obvious what each part does and keeps the repo usable without relying on "Alec's stuff" or "Andrew's stuff."
