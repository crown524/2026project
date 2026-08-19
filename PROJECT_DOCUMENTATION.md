# DAIC-WOZ Depression Risk Screening System

## Project Overview

**Competition ID:** XH-202617  
**Task:** Multimodal depression risk assessment system based on DAIC-WOZ dataset  
**Core Architecture:** LLM extracts multimodal evidence → GBDT generates risk score → SHAP provides explainability

**Design Philosophy:** Layered architecture - not a pure black box, but a hybrid system combining "LLM semantic understanding + small model calibrated scoring" for both accuracy and interpretability.

---

## Dataset Status

### Raw Data
- **Source:** DAIC-WOZ (Depression, Anxiety, and Clinical Interview - Wizard of Oz)
- **Format:** 189 `*_P.zip` packages, each ~604 MB, total ~300 GB
- **Content:** Each package contains a complete interview session between a participant and Ellie virtual interviewer
  - `*_TRANSCRIPT.csv` - sentence-by-sentence transcripts (participant + interviewer dialogue)
  - `*_CLNF_features3D.txt` / `*_CLNF_AUs.txt` - visual features (OpenFace)
  - `*_COVAREP.csv` - acoustic features (COVAREP toolkit)
  - `*_P/*.jpg` - video frames (for HOG, we skip these)
  - `*_AUDIO.wav` - raw audio (optional preserve, not included in export package)

### Extracted Layer: data/extracted/ (2.5 GB)
- `transcripts/*.csv` - 189 transcript files, extracted locally
- `covarep/*.csv` - acoustic features (83 dims)
- `clnf/*.txt` - visual features (AU + 3D landmarks, 47 dims)
- **Excluded items (saves ~200 GB):**
  - HOG features (~73% of volume, we don't use)
  - `features3D.txt` (CLNF already contains it)
  - `AUDIO.wav` (can be saved separately)

### Feature Layer: outputs/features/ (~500 KB)
Currently loaded with **189 people complete dataset**:

| File | Rows | Cols | Description |
|---|---|---|---|
| `a1_transcript_timing.csv` | 189 | 35 | tt lane: transcript timing features (pauses, speech rate, turns) |
| `a3_acoustic.csv` | 189 | 84 | ac lane: acoustic features (pitch, formants, voice quality) |
| `a5_visual.csv` | 189 | 48 | vi lane: visual features (AU intensity, gaze, head pose) |
| `a2_text_observations.csv` | 189 | 53 | obs lane: LLM-extracted PHQ-8 symptom observations |
| `d1_direct_scores.csv` | 189 | 4 | d1 lane: LLM direct PHQ-8 total score estimates |

**Backup:** Old 41-person version in `outputs/features/_backup_41p/`

### Labels: data/labels/ (5 KB)
- `train_split_Depression_AVEC2017.csv` - 107 people, PHQ-8 total score + binary label
- `dev_split_Depression_AVEC2017.csv` - 35 people
- `test_split_Depression_AVEC2017.csv` - 47 people (locked, use only once for final evaluation)
- `full_test_split.csv` - test set complete info (with participant IDs)

**Trainable samples:** 142 people (107 train + 35 dev), remaining 47 people locked in test set.

---

## File Structure and Functions

### Core Scripts

#### 1. config.py
**Function:** Global configuration and path management

- **Data path resolution:** `_resolve_zip_dir()` finds raw zip location by priority:
  1. Environment variable `DAIC_ZIP_DIR`
  2. `zip_dir` in `local_paths.json`
  3. Default `~/Desktop/daic_woz_packages`
- **Key constants:**
  - `EXCLUDED_KEYWORDS = ("hog", "features3D", "features.txt", "AUDIO.wav")` - skip these during extraction, saves 200 GB
  - `COL_PARTICIPANT_ID = "Participant_ID"` - ID column name in label table
  - `COL_PHQ8_SCORE = "PHQ8_Score"` - PHQ-8 total score (0-24, continuous)
  - `COL_PHQ8_BINARY = "PHQ8_Binary"` - binary classification label (0=healthy, 1=depressed)
- **Random seed:** `RANDOM_SEED = 42` ensures reproducibility

#### 2. extract_from_zips.py
**Function:** Extract transcripts, acoustic, and visual features on-demand from 189 raw zip packages

**Workflow:**
1. Iterate through zip directory, find all `*_P.zip`
2. For each file, check if `data/extracted/` already has same-name same-size file
3. If not exists, unzip and extract:
   - `*_TRANSCRIPT.csv` → `transcripts/`
   - `*_COVAREP.csv` → `covarep/`
   - `*_CLNF_AUs.txt` → `clnf/`
4. Skip files in `EXCLUDED_KEYWORDS`
5. Report cumulative extraction volume and savings

**Idempotency:** Already-extracted files won't be re-extracted, so re-running script is safe.

**Usage:**
```bash
python extract_from_zips.py
```

#### 3. build_features.py
**Function:** Compute three deterministic feature lanes (tt / ac / vi) from extracted layer

**Input:** `data/extracted/{transcripts,covarep,clnf}/`  
**Output:** `outputs/features/{a1_transcript_timing, a3_acoustic, a5_visual}.csv`

**Feature sources:**
- **tt lane (34 dims):** `features/transcript_timing.py`
  - Pause statistics (silence_gap_mean, silence_gap_std, silence_total_s)
  - Speech rate (words_per_second, chars_per_second)
  - Turn statistics (patient_turns, interviewer_turns, turn_duration_*)
  - Response latency (response_latency_mean, response_latency_p90)
- **ac lane (83 dims):** `features/acoustic.py`
  - Fundamental frequency (f0_mean, f0_std, f0_iqr)
  - Formants (formant_f1/f2/f3 mean/std/quantiles)
  - Voice quality (jitter, shimmer, naq, h1h2, psp)
  - MFCC coefficients (mcep0-mcep11 statistics)
- **vi lane (47 dims):** `features/visual.py`
  - AU (Action Unit) intensity (au01-au45 mean/std/activation rate)
  - Gaze (gaze_wander_mean, gaze_fixed_rate)
  - Head pose (pose_pitch/yaw/roll variability)

**Usage:**
```bash
python build_features.py
```

#### 4. run_text_skill.py
**Function:** Call LLM to generate two text feature lanes (obs / d1)

**Input:** `data/extracted/transcripts/*.csv`  
**Output:**
- `outputs/features/a2_text_observations.csv` (obs lane, 26 observation columns)
- `outputs/features/d1_direct_scores.csv` (d1 lane, PHQ-8 total score)
- `outputs/skill_cache/{text_observation,d1_direct_scoring}/*.json` (cache to avoid re-calling API)

**Key parameters:**
- `--skill {text_observation, d1_direct_scoring}` - select Skill
- `--model MODEL` - override model in config file (e.g. `gpt-5.6`, `claude-sonnet-5`)
- `--only PARTICIPANT_ID ...` - only run specified samples (for testing)
- `--limit N` - only run first N samples
- `--dry-run` - don't call API, only print prompt

**Skill prompt locations:**
- `skills/text_observation/skill.md` - PHQ-8 symptom observation extraction
- `skills/d1_direct_scoring/skill.md` - LLM direct PHQ-8 total score

**API config:** `data/skill_models.json` (plaintext, don't commit to repo)

**Usage:**
```bash
# Generate obs lane (189 people, ~5 minutes)
python run_text_skill.py --skill text_observation

# Generate d1 lane
python run_text_skill.py --skill d1_direct_scoring

# Test on one sample only
python run_text_skill.py --skill text_observation --only 300 --dry-run
```

#### 5. run_experiment.py
**Function:** Unified ablation experiment script, choose any feature lane combination, run one experiment group with one command

**Supported ablation groups:**
- A1: `--lanes tt` - transcript timing only
- A2: `--lanes obs` - text observations only
- A3: `--lanes ac` - acoustic only
- A5: `--lanes vi` - visual only
- B4: `--lanes tt,obs,ac,vi` - full config (deterministic three lanes + obs)
- D1: `--lanes d1` - LLM direct scoring (no model training)
- D2: equivalent to B4 (observation features + small model = layered architecture)

**Model options:**
- `--model gbdt` - GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05)
- `--model ridge` - Ridge(alpha=1.0) linear baseline
- `--model both` - run both

**Evaluation protocols (automatically runs both):**
1. **Official split:** train on train → evaluate on dev (comparable with literature)
2. **LOSO (Leave-One-Subject-Out):** merge train+dev for leave-one-out cross-validation (more stable with small sample)

**Output:** `outputs/results/exp_{lanes}_{timestamp}.json`

**Per-subject predictions (per_subject):** Every result includes `per_subject` dictionary, format `{participant_id: prediction}`, used for paired comparison.

**Usage:**
```bash
# Run obs single lane
python run_experiment.py --lanes obs

# Run full config
python run_experiment.py --lanes tt,obs,ac,vi

# LLM direct scoring
python run_experiment.py --lanes d1
```

#### 6. compare_configs.py
**Function:** Paired comparison of two experiment configs, bootstrap 95% CI on difference

**Why needed:** Two configs' CIs overlapping ≠ difference not significant. To judge "is B truly stronger than A" must do interval estimation on **the difference itself**, and must be paired on the same batch of subjects - otherwise each config's CI will double-count shared subject sampling variation, masking true difference.

**Dependency:** `per_subject` field written by `run_experiment.py` (per-person predictions)

**Usage:**
```bash
# Compare D1 vs B4 (core thesis of layered architecture)
python compare_configs.py exp_d1_20260811T120347Z.json exp_tt_obs_ac_vi_20260811T120238Z.json --label "train+dev"

# Only compare LOSO protocol
python compare_configs.py A.json B.json --label "LOSO"
```

**Output example:**
```
Paired comparison (same batch of subjects, bootstrap 95% CI on difference)
  A: d1  [D1 LLM direct scoring · train+dev all]
  B: tt+obs+ac+vi  [tt+obs+ac+vi(gbdt) · LOSO train+dev]
  Common subjects: 142 people

  Spearman ρ (higher better)
    A = 0.742   B = 0.630   difference(B-A) = -0.112
    Difference 95% CI [-0.216, -0.019]
    ��� Interval does not cross zero: A significantly better

  RMSE (lower better)
    A = 5.191   B = 4.582   difference(B-A) = -0.609
    Difference 95% CI [-1.093, -0.075]
    → Interval does not cross zero: B significantly better
```

#### 7. eval_utils.py
**Function:** Evaluation metric computation and result formatting

**Core functions:**
- `load_labels(with_test=False)` - read labels, default not reading test set (discipline requirement)
- `evaluate(y_true, y_pred, label)` - compute full metrics with bootstrap confidence intervals:
  - RMSE, MAE
  - Spearman ρ (+ p-value + 95% CI)
  - Pearson r (+ p-value)
  - PR-AUC (threshold=10, i.e. PHQ≥10 for depression) (+ 95% CI)
- `print_results(result_dict)` - formatted result printing

**Bootstrap parameter:** `N_BOOTSTRAP = 1000` (configurable in `config.py`)

---

### Data Export and Portability

#### 8. export_tool.py + export_gui.bat
**Function:** Dataset export tool (GUI), turns 300 GB raw data into ~500 KB portable artifact

**GUI workflow:**
1. Select raw zip source folder
2. Select export target folder (default `~/Desktop/daic_export_package`)
3. Check options:
   - **Extract zip** - run `extract_from_zips.py` (can disable if already extracted)
   - **Recalculate features** - run `build_features.py` (check if feature code changed)
   - **Generate text observations and D1 scoring** - run `run_text_skill.py --skill text_observation` and `--skill d1_direct_scoring` (requires API call, necessary for obs/d1 lanes)
   - **Save raw audio separately** - save `AUDIO.wav` copy to extracted layer (A4 backup, not in export package, takes space)
   - **Include transcript originals and Skill cache** - pack verbatim transcripts and LLM cache (protocol-restricted, USB only, cannot distribute)
4. Click "Start Export"

**Artifact tiers:**
- **Distributable (~500 KB):** Five feature tables (a1/a3/a5/a2/d1) + labels + official split. Pure statistics, cannot reverse to original.
- **Restricted (~7 MB):** Transcript originals + Skill cache. Contains verbatim quotes, DAIC-WOZ protocol restricted.

**Export package contents:**
```
daic_export_package/
���── outputs/
│   └── features/
│       ├── a1_transcript_timing.csv
│       ├─��� a2_text_observations.csv
│       ├���─ a3_acoustic.csv
│       ├── a5_visual.csv
│       ���── d1_direct_scores.csv
├── data/
���   └── labels/
│       ├── train_split_Depression_AVEC2017.csv
���       ├���─ dev_split_Depression_AVEC2017.csv
│       ���── test_split_Depression_AVEC2017.csv
│       └── full_test_split.csv
├─��� data/extracted/transcripts/  (if checked "include transcript originals")
├── outputs/skill_cache/         (if checked "include Skill cache")
└── export_notes.txt
```

**export_notes.txt content:**
- Source dataset: number of zips
- Extracted transcripts: number of files
- Included files: count, KB
- **Feature lane people counts** (key, shows which lane didn't keep up):
  ```
  tt   189 people    transcript timing
  ac   189 people    acoustic
  vi   189 people    visual
  obs  189 people    text observations
  d1   189 people    LLM direct scoring
  ```
  If counts don't match (e.g. d1 only has 188 people), there's a warning: "Experiment script inner-joins on participant_id, multi-lane combinations will be cut to minimum without error."

**Usage (on data machine):**
```bash
# Double-click export_gui.bat, or
python export_tool.py
```

Then copy the entire `daic_export_package` folder away (USB / network transfer both OK, just don't publicly upload).

#### 9. make_portable.py
**Function:** Package entire project into USB-portable package (code only, no data)

**Package contents:**
- Code: `*.py`, `features/*.py`, `skills/*/*.md`
- Labels: `data/labels/*.csv` (5 KB, must include)
- Config template: `data/skill_models.json` (API key stripped by default)
- Docs: `*.md`
- Entry points: `setup_and_run.bat`, `setup_and_run.py`, `export_gui.bat`

**Not included:**
- `.venv` (541 MB, target machine rebuilds)
- `data/extracted` (2.5 GB, target machine has raw packages)
- `outputs/` (unless `--with-features`)

**Usage:**
```bash
# Output to desktop
python make_portable.py

# Output to USB, with feature tables
python make_portable.py --out D:\ --with-features

# Output folder (not zip)
python make_portable.py --folder

# Include API key (default stripped, avoid leakage)
python make_portable.py --with-key
```

**Output:** `~/Desktop/daic_project_portable.zip` (or folder)

#### 10. setup_and_run.py + setup_and_run.bat
**Function:** One-click deployment and startup on new computer

**Workflow:**
1. Detect Python environment (3.10+)
2. Create `.venv` virtual environment
3. Install dependencies (`pip install -r requirements.txt`)
4. Detect if dependencies complete (`import pandas, sklearn, streamlit, httpx`)
5. Ask for raw zip location path, write to `local_paths.json`
6. Start Streamlit experiment panel (`streamlit run app_ui.py`)

**Usage (on target machine):**
```bash
# Windows: double-click
setup_and_run.bat

# Or manual
python setup_and_run.py
```

---

### Experiment Panel and Configuration

#### 11. app_ui.py
**Function:** Streamlit experiment panel, Web UI manages entire pipeline

**Pages:**
1. **Experiment Panel** - select feature lanes + model, click to run experiment, real-time display results
2. **Model Config** - edit `data/skill_models.json`, change API key / base_url / model
3. **Results Comparison** - historical experiment result comparison, plot Spearman ρ and RMSE comparison charts
4. **Skill Preview** - view two Skills' prompt contents

**Usage:**
```bash
streamlit run app_ui.py
```

Then browser opens `http://localhost:8501`

#### 12. llm_providers.py
**Function:** LLM call unified interface, supports Anthropic / OpenAI-compatible

**Config file:** `data/skill_models.json`
```json
{
  "text_observation": {
    "provider": "openai_compatible",
    "model": "gpt-5.6",
    "api_key": "sk-...",
    "base_url": "https://skyapi2026.com",
    "reasoning": "off"
  },
  "d1_direct_scoring": {
    "provider": "openai_compatible",
    "model": "claude-sonnet-5",
    "api_key": "sk-...",
    "base_url": "https://skyapi2026.com",
    "reasoning": "off"
  }
}
```

**Core functions:**
- `resolve_model(skill, cli_model)` - resolve model config, priority: CLI param > config file > env var
- `call_llm(cfg, messages, structured_output_schema)` - unified call interface, auto-handles structured output

**Supported providers:**
- `anthropic` - official Anthropic API
- `openai_compatible` - OpenAI-compatible endpoints (Claude / GPT both work)

---

### Auxiliary Scripts

#### 13. make_download_list.py
**Function:** Generate DAIC-WOZ dataset download manifest

**Usage:**
```bash
python make_download_list.py
```

**Output:** `download_urls.txt`, one download link per line, usable with `wget -i` for batch download.

#### 14. run_a1_baseline.py
**Function:** Run A1 baseline (transcript timing only), quick pipeline verification

**Equivalent to:**
```bash
python run_experiment.py --lanes tt
```

---

## Current Experimental Results (142 people, LOSO)

| Config | Spearman ρ | 95% CI | RMSE |
|---|---|---|---|
| **B0 mean baseline** | -1.0 (artifact) | ��� | 5.774 |
| **A1 tt transcript timing** | 0.310 | [0.152, 0.450] | 5.903 |
| **A3 ac acoustic** | 0.103 | [-0.057, 0.268] | 6.071 |
| **A5 vi visual** | -0.071 | [-0.231, 0.100] | 6.617 |
| **A2 obs text observations** | 0.597 | [0.477, 0.692] | 4.546 |
| **B4 four-lane full config** | 0.630 | [0.522, 0.718] | 4.582 |
| **D1 LLM direct scoring** | 0.742 | [0.659, 0.806] | 5.191 |

**Key findings:**

1. **D1 vs D2 split in half:**
   - Ranking: D1 significantly better (ρ 0.742 vs 0.630, paired comparison difference CI doesn't cross zero)
   - Absolute error: D2 significantly better (RMSE 4.582 vs 5.191, difference CI doesn't cross zero)
   
   **Meaning:** LLM knows who is more severe, but doesn't know what score to give (calibration bias). GBDT calibrated the scale, at the cost of losing some ranking precision.

2. **Signal almost entirely in obs lane:** B4's feature importance Top 2 are `obs__obs_total_n` (0.191) and `obs__obs_depressed_n` (0.130), totaling 0.32. COVAREP 83 dims + CLNF 47 dims = 130 dims only contributed 5 useful features.

3. **Acoustic and visual currently noise:** Both ac and vi CIs cross zero, adding them makes RMSE worse. Paired comparison obs single lane vs B4, both metrics' difference CIs all cross zero - adding tt/ac/vi shows no provable benefit.

---

## TODO Features

### Short-term (doable in 2 weeks)
1. **Improve obs Skill quality:**
   - Add Chain-of-Thought (segment analysis ��� aggregate)
   - Separate symptom frequency vs severity
   - Require negative evidence (`evidence_for` vs `evidence_against`)
   - PHQ-8 calibration anchors (few-shot examples)
   
2. **Architecture improvements:**
   - Add D1 raw score to D2 (learn LLM's systematic bias)
   - Stacking: M_obs + M_certain + M_d1 ��� Meta-GBDT
   - Isotonic regression post-processing (order-preserving calibration)

3. **SHAP explainability:**
   - Pick 3 representative samples (mild/moderate/severe)
   - Show "why this score"
   - Compare D1 (black box) vs D2 (traceable)

### Mid-term (needs more data or hardware)
4. **A4 wav2vec lane:** Use pre-trained speech model to replace COVAREP
5. **F1/F2 fusion layer:** Weighted ensemble of multiple GBDTs
6. **E1 cross-corpus transfer:** Train on DAIC-WOZ → test on other datasets (e.g. E-DAIC)

### Long-term (paper refinement)
7. **Multi-annotator collaboration:** Multiple annotators score same sample, confidence weighting
8. **Active learning:** Find samples where D1 and D2 predictions differ most, manual review
9. **Test set final evaluation:** 47 locked people, use only once

---

## Safety and Protocol Constraints

### DAIC-WOZ Usage Protocol
- **Raw data** (transcripts, audio, video, COVAREP/CLNF frames) strictly no external distribution, only usable on protocol-signed machines
- **Feature tables** (irreversible statistics) freely transferable
- **Prohibited:** Upload to public cloud servers, share with unsigned personnel, use for commercial purposes

### Test Set Discipline
- Test set 47 people **can only be used once**, after all hyperparameters and architecture choices finalized on train+dev 142 people
- Prohibited: tune hyperparameters on test, select features on test, iterate prompts on test
- Once you saw test results and changed code, test is contaminated

### Prompt Iteration Discipline
- Can only iterate prompts on train set outputs
- Prohibited: "This sample LLM scored wrong, I'll change prompt to make it right" - this is test leakage variant
- Allowed: "LLM systematically overestimates severe patients on train set, I'll add calibration anchors"

---

## Dependencies and Environment

### Python Version
- **Required:** Python 3.10+
- **Tested:** Python 3.11.9

### Core Dependencies (requirements.txt)
```
pandas>=2.1.0
numpy>=1.24.0
scikit-learn>=1.3.0
scipy>=1.11.0
anthropic>=0.18.0
httpx>=0.24.0
streamlit>=1.28.0
plotly>=5.17.0
```

### Environment Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## FAQ

### Q1: Why is D2's ρ lower than D1?
**A:** Because D2 only saw 26 counts extracted by obs, while D1 saw complete transcripts. Information inequality. Solution: expand obs features (temporal segmentation, co-occurrence matrix, LLM confidence) + add D1 raw score to D2.

### Q2: Why are ac and vi lanes so poor?
**A:** COVAREP and CLNF are global statistics (average pitch over entire audio), losing temporal patterns ("pitch goes high then low" dynamic changes). Depression marker isn't "average pitch low", but "prosodic flattening".

### Q3: obs has 53 columns, why say 26 observations?
**A:** 26 PHQ-8 related observations + metadata columns (`abstained`, `data_sufficiency`, `quote_verification_failed`) + multiple statistics per observation (occurrence count, max frequency, explicit/negated).

### Q4: Why not use LLM end-to-end?
**A:** D1's scale is biased (RMSE 5.19 vs D2's 4.58), and it's a black box. Layered architecture can calibrate scale + provide SHAP explainability, meeting medical regulation requirements.

### Q5: When to use test set?
**A:** All code frozen, train final model on 142 people, evaluate **once** on 47 people, write to paper. If test results differ too much from LOSO, don't change model, honestly report in Discussion.

---

## Quick Start

### First run (have raw zip)
```bash
# 1. Extract raw data
python extract_from_zips.py

# 2. Compute deterministic three lanes
python build_features.py

# 3. Generate text observations (need API key)
python run_text_skill.py --skill text_observation
python run_text_skill.py --skill d1_direct_scoring

# 4. Run experiments
python run_experiment.py --lanes obs
python run_experiment.py --lanes tt,obs,ac,vi
python run_experiment.py --lanes d1

# 5. Paired comparison
python compare_configs.py exp_d1_*.json exp_tt_obs_ac_vi_*.json --label "train+dev"
```

### First run (only have export package)
```bash
# 1. Merge export package's outputs/ and data/ into project directory

# 2. Run experiments directly
python run_experiment.py --lanes obs

# 3. Or open Streamlit panel
streamlit run app_ui.py
```

### Data machine export workflow
```bash
# 1. Double-click export_gui.bat

# 2. Check:
#    - Extract zip
#    - Recalculate features
#    - Generate text observations and D1 scoring (prerequisite: have API key)

# 3. Click start export

# 4. Copy export package away
```

---

## Project Status

**Current stage:** Five feature lanes all in place (189 people), 142-person ablation experiments complete, discovered meaningful D1 vs D2 split.

**Next step:** Improve obs Skill quality + architecture improvements, goal is to raise D2's ρ from 0.630 to 0.70+, while maintaining RMSE advantage.

**Final goal:** Prove "LLM extracts features + small model predicts" layered architecture, while preserving ranking ability, can significantly improve absolute error and explainability, suitable for clinical deployment.

---

**Doc version:** 2026-08-13  
**Data version:** 189 people complete, 142 people labeled  
**Experiment version:** D1 vs D2 first non-crossing-zero
