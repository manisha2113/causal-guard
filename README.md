# CausalGuard

CausalGuard is a research framework and scanning platform designed to analyze, detect, and safeguard Large Language Models (LLMs) against vulnerabilities and safety risks through causal analysis and mechanistic interpretability techniques.

## Project Structure Overview

- **`ai_engine/`**: Core research algorithms, model scanning pipelines, causal tracing, and evaluation logic. Includes a dedicated `tests/` suite for unit testing.
- **`backend/`**: Application server layer structured into modular components:
  - `controllers/`: Request handling and API routing.
  - `services/`: Business logic, scan orchestration, and background worker coordination.
  - `models/`: Data schemas, domain representations, and entity models.
  - `database/`: Database connection management, clients, and repository utilities.
- **`frontend/`**: Web user interface for visualizing scan results, causal graphs, and platform metrics.
- **`data/`**: Datasets, evaluation prompts, and raw benchmarking inputs (large CSVs are ignored via `.gitignore`).
- **`experiments/`**: Research experiment scripts, configurations, and checkpoint trackers.
- **`reports/`**: Generated analytical reports, visualizations, figures, and exported evaluation summaries.
- **`requirements.txt`**: Core Python dependencies for ML research and backend testing.
- **`.env.example`**: Template for environment configuration variables (e.g., MongoDB connection URI).
- **`.gitignore`**: Git ignore rules for Python artifacts, virtual environments, environment files, and dataset CSVs.

## Datasets

CausalGuard downloads existing public datasets through the Hugging Face `datasets` library for alignment and safety evaluations:

- **`allenai/sciq`**: Science questions used for the `lie` task (normal questions paired with instruction-induced dishonest responses).
- **`walledai/AdvBench`**: Prompts used as optional normal class in the `jailbreak` task (when `--jailbreak-normal advbench`).
- **`TrustAIRLab/in-the-wild-jailbreak-prompts`**: Real-world collected prompts (`jailbreak_2023_12_25` for misbehavior class, `regular_2023_12_25` for default normal class).
- **`lmsys/toxic-chat`** (`toxicchat0124`): Real-world user interactions with toxicity labels for the `toxicity` task.
- **`truthfulqa/truthful_qa`**: Benchmark measuring model truthfulness for the `truthfulqa` evaluation task.

> [!NOTE]
> Review each dataset's respective license and usage terms on the Hugging Face Hub prior to downloading or using them in research or production.

### Generating Benchmark Datasets
Build balanced dataset CSVs across tasks with length distribution matching:
```bash
python -m ai_engine.build_datasets --task all --n-per-class 300 --max-chars 600 --seed 42 --jailbreak-normal regular --length-match --out-dir data
```

## Usage

### Single Prompt Scan
Scan a single prompt to compute token-level (Definition 2) and layer-level (Definition 3) causal attributions:
```bash
python -m ai_engine.cli scan --prompt "Who developed Windows 95?" --out result.json
```
Use `--no-chat-template` to evaluate raw prompt text without template formatting.

### Batch Scanning
Run batch inference across a CSV dataset with automated checkpointing and resume support:
```bash
python -m ai_engine.cli batch --input data/sample_prompts.csv --output data/features.csv
```
Use `--limit N` to test on the first N unprocessed rows.

#### Resume Functionality & Fault Tolerance
- **Checkpointing**: Every row is written and flushed immediately upon completion.
- **Automatic Resume**: If `features.csv` already exists, previously completed row IDs are read and skipped automatically, avoiding recomputation.
- **Error Handling**: If an individual prompt fails, its ID and error are appended to `<output>.errors.csv`, and batch scanning seamlessly continues.
- **Token Heatmap Data**: Full per-token attribution vectors are logged incrementally to `<output>.tokens.jsonl`.

## Running on Kaggle

To run batch feature extraction on Kaggle GPUs:

1. **Environment Settings**: In your Kaggle notebook, turn **Internet** on and select a **GPU accelerator** (e.g. T4 x2 or P100).
2. **Clone & Install**:
   ```bash
   git clone https://github.com/<your-username>/LLM-Scan-Platform.git
   cd LLM-Scan-Platform
   pip install -r requirements-kaggle.txt
   ```
3. **Authentication**: Add your Hugging Face access token as a Kaggle secret (`HF_TOKEN`) and log in:
   ```python
   from kaggle_secrets import UserSecretsClient
   from huggingface_hub import login
   token = UserSecretsClient().get_secret("HF_TOKEN")
   login(token=token)
   ```
4. **Attach Dataset**: Attach the Kaggle dataset containing the four task CSVs (`lie.csv`, `jailbreak.csv`, `toxicity.csv`, `truthfulqa.csv`).
5. **Run Extraction**:
   - **TinyLlama** (unquantized):
     ```bash
     python experiments/kaggle_extract.py \
       --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
       --quantize none \
       --data-dir /kaggle/input/causalguard-datasets \
       --out-dir /kaggle/working/features
     ```
   - **Mistral 7B** (4-bit quantized with NF4):
     ```bash
     python experiments/kaggle_extract.py \
       --model mistralai/Mistral-7B-Instruct-v0.2 \
       --quantize 4bit \
       --data-dir /kaggle/input/causalguard-datasets \
       --out-dir /kaggle/working/features
     ```
