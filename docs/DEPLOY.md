# Deploying: GitHub repo + Hugging Face Space

Everything here is done by you, in your own accounts. Nothing in this repo creates accounts, pushes, or stores secrets for you. Never paste a key into a file in the repo or into a chat.

## 0. Before you start

- A Hugging Face account (free) and a GitHub account (free).
- A Gemini API key from https://aistudio.google.com/apikey (free tier).
- `git` configured, and Docker optional (only to test the image locally first, see the README).

## 1. Create the Hugging Face Space

1. Go to https://huggingface.co/new-space.
2. Owner: your user. Space name: `eu-ai-gdpr-assistant`. License: your choice (MIT is fine for the code).
3. **Space SDK: Docker** (choose "Blank" template). Hardware: **CPU basic (free)**. Visibility: Public.
4. Click **Create Space**. You now have an empty Space at `https://huggingface.co/spaces/<your-user>/eu-ai-gdpr-assistant`.
5. In the Space: **Settings -> Variables and secrets -> New secret**: name `GOOGLE_API_KEY`, value = your Gemini key. (Secrets are not visible in logs or to visitors.)

### Why README.md starts with a block between `---` lines

Hugging Face reads the Space configuration from YAML front-matter at the top of `README.md` in the Space repo:

```yaml
---
title: EU AI Act + GDPR Assistant
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---
```

`sdk: docker` tells HF to build our `Dockerfile`; `app_port: 7860` is the port Streamlit listens on. On GitHub the same block is shown as a small table at the top of the README: harmless, just slightly ugly. (The alternative is a separate branch with a different README; that is more work and easy to forget, so we keep one README.)

## 2. Create the GitHub repository

1. https://github.com/new -> name `eu-ai-gdpr-assistant`, Public, **do not** add a README/.gitignore/licence (the repo already has a README, a `.gitignore` and an MIT `LICENSE` for the code).
2. In this project folder:
   ```bash
   git remote add origin https://github.com/<your-user>/eu-ai-gdpr-assistant.git
   git branch -M main
   git push -u origin main
   ```
   Check first that `git status` is clean and `.env` is not tracked (`git ls-files | grep -x .env` prints nothing).

## 3. Add the GitHub Actions secrets and variable

In the GitHub repo: **Settings -> Secrets and variables -> Actions**.

| Kind | Name | Value |
|---|---|---|
| Secret | `HF_TOKEN` | A Hugging Face access token with **Write** permission (https://huggingface.co/settings/tokens -> New token -> type "Write"). Better: a fine-grained token limited to your Space. |
| Secret | `GOOGLE_API_KEY` | Your Gemini key (only used by the 5-question eval smoke test on `main`). |
| Variable (Variables tab) | `HF_SPACE_ID` | `<your-user>/eu-ai-gdpr-assistant` |

Without these the pipeline still runs lint + tests; the eval-smoke and deploy jobs print "skipping" and finish green.

## 4. First deploy

Push to `main` (or re-run the latest workflow under **Actions**). The `deploy` job force-pushes a clean single-commit snapshot of the repo to the Space's `main` branch; `docs/superpowers/` (planning notes) is excluded from that snapshot, and the token is passed as an HTTP header, never in the URL. Then:

1. Open the Space -> **Logs**. The first build takes about 10 to 15 minutes (installs dependencies, builds the index, downloads the embedding model into the image). Later builds reuse cached layers.
2. When the status says **Running**, open `https://<your-user>-eu-ai-gdpr-assistant.hf.space` and ask a question.
3. If it says the service is unavailable when you ask, the `GOOGLE_API_KEY` Space secret is missing or wrong (the app starts without it on purpose, and `/ask` answers 503).

You can also deploy without CI by hand: `git remote add space https://huggingface.co/spaces/<your-user>/eu-ai-gdpr-assistant` then `git push space main` (use your HF username and a Write token as the password).

## 5. After deploying

- Put the public URL and a screenshot at the top of `README.md` (replace "coming soon").
- Free CPU Spaces go to sleep after inactivity and wake on the next visit (the first load takes a minute).
- Rate limits (10 questions per IP per hour, 200 per day overall) are set by environment variables; change them in the Space under **Variables** (not secrets): `RATE_LIMIT_PER_HOUR`, `DAILY_CAP`.
- Then run the full evaluation locally with your key (`uv run python eval/run_eval.py`) and add the answer-quality numbers to the README.
