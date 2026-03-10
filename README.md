PRReview
========

A lightweight, fast Command Line Interface (CLI) tool written in Go that uses Google's Gemini AI to help you review GitHub Pull Requests. Vibe-coded with Gemini.

Unlike fully autonomous agents that post comments directly, this tool is designed for a **"human-in-the-loop"** workflow. It fetches the PR context, analyzes the code or documentation changes, and prints structured, highly actionable feedback directly to your terminal. You can then review the AI's suggestions, copy the valuable parts, and post them in your GitHub review.

## ✨ Features

*   **URL Parsing:** Just paste a standard GitHub PR URL (e.g., `https://github.com/owner/repo/pull/123`).
*   **Rich Context Gathering:** Automatically fetches the PR's code diff, title, description, and commit messages to understand the author's true intent.
*   **Smart Context Switching:** Dynamically adjusts its review style based on file extensions. It checks code for logic, security, and performance issues, but shifts to checking technical accuracy and readability for Markdown/documentation files.
*   **Testing Evaluation:** Recommends specific Unit or End-to-End tests based on the logic changed in the PR.
*   **Custom Prompts (Nudging):** Use the `-p` flag to give the AI specific instructions or ask targeted questions about the PR before it reviews.

## 🛠 Prerequisites

*   [Go](https://go.dev/doc/install) (1.20 or later recommended)
*   A **Gemini API Key** (Get one from [Google AI Studio](https://aistudio.google.com/))
*   *(Optional but Recommended)* A **GitHub Personal Access Token** to avoid rate limits.

## ⚙️ Configuration

You need to set your API keys as environment variables. Add these to your `~/.bashrc`, `~/.zshrc`, or export them directly in your terminal:

```bash
# Required to generate the review
export GEMINI_API_KEY="your_gemini_api_key_here"

# Optional, but highly recommended. Without it, GitHub limits you to 60 requests/hour.
export GITHUB_TOKEN="your_github_personal_access_token_here"
```

## 🚀 Installation & Build

This project includes a `Makefile` for easy compilation.


1.  Build the binary:

    ```bash
    make build
    ```

2.  (Optional) Install it globally to your `$GOPATH/bin` so you can run it from anywhere:

    ```bash
    make install
    ```

## 📖 Usage

Run the tool by passing a GitHub PR URL.

**Standard Review:**

```bash
prreview https://github.com/facebook/react/pull/28741
```

**Targeted Review (Custom Prompting):**
If you have a specific concern, you can "nudge" the AI using the `-p` or `--prompt` flag:

```bash
prreview -p "Pay close attention to how the database inputs are sanitized. I'm worried about SQL injection." https://github.com/facebook/react/pull/28741
```

## 🏗 Architecture

1.  **Input:** The CLI accepts a GitHub PR URL and an optional prompt string.
2.  **Fetch:** It makes three calls to the GitHub API to gather the raw diff, the PR metadata (Title/Body), and the Commit History.
3.  **Prompt Construction:** It merges this data into a highly structured system prompt.
4.  **AI Generation:** The payload is sent to the Gemini 2.5 Pro model.
5.  **Output:** The AI returns a Markdown-formatted response categorized by "High-Level Summary," "Actionable Feedback," and "Testing & Validation," which is printed to standard output.
