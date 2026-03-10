package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"strings"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
)

// --- Structs for GitHub JSON Responses ---

type PRDetails struct {
	Title string `json:"title"`
	Body  string `json:"body"`
}

type CommitItem struct {
	Commit struct {
		Message string `json:"message"`
	} `json:"commit"`
}

// --- Helper Functions ---

func parsePRURL(prURL string) (owner, repo, prNumber string, err error) {
	u, err := url.Parse(prURL)
	if err != nil {
		return "", "", "", fmt.Errorf("invalid URL: %v", err)
	}

	path := strings.Trim(u.Path, "/")
	parts := strings.Split(path, "/")

	if len(parts) < 4 || parts[2] != "pull" {
		return "", "", "", fmt.Errorf("invalid GitHub PR URL format. Expected: https://github.com/owner/repo/pull/123")
	}

	return parts[0], parts[1], parts[3], nil
}

func doGitHubRequest(apiURL, acceptHeader, token string) ([]byte, error) {
	req, err := http.NewRequest("GET", apiURL, nil)
	if err != nil {
		return nil, err
	}

	req.Header.Set("Accept", acceptHeader)
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode == http.StatusForbidden && strings.Contains(strings.ToLower(string(bodyBytes)), "rate limit") {
		return nil, fmt.Errorf("GitHub API rate limit exceeded")
	} else if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(bodyBytes))
	}

	return bodyBytes, nil
}

// --- GitHub Data Fetchers ---

func getPRDiff(owner, repo, prNumber, token string) (string, error) {
	apiURL := fmt.Sprintf("https://api.github.com/repos/%s/%s/pulls/%s", owner, repo, prNumber)
	bytes, err := doGitHubRequest(apiURL, "application/vnd.github.v3.diff", token)
	if err != nil {
		return "", err
	}
	return string(bytes), nil
}

func getPRMetadata(owner, repo, prNumber, token string) (PRDetails, error) {
	apiURL := fmt.Sprintf("https://api.github.com/repos/%s/%s/pulls/%s", owner, repo, prNumber)
	bytes, err := doGitHubRequest(apiURL, "application/vnd.github.v3+json", token)

	var details PRDetails
	if err != nil {
		return details, err
	}

	err = json.Unmarshal(bytes, &details)
	return details, err
}

func getPRCommits(owner, repo, prNumber, token string) (string, error) {
	apiURL := fmt.Sprintf("https://api.github.com/repos/%s/%s/pulls/%s/commits", owner, repo, prNumber)
	bytes, err := doGitHubRequest(apiURL, "application/vnd.github.v3+json", token)
	if err != nil {
		return "", err
	}

	var commits []CommitItem
	if err = json.Unmarshal(bytes, &commits); err != nil {
		return "", err
	}

	var messages []string
	for _, c := range commits {
		messages = append(messages, "- "+c.Commit.Message)
	}
	return strings.Join(messages, "\n"), nil
}

// --- AI Generation ---

func generateReview(ctx context.Context, details PRDetails, commits, diffText, apiKey string) (string, error) {
	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		return "", fmt.Errorf("failed to create Gemini client: %v", err)
	}
	defer client.Close()

	model := client.GenerativeModel("gemini-2.5-pro")

	// The prompt now includes all the rich context we fetched
	prompt := fmt.Sprintf(`You are an expert Senior Software Engineer tasked with reviewing a pull request. Your goal is to provide constructive, actionable, and highly focused feedback.

Your tone should be collaborative and objective. You are providing feedback to a human developer, so be respectful but direct.

I will provide you with the PR Title, the PR Description, the Commit Messages, and the raw Code Diff.

### INSTRUCTIONS:
1.  **Analyze Intent:** Read the PR Title, Description, and Commits first. Understand what the author is *trying* to accomplish.
2.  **Dynamic Language Detection:** Identify the programming languages and frameworks being used based on the file paths and extensions in the diff. Apply the standard, recognized best practices and idiomatic patterns for those specific languages (e.g., idiomatic error handling in Go, Pythonic list comprehensions, or safe memory management in Rust).
3.  **Review the Diff:** Evaluate the code changes against the author's stated intent. Does the code actually do what they claim?
4.  **Focus on High-Impact Issues:**
    *   Logic errors, race conditions, or unhandled edge cases.
    *   Security vulnerabilities (e.g., injection flaws, poor data sanitization).
    *   Significant performance bottlenecks.
    *   Anti-patterns or deviations from standard language-specific conventions.
5.  **Evaluate Testing:** Check if the PR includes updates to test files. If core logic was changed or added without corresponding tests, flag it. Specifically recommend whether a **Unit Test** (for isolated functions/utilities) or an **End-to-End/Integration Test** (for API routes/workflows) would provide the most value for this specific change.
6.  **Strictly Ignore:**
    *   Minor stylistic choices, formatting, or syntax preferences (assume a linter handles this).
    *   Changes to auto-generated files (e.g., package-lock.json, go.sum, compiled assets).

Here is the context provided by the author:
**PR Title:** %s
**PR Description:**
%s

**Commit Messages:**
%s

### OUTPUT FORMAT:
Format your response in Markdown, using the following structure so I can easily copy/paste it into GitHub:

**High-Level Summary**
[1-2 sentences summarizing your overall impression of the PR and whether it achieves its goal safely.]

**Actionable Feedback**
*   **[Severity: Critical/Moderate/Minor]** - **File: filename**: Explain the issue clearly. Mention if it violates a language-specific best practice. Provide a short snippet of the suggested fix if applicable.
*   **[Question]** - If something is ambiguous, ask a clarifying question about the author's intent.

**Testing & Validation**
[Assess the test coverage in the PR. Clearly state if tests are missing for new logic, and recommend specifically what kind of test (Unit vs. E2E) should be written to validate the change safely.]

If the PR looks excellent and has no notable issues, simply output: "LGTM! The code aligns with the description, tests are sufficient, and I see no security, performance, or logic issues."

Here is the raw diff to review:
%s`, details.Title, details.Body, commits, diffText)

	fmt.Println("Analyzing the diff and context with AI... (this might take a few seconds)\n")

	resp, err := model.GenerateContent(ctx, genai.Text(prompt))
	if err != nil {
		return "", fmt.Errorf("failed to generate content: %v", err)
	}

	if len(resp.Candidates) > 0 && len(resp.Candidates[0].Content.Parts) > 0 {
		if textPart, ok := resp.Candidates[0].Content.Parts[0].(genai.Text); ok {
			return string(textPart), nil
		}
	}

	return "", fmt.Errorf("unexpected response format from Gemini API")
}

func main() {
	if len(os.Args) < 2 {
		fmt.Printf("Usage: %s <github-pr-url>\n", os.Args[0])
		os.Exit(1)
	}

	prURL := os.Args[1]
	geminiKey := os.Getenv("GEMINI_API_KEY")
	githubToken := os.Getenv("GITHUB_TOKEN")

	if geminiKey == "" {
		log.Fatal("Error: Please set the GEMINI_API_KEY environment variable.")
	}

	owner, repo, prNumber, err := parsePRURL(prURL)
	if err != nil {
		log.Fatalf("Error parsing URL: %v", err)
	}

	fmt.Printf("Fetching data for PR #%s from %s/%s...\n", prNumber, owner, repo)

	// Fetch all three pieces of context
	diffText, err := getPRDiff(owner, repo, prNumber, githubToken)
	if err != nil {
		log.Fatalf("Error fetching diff: %v", err)
	}

	details, err := getPRMetadata(owner, repo, prNumber, githubToken)
	if err != nil {
		log.Fatalf("Error fetching metadata: %v", err)
	}

	commits, err := getPRCommits(owner, repo, prNumber, githubToken)
	if err != nil {
		log.Fatalf("Error fetching commits: %v", err)
	}

	if strings.TrimSpace(diffText) == "" {
		fmt.Println("The diff is empty. Are there any file changes in this PR?")
		os.Exit(0)
	}

	ctx := context.Background()
	reviewOutput, err := generateReview(ctx, details, commits, diffText, geminiKey)
	if err != nil {
		log.Fatalf("AI Error: %v", err)
	}

	fmt.Println("\n--- AI Review Feedback ---")
	fmt.Println(reviewOutput)
	fmt.Println("--------------------------")
}
