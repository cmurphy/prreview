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
	prompt := fmt.Sprintf(`You are an expert senior software engineer. Please review the following code diff from a pull request.

Here is the context provided by the author:
**PR Title:** %s
**PR Description:**
%s

**Commit Messages:**
%s

Focus your review on:
1. Logic errors or bugs.
2. Security vulnerabilities.
3. Performance issues.
4. Whether the code actually accomplishes what the PR description and commits claim it does.

Do not nitpick minor stylistic changes. Format your response in clear bullet points that I can easily copy and paste into a GitHub review.

Here is the diff:
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
