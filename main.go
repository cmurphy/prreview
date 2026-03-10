package main

import (
	"context"
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

// parsePRURL extracts the owner, repo, and PR number from a standard GitHub PR URL.
func parsePRURL(prURL string) (owner, repo, prNumber string, err error) {
	u, err := url.Parse(prURL)
	if err != nil {
		return "", "", "", fmt.Errorf("invalid URL: %v", err)
	}

	// Remove leading/trailing slashes and split
	path := strings.Trim(u.Path, "/")
	parts := strings.Split(path, "/")

	if len(parts) < 4 || parts[2] != "pull" {
		return "", "", "", fmt.Errorf("URL does not appear to be a valid GitHub Pull Request URL. Expected format: https://github.com/owner/repo/pull/123")
	}

	return parts[0], parts[1], parts[3], nil
}

// getPRDiff fetches the raw diff text from the GitHub API.
func getPRDiff(owner, repo, prNumber, token string) (string, error) {
	apiURL := fmt.Sprintf("https://api.github.com/repos/%s/%s/pulls/%s", owner, repo, prNumber)

	req, err := http.NewRequest("GET", apiURL, nil)
	if err != nil {
		return "", err
	}

	// The Accept header is required to get the raw diff
	req.Header.Set("Accept", "application/vnd.github.v3.diff")

	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	bodyString := string(bodyBytes)

	if resp.StatusCode == http.StatusForbidden && strings.Contains(strings.ToLower(bodyString), "rate limit") {
		return "", fmt.Errorf("GitHub API rate limit exceeded. Try setting the GITHUB_TOKEN environment variable")
	} else if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("error fetching PR (status %d): %s", resp.StatusCode, bodyString)
	}

	return bodyString, nil
}

// generateReview sends the diff to Gemini and returns the feedback.
func generateReview(ctx context.Context, diffText, apiKey string) (string, error) {
	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		return "", fmt.Errorf("failed to create Gemini client: %v", err)
	}
	defer client.Close()

	model := client.GenerativeModel("gemini-2.5-pro")

	prompt := fmt.Sprintf(`You are an expert senior software engineer. Please review the following code diff from a pull request.
    
Focus on:
1. Logic errors or bugs.
2. Security vulnerabilities.
3. Performance issues.

Do not nitpick minor stylistic changes. Format your response in clear bullet points that I can easily copy and paste into a GitHub review.

Here is the diff:
%s`, diffText)

	fmt.Println("Analyzing the diff with AI... (this might take a few seconds)\n")

	resp, err := model.GenerateContent(ctx, genai.Text(prompt))
	if err != nil {
		return "", fmt.Errorf("failed to generate content: %v", err)
	}

	// Safely extract the text from the response structure
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
		fmt.Println("Example: pr-reviewer https://github.com/facebook/react/pull/28741")
		os.Exit(1)
	}

	prURL := os.Args[1]

	geminiKey := os.Getenv("GEMINI_API_KEY")
	githubToken := os.Getenv("GITHUB_TOKEN")

	if geminiKey == "" {
		log.Fatal("Error: Please set the GEMINI_API_KEY environment variable.")
	}

	if githubToken == "" {
		fmt.Println("Note: GITHUB_TOKEN is not set. You are limited to 60 GitHub API requests per hour.")
	}

	owner, repo, prNumber, err := parsePRURL(prURL)
	if err != nil {
		log.Fatalf("Error parsing URL: %v", err)
	}

	fmt.Printf("Fetching PR #%s from %s/%s...\n", prNumber, owner, repo)

	diffText, err := getPRDiff(owner, repo, prNumber, githubToken)
	if err != nil {
		log.Fatalf("Error: %v", err)
	}

	if strings.TrimSpace(diffText) == "" {
		fmt.Println("The diff is empty. Are there any file changes in this PR?")
		os.Exit(0)
	}

	ctx := context.Background()
	reviewOutput, err := generateReview(ctx, diffText, geminiKey)
	if err != nil {
		log.Fatalf("AI Error: %v", err)
	}

	fmt.Println("--- AI Review Feedback ---")
	fmt.Println(reviewOutput)
	fmt.Println("--------------------------")
}
