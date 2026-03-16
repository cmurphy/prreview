package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/iterator"
	"google.golang.org/api/option"
)

// --- Structs for GitHub JSON Responses ---

type PRDetails struct {
	Title string `json:"title"`
	Body  string `json:"body"`
	Base  struct {
		Sha string `json:"sha"`
	} `json:"base"`
}

type CommitItem struct {
	Commit struct {
		Message string `json:"message"`
	} `json:"commit"`
}

// Spinner represents a simple, zero-dependency terminal loading indicator
type Spinner struct {
	stopChan chan struct{}
}

// StartSpinner begins the background animation with a message
func StartSpinner(message string) *Spinner {
	s := &Spinner{stopChan: make(chan struct{})}

	go func() {
		// Braille spinner frames for a smooth circle
		frames := []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}
		i := 0
		for {
			select {
			case <-s.stopChan:
				return
			default:
				// \r moves the cursor to the start of the line so it overwrites itself
				fmt.Printf("\r\033[36m%s\033[m %s", frames[i], message)
				time.Sleep(100 * time.Millisecond)
				i = (i + 1) % len(frames)
			}
		}
	}()

	return s
}

// Stop halts the animation and clears the line for the final output
func (s *Spinner) Stop() {
	close(s.stopChan)
	// \033[2K clears the entire current line, \r resets the cursor
	fmt.Printf("\r\033[2K")
}

// --- Helper Functions ---

// parseGitHubURL extracts the owner, repository, and optionally the PR number from a GitHub URL.
func parseGitHubURL(rawURL string) (owner, repo, prNumber string, err error) {
	u, err := url.Parse(rawURL)
	if err != nil {
		return "", "", "", fmt.Errorf("failed to parse URL: %w", err)
	}

	if u.Host != "github.com" {
		return "", "", "", fmt.Errorf("unsupported host: %s, expected github.com", u.Host)
	}

	// Trim leading/trailing slashes to prevent empty strings in the slice
	path := strings.Trim(u.Path, "/")
	parts := strings.Split(path, "/")

	if len(parts) < 2 {
		return "", "", "", fmt.Errorf("invalid GitHub URL, missing owner or repo: %s", u.Path)
	}

	owner = parts[0]
	// Strip .git just in case a clone URL is passed to the init command
	repo = strings.TrimSuffix(parts[1], ".git")

	// Check if this is a pull request URL (format: /owner/repo/pull/123)
	if len(parts) >= 4 && parts[2] == "pull" {
		prNumber = parts[3]
	}

	return owner, repo, prNumber, nil
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

// shouldIgnore determines if a file should be excluded from the LLM context
func shouldIgnore(path string) bool {
	// Ignore common dependency and build directories
	ignoredDirs := []string{"node_modules", "vendor", "dist", "build", ".git", ".github"}
	parts := strings.Split(path, "/")
	for _, part := range parts {
		for _, dir := range ignoredDirs {
			if part == dir {
				return true
			}
		}
	}

	// Ignore lockfiles and large generated files
	baseName := filepath.Base(path)
	if baseName == "package-lock.json" || baseName == "yarn.lock" || baseName == "go.sum" {
		return true
	}

	// Ignore non-text extensions
	ext := strings.ToLower(filepath.Ext(path))
	ignoredExts := map[string]bool{
		".png": true, ".jpg": true, ".jpeg": true, ".gif": true, ".svg": true, ".ico": true,
		".mp4": true, ".mp3": true, ".wav": true,
		".zip": true, ".tar": true, ".gz": true,
		".pdf": true, ".exe": true, ".dll": true, ".so": true, ".dylib": true,
		".bin": true, ".wasm": true,
	}

	return ignoredExts[ext]
}

// getProfileCachePath returns the path for the repository's AI profile
func getProfileCachePath(owner, repo string) (string, error) {
	cacheDir, err := os.UserCacheDir()
	if err != nil {
		return "", err
	}

	repoCacheDir := filepath.Join(cacheDir, "prreview", owner, repo)
	if err := os.MkdirAll(repoCacheDir, 0755); err != nil {
		return "", err
	}

	return filepath.Join(repoCacheDir, "ai-profile.md"), nil
}

// fetchRepoConfigs tries to download key architectural files from the repository
func fetchRepoConfigs(owner, repo, token string) string {
	// These files contain the "rules" of the repo without the noise of business logic or tests
	targetFiles := []string{
		"README.md",
		"CONTRIBUTING.md",
		"go.mod",
		"package.json",
		"tsconfig.json",
		".eslintrc.json",
		"pyproject.toml",
		"Cargo.toml",
	}

	var configs strings.Builder

	for _, file := range targetFiles {
		apiURL := fmt.Sprintf("https://api.github.com/repos/%s/%s/contents/%s", owner, repo, file)
		// Requesting the raw media type so GitHub doesn't send base64 encoded JSON back
		bytesData, err := doGitHubRequest(apiURL, "application/vnd.github.v3.raw", token)
		if err == nil && len(bytesData) > 0 {
			configs.WriteString(fmt.Sprintf("\n--- FILE: %s ---\n%s\n", file, string(bytesData)))
		}
	}

	return configs.String()
}

// openInEditor opens a temporary file in the user's default text editor
func openInEditor(initialContent string) (string, error) {
	editor := os.Getenv("EDITOR")
	if editor == "" {
		if runtime.GOOS == "windows" {
			editor = "notepad"
		} else {
			editor = "nano" // safe fallback for Unix
		}
	}

	// Create a temporary file
	tmpFile, err := os.CreateTemp("", "pr-reviewer-profile-*.md")
	if err != nil {
		return "", fmt.Errorf("failed to create temp file: %v", err)
	}
	defer os.Remove(tmpFile.Name()) // Clean up afterwards

	// Write the AI's initial draft to the file
	if _, err := tmpFile.Write([]byte(initialContent)); err != nil {
		return "", fmt.Errorf("failed to write to temp file: %v", err)
	}
	tmpFile.Close() // Close it so the editor can safely open it

	fmt.Printf("Opening draft profile in %s for manual review...\n", editor)

	// Launch the editor and attach standard input/output
	cmd := exec.Command(editor, tmpFile.Name())
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("editor execution failed: %v", err)
	}

	// Read the final, human-edited content
	finalBytes, err := os.ReadFile(tmpFile.Name())
	if err != nil {
		return "", fmt.Errorf("failed to read edited file: %v", err)
	}

	return string(finalBytes), nil
}

// extractBaseFiles parses the diff to find files that existed before the PR
func extractBaseFiles(diffText string) []string {
	var files []string
	seen := make(map[string]bool)

	lines := strings.Split(diffText, "\n")
	for _, line := range lines {
		if strings.HasPrefix(line, "--- a/") {
			path := strings.TrimPrefix(line, "--- a/")
			if path != "/dev/null" && !seen[path] {
				seen[path] = true
				files = append(files, path)
			}
		}
	}
	return files
}

// fetchLocalContext retrieves the full, original text of the modified files
func fetchLocalContext(owner, repo, baseSHA, token string, files []string) string {
	var localContext strings.Builder

	for _, file := range files {
		// Skip binaries, lockfiles, and tests (to prevent test-hallucination bleed)
		if shouldIgnore(file) {
			continue
		}

		apiURL := fmt.Sprintf("https://api.github.com/repos/%s/%s/contents/%s?ref=%s", owner, repo, file, baseSHA)

		// Use the raw media type so GitHub returns plain text instead of base64 JSON
		bytesData, err := doGitHubRequest(apiURL, "application/vnd.github.v3.raw", token)
		if err != nil {
			fmt.Printf("Notice: Could not fetch base file %s (error: %v).\n", file, err)
			continue
		}

		localContext.WriteString(fmt.Sprintf("\n--- ORIGINAL FILE: %s ---\n%s\n", file, string(bytesData)))
	}

	return localContext.String()
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

// runInitCommand handles the 'init' subcommand flow
func runInitCommand(ctx context.Context, owner, repo, githubToken, geminiKey string) error {
	profilePath, err := getProfileCachePath(owner, repo)
	if err != nil {
		return err
	}

	fmt.Printf("Analyzing configuration files for %s/%s...\n", owner, repo)
	configs := fetchRepoConfigs(owner, repo, githubToken)
	if configs == "" {
		fmt.Println("No standard config files found. The AI will start from scratch.")
	}

	fmt.Println("Asking AI to draft the initial Codebase Profile. Please wait...")

	// The prompt telling Gemini to write the draft rulebook
	prompt := fmt.Sprintf(`You are a Staff Software Engineer. I am going to give you the configuration files, manifest files, and documentation for a code repository.

Your job is to write a strict, concise "Rulebook" for this codebase. Reviewers will use this rulebook to evaluate Pull Requests.

Do NOT summarize what the app does. ONLY extract architectural rules, library mandates, and testing conventions.

Focus on:
1. Primary languages and frameworks.
2. Mandated libraries (e.g., "Uses zap for logging, do not use standard log").
3. Linting and formatting rules implied by the configs.
4. Testing frameworks and conventions.

Output the rules as a clean Markdown list. Keep it under 300 words.

Here are the configuration files:
%s`, configs)

	client, err := genai.NewClient(ctx, option.WithAPIKey(geminiKey))
	if err != nil {
		return err
	}
	defer client.Close()

	model := client.GenerativeModel("gemini-2.5-pro")
	resp, err := model.GenerateContent(ctx, genai.Text(prompt))
	if err != nil {
		return err
	}

	// Extract the AI's text
	var draftProfile strings.Builder
	draftProfile.WriteString("# AI Codebase Profile for " + owner + "/" + repo + "\n")
	draftProfile.WriteString("<!-- EDIT THIS FILE. Delete hallucinations. Add unwritten team rules. Save and close to continue. -->\n\n")

	for _, part := range resp.Candidates[0].Content.Parts {
		if textPart, ok := part.(genai.Text); ok {
			draftProfile.WriteString(string(textPart))
		}
	}

	// 🛑 THE HUMAN IN THE LOOP 🛑
	finalProfile, err := openInEditor(draftProfile.String())
	if err != nil {
		return err
	}

	// Save the edited profile to the persistent cache
	if err := os.WriteFile(profilePath, []byte(finalProfile), 0644); err != nil {
		return fmt.Errorf("failed to save profile to cache: %v", err)
	}

	fmt.Printf("\n✅ Profile saved successfully to %s\n", profilePath)
	return nil
}

func generateReview(ctx context.Context, details PRDetails, commits, diffText, apiKey, customPrompt, repoProfile, localContextText string, numDrafts uint) error {
	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		return fmt.Errorf("failed to create Gemini client: %v", err)
	}
	defer client.Close()

	model := client.GenerativeModel("gemini-2.5-pro")

	// Conditionally inject the user's specific nudge
	customInstructions := ""
	if customPrompt != "" {
		customInstructions = fmt.Sprintf("\n### SPECIAL INSTRUCTIONS FROM THE REVIEWER:\nPlease pay special attention to the following concern/question when reviewing this PR:\n\"%s\"\n", customPrompt)
	}

	basePrompt := fmt.Sprintf(`You are an expert Senior Software Engineer and Technical Writer reviewing a pull request. Your goal is to provide constructive, actionable, and highly focused feedback.

Your tone should be collaborative and objective. You are providing feedback to a human developer, so be respectful but direct.

Here is the AI Codebase Profile containing the global architectural rules for this repository:
<repository_rules>
%s
</repository_rules>

Here is the original, full text of the files being modified. Use this to understand the surrounding logic, imports, and struct definitions. DO NOT review this original code for bugs.
<original_files>
%s
</original_files>

Here is the context provided by the author:
<pull_request_metadata>
**PR Title:** %s
**PR Description:** %s
**Commit Messages:** %s
</pull_request_metadata>

%s

Here are the actual code changes. THIS IS THE ONLY CODE YOU ARE REVIEWING:
<pull_request_diff>
%s
</pull_request_diff>

### STRICT INSTRUCTIONS:
1.  **Enforce Repo Rules:** Strictly apply the guidelines found in <repository_rules>.
2.  **Context Switching (Code vs. Docs):** Analyze the file extensions in the <pull_request_diff>. 
    *   For Code: Check for logic errors, race conditions, unhandled edge cases, security vulnerabilities, performance bottlenecks, and anti-patterns.
    *   For Docs (Markdown/Txt): Focus on technical correctness, clarity, and readability.
3.  **Intent Matching:** Ensure the changes in the diff actually align with the author's stated intent in the <pull_request_metadata>.
4.  **Local Context:** Use the <original_files> ONLY to verify that the changes correctly interact with existing functions and structs. Do not point out bugs in, or credit the author for, code found here.
5.  **Evaluate Testing:** Assess if the tests added or modified in the <pull_request_diff> are sufficient and appropriate for the new logic. If core logic was added without tests, explicitly flag it.
6.  **Strictly Ignore:** Minor stylistic choices, formatting preferences, and auto-generated files.

### OUTPUT FORMAT:
**High-Level Summary**
[1-2 sentences summarizing your overall impression and if the PR achieves its stated goal.]

**Actionable Feedback**
* **[Severity: Critical/Moderate/Minor]** - **File: filename**: Explain the issue clearly and provide a snippet of the suggested fix.
* **[Question]** - If something is ambiguous or intent is unclear, ask a clarifying question.

**Testing & Validation**
[Assess the test coverage added in the diff. Recommend specific tests if missing. Output "N/A - Documentation only" for pure docs PRs.]

**Other Notes**
[Add your private notes about the reviewer's concern or question provided in the special instructions, if such instructions were given.]

If the PR looks excellent and has no notable issues, simply output: "LGTM! The code aligns with the description, tests are sufficient, and I see no security, performance, or logic issues."
`, repoProfile, localContextText, details.Title, details.Body, commits, customInstructions, diffText)

	// --- PATH A: Fast Single-Pass Review (Cost/Time Efficient) ---
	if numDrafts == 1 {
		fmt.Println("\n--------------------------------------------------")
		spinner := StartSpinner("AI is analyzing the PR...")

		iter := model.GenerateContentStream(ctx, genai.Text(basePrompt))

		firstTokenReceived := false
		for {
			resp, err := iter.Next()

			// iterator.Done means the stream has finished successfully
			if err == iterator.Done {
				break
			}
			if err != nil {
				if !firstTokenReceived {
					spinner.Stop()
				}
				return fmt.Errorf("streaming error: %v", err)
			}

			// The moment we get our first piece of data, stop the spinner
			if !firstTokenReceived {
				spinner.Stop()
				firstTokenReceived = true
			}

			if len(resp.Candidates) > 0 && len(resp.Candidates[0].Content.Parts) > 0 {
				if textPart, ok := resp.Candidates[0].Content.Parts[0].(genai.Text); ok {
					fmt.Print(string(textPart))
				}
			}
		}

		fmt.Println("\n--------------------------------------------------")
		return nil
	}

	// --- PATH B: Concurrent Ensemble Review (High Quality / High Recall) ---
	fmt.Println("\n--------------------------------------------------")
	aiSpinner := StartSpinner(fmt.Sprintf("AI ensemble (%d drafts) is analyzing the PR concurrently...", numDrafts))

	drafts := make([]string, numDrafts)
	var wg sync.WaitGroup
	var mu sync.Mutex
	var fetchErr error

	for i := uint(0); i < numDrafts; i++ {
		wg.Add(1)
		go func(index uint) {
			defer wg.Done()

			// Use a separate model instance for the concurrent calls
			draftModel := client.GenerativeModel("gemini-2.5-pro")
			resp, err := draftModel.GenerateContent(ctx, genai.Text(basePrompt))

			mu.Lock()
			defer mu.Unlock()

			if err != nil {
				fetchErr = err
				return
			}
			if len(resp.Candidates) > 0 && len(resp.Candidates[0].Content.Parts) > 0 {
				if textPart, ok := resp.Candidates[0].Content.Parts[0].(genai.Text); ok {
					drafts[index] = string(textPart)
				}
			}
		}(i)
	}

	wg.Wait()
	aiSpinner.Stop()

	if fetchErr != nil {
		return fmt.Errorf("ensemble analysis failed: %v", fetchErr)
	}

	// Build the Synthesis Prompt dynamically based on how many drafts were requested
	var draftsContent strings.Builder
	for i, draft := range drafts {
		draftsContent.WriteString(fmt.Sprintf("<draft_%d>\n%s\n</draft_%d>\n\n", i+1, draft, i+1))
	}

	mergeSpinner := StartSpinner("Synthesizing and deduplicating final report...")

	synthesisPrompt := fmt.Sprintf(`You are a Lead Staff Engineer reviewing a Pull Request.
I have asked %d Senior Engineers to independently review the same PR. Here are their draft reports:

%s

Your job is to merge these drafts into a single, cohesive, master review.

### STRICT INSTRUCTIONS:
1.  **Deduplicate:** Semantically combine identical findings. Do not repeat the same issue twice.
2.  **Filter Noise:** If one draft hallucinates an issue that is factually incorrect or contradicts the other drafts, discard it.
3.  **Consolidate:** Keep the tone collaborative, objective, and direct.

### OUTPUT FORMAT:
Output using the exact same format as the drafts:
**High-Level Summary**
**Actionable Feedback** (Grouped logically)
**Testing & Validation**
`, numDrafts, draftsContent.String())

	iter := model.GenerateContentStream(ctx, genai.Text(synthesisPrompt))
	firstToken := false

	for {
		resp, err := iter.Next()
		if err == iterator.Done {
			break
		}
		if err != nil {
			if !firstToken {
				mergeSpinner.Stop()
			}
			return fmt.Errorf("streaming error: %v", err)
		}
		if !firstToken {
			mergeSpinner.Stop()
			firstToken = true
		}
		if len(resp.Candidates) > 0 && len(resp.Candidates[0].Content.Parts) > 0 {
			if textPart, ok := resp.Candidates[0].Content.Parts[0].(genai.Text); ok {
				fmt.Print(string(textPart))
			}
		}
	}

	fmt.Println("\n--------------------------------------------------")
	return nil
}

func main() {
	// Define the optional flag
	var customPrompt string
	var drafts uint
	flag.StringVar(&customPrompt, "p", "", "Optional: Specific instructions or questions for the AI regarding this PR")
	flag.StringVar(&customPrompt, "prompt", "", "Optional: Specific instructions or questions for the AI regarding this PR")
	flag.UintVar(&drafts, "drafts", 1, "Optional: Number of AI iterations to run to synthesize feedback (increases costs)")

	// Parse the flags
	flag.Parse()

	// After flags are parsed, flag.Args() holds the remaining non-flag arguments (the URL)
	args := flag.Args()
	if len(args) < 1 {
		fmt.Printf("Usage: %s [flags] <github-pr-url>\n", os.Args[0])
		fmt.Println("\nFlags:")
		flag.PrintDefaults()
		fmt.Println("\nExample: prreview -p \"Did they properly handle null values in the user payload?\" --drafts 3 https://github.com/facebook/react/pull/28741")
		os.Exit(1)
	}

	ctx := context.Background()

	githubToken := os.Getenv("GITHUB_TOKEN")
	geminiKey := os.Getenv("GEMINI_API_KEY")
	if geminiKey == "" {
		log.Fatal("Error: Please set the GEMINI_API_KEY environment variable.")
	}

	if os.Args[1] == "init" {
		if len(os.Args) < 3 {
			log.Fatalf("Usage: prreview init <repo-url>")
		}
		repoURL := os.Args[2]
		owner, repo, _, err := parseGitHubURL(repoURL)
		if err != nil {
			log.Fatalf("Invalid URL: %v", err)
		}

		if err := runInitCommand(ctx, owner, repo, githubToken, geminiKey); err != nil {
			log.Fatalf("Init failed: %v", err)
		}
		return // Exit after init finishes
	}

	prURL := args[0]

	owner, repo, prNumber, err := parseGitHubURL(prURL)
	if err != nil {
		log.Fatalf("Error parsing URL: %v", err)
	}

	// Check if the human-verified profile exists in cache
	profilePath, _ := getProfileCachePath(owner, repo)
	repoProfile := ""
	if cachedProfile, err := os.ReadFile(profilePath); err == nil {
		repoProfile = string(cachedProfile)
	} else {
		fmt.Printf("⚠️ No Codebase Profile found. Run 'prreview init %s' to create one for better reviews.\n\n", "https://github.com/"+owner+"/"+repo)
		repoProfile = "No specific repository rules provided. Apply general best practices."
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

	spinner := StartSpinner("Fetching local file context for the modified files...")
	modifiedFiles := extractBaseFiles(diffText)
	localContextText := fetchLocalContext(owner, repo, details.Base.Sha, githubToken, modifiedFiles)
	spinner.Stop()
	fmt.Println("✅ Local file context fetched.")

	err = generateReview(ctx, details, commits, diffText, geminiKey, customPrompt, repoProfile, localContextText, drafts)
	if err != nil {
		log.Fatalf("AI Error: %v", err)
	}
}
