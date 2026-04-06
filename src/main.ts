import { readFileSync } from "fs";
import * as core from "@actions/core";
import OpenAI from "openai";
import { Octokit } from "@octokit/rest";
import parseDiff, { File } from "parse-diff";
import minimatch from "minimatch";
import { VERSION } from "./version";

const GITHUB_TOKEN: string = core.getInput("GITHUB_TOKEN");
const OPENAI_API_KEY: string = core.getInput("OPENAI_API_KEY");
const OPENAI_API_MODEL: string = core.getInput("OPENAI_API_MODEL");
const MAX_COMMENTS_PER_PR: number = parseInt(core.getInput("MAX_COMMENTS") || "10", 10);

const octokit = new Octokit({ auth: GITHUB_TOKEN });

const openai = new OpenAI({
  apiKey: OPENAI_API_KEY,
});

interface PRDetails {
  owner: string;
  repo: string;
  pull_number: number;
  title: string;
  description: string;
  action: string;
  before?: string;
  after?: string;
}

async function getPRDetails(): Promise<PRDetails> {
  const { repository, number, action, before, after } = JSON.parse(
    readFileSync(process.env.GITHUB_EVENT_PATH || "", "utf8")
  );

  const prResponse = await octokit.pulls.get({
    owner: repository.owner.login,
    repo: repository.name,
    pull_number: number,
  });
  return {
    owner: repository.owner.login,
    repo: repository.name,
    pull_number: number,
    title: prResponse.data.title ?? "",
    description: prResponse.data.body ?? "",
    action,
    before,
    after,
  };
}

async function getDiff(
  owner: string,
  repo: string,
  pull_number: number
): Promise<string | null> {
  const response = await octokit.pulls.get({
    owner,
    repo,
    pull_number,
    mediaType: { format: "diff" },
  });
  // @ts-expect-error - response.data is a string
  return response.data;
}

async function analyzeCode(
  parsedDiff: File[],
  prDetails: PRDetails
): Promise<Array<{ body: string; path: string; line: number }>> {
  const comments: Array<{ body: string; path: string; line: number }> = [];

  for (const file of parsedDiff) {
    if (file.to === "/dev/null") continue; // Ignore deleted files
    if (file.chunks.length === 0) continue;

    // Send entire file diff at once for better context
    const prompt = createPrompt(file, prDetails);
    const aiResponse = await getAIResponse(prompt);
    if (aiResponse) {
      const newComments = createComment(file, aiResponse);
      if (newComments) {
        comments.push(...newComments);
      }
    }
  }
  return comments;
}

function createPrompt(file: File, prDetails: PRDetails): string {
  // Combine all chunks into a single diff for full file context
  const fullDiff = file.chunks
    .map((chunk) => {
      const chunkHeader = chunk.content;
      const chunkChanges = chunk.changes
        // @ts-expect-error - ln and ln2 exists where needed
        .map((c) => `${c.ln ? c.ln : c.ln2} ${c.content}`)
        .join("\n");
      return `${chunkHeader}\n${chunkChanges}`;
    })
    .join("\n\n");

  return `You are a strict code review agent.

STRICT RULES — follow them exactly:
1. **Only report issues you are confident about.** If there is meaningful doubt, do not report it.
2. **Focus ONLY on these categories:**
   - Bugs or potential bugs
   - Performance issues
   - Good-practice violations with real engineering impact
3. **Do NOT report:**
   - Pure style, naming, or formatting preferences
   - Warnings like "if you changed X, make sure Y is updated"
   - Missing comments or documentation
   - Speculative or uncertain concerns
4. For every reported issue, the reviewComment **must contain exactly these markdown sections in this order**:
   - \`### Issue\`
   - \`### Why this is a bug\` (required **only** when category is bug/potential bug; for performance/good-practice use \`### Why this matters\`)
   - \`### Suggested AI Fix Prompt\`
5. In \`### Suggested AI Fix Prompt\`, provide a copy-paste-ready prompt that includes:
   - namespace/module
   - file path
   - exact line number
   - a direct instruction describing the fix to implement

RESPONSE FORMAT (strict JSON):
{"reviews": [{"lineNumber": <line_number>, "reviewComment": "<markdown with required sections>"}]}

- If there are no valid findings in the allowed categories, return: {"reviews": []}
- Keep each section concise and actionable.
- Write in GitHub Markdown.

---

File: "${file.to}"
Pull request title: ${prDetails.title}
Pull request description:
${prDetails.description}

Complete file diff to review:

\`\`\`diff
${fullDiff}
\`\`\`
`;
}

// Models that support JSON response format
const JSON_MODE_MODELS = [
  "gpt-4o",
  "gpt-4o-mini",
  "gpt-4-turbo",
  "gpt-4-turbo-preview",
  "gpt-4-1106-preview",
  "gpt-4-0125-preview",
  "gpt-3.5-turbo-1106",
  "gpt-3.5-turbo-0125",
];

function supportsJsonMode(model: string): boolean {
  return JSON_MODE_MODELS.some((m) => model.startsWith(m));
}

async function getAIResponse(prompt: string): Promise<Array<{
  lineNumber: string;
  reviewComment: string;
}> | null> {
  const queryConfig = {
    model: OPENAI_API_MODEL,
    temperature: 0.2,
    max_tokens: 4096, // Increased for full file context
    top_p: 1,
    frequency_penalty: 0,
    presence_penalty: 0,
  };

  try {
    const response = await openai.chat.completions.create({
      ...queryConfig,
      // Enable JSON mode for models that support it
      ...(supportsJsonMode(OPENAI_API_MODEL)
        ? { response_format: { type: "json_object" } }
        : {}),
      messages: [
        {
          role: "system",
          content: prompt,
        },
      ],
    });

    const res = response.choices[0].message?.content?.trim() || "{}";
    return JSON.parse(res).reviews;
  } catch (error) {
    console.error("Error:", error);
    return null;
  }
}

function createComment(
  file: File,
  aiResponses: Array<{
    lineNumber: string;
    reviewComment: string;
  }>
): Array<{ body: string; path: string; line: number }> {
  return aiResponses.flatMap((aiResponse) => {
    if (!file.to) {
      return [];
    }
    return {
      body: `${aiResponse.reviewComment}\n\n---\n*AI Code Reviewer · version \`${VERSION}\`*`,
      path: file.to,
      line: Number(aiResponse.lineNumber),
    };
  });
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Post review comments with rate limiting and retry logic.
 * - Batches all comments into a single review (1 API call)
 * - Respects GitHub rate limits with exponential backoff
 * - Limits total comments to MAX_COMMENTS_PER_PR
 */
async function safeCreateReview(
  owner: string,
  repo: string,
  pull_number: number,
  comments: Array<{ body: string; path: string; line: number }>
) {
  // Limit comments to avoid noise and reduce API usage
  const limitedComments = comments.slice(0, MAX_COMMENTS_PER_PR);
  
  if (comments.length > MAX_COMMENTS_PER_PR) {
    console.log(`⚠️ Limiting comments from ${comments.length} to ${MAX_COMMENTS_PER_PR}`);
  }

  if (limitedComments.length === 0) {
    console.log("No comments to post.");
    return;
  }

  const maxRetries = 3;
  let attempt = 0;

  while (attempt < maxRetries) {
    try {
      console.log(`📝 Posting ${limitedComments.length} comments as a single review (attempt ${attempt + 1})...`);
      
      const response = await octokit.pulls.createReview({
        owner,
        repo,
        pull_number,
        comments: limitedComments,
        event: "COMMENT",
      });

      // Check rate limit headers for proactive waiting
      const remaining = Number(response.headers["x-ratelimit-remaining"] || 100);
      const reset = Number(response.headers["x-ratelimit-reset"] || 0);

      console.log(`✅ Review posted successfully. Rate limit remaining: ${remaining}`);

      if (remaining < 10 && reset > 0) {
        const now = Math.floor(Date.now() / 1000);
        const waitSec = Math.max(0, reset - now);
        console.log(`⚠️ Low rate limit (${remaining}), would wait ${waitSec}s before next call`);
      }

      return; // Success, exit
    } catch (err: any) {
      const status = err.status || err.response?.status;
      const retryAfter = err.response?.headers?.["retry-after"];

      // Handle rate limiting (403 or 429)
      if (status === 403 || status === 429) {
        attempt++;
        const baseDelay = retryAfter ? parseInt(retryAfter, 10) * 1000 : 60000;
        const backoffDelay = baseDelay * Math.pow(2, attempt - 1); // Exponential backoff
        
        console.log(`⏳ Rate limited (${status}). Waiting ${backoffDelay / 1000}s before retry ${attempt}/${maxRetries}...`);
        await sleep(backoffDelay);
      } else {
        // Non-rate-limit error, don't retry
        console.error("GitHub API error:", err.message || err);
        throw err;
      }
    }
  }

  console.error(`❌ Failed to post review after ${maxRetries} attempts due to rate limiting.`);
}

async function main() {
  console.log("starting review process...");
  const prDetails = await getPRDetails();
  let diff: string | null;
  // const eventData = JSON.parse(
  //   readFileSync(process.env.GITHUB_EVENT_PATH ?? "", "utf8")
  // );

  if (prDetails.action === "opened") {
    diff = await getDiff(
      prDetails.owner,
      prDetails.repo,
      prDetails.pull_number
    );
  } else if (prDetails.action === "synchronize") {
    const newBaseSha = prDetails.before;
    const newHeadSha = prDetails.after;

    if (!newBaseSha || !newHeadSha) {
      console.log("Missing base or head SHA for synchronize event");
      return;
    }

    const response = await octokit.repos.compareCommits({
      headers: {
        accept: "application/vnd.github.v3.diff",
      },
      owner: prDetails.owner,
      repo: prDetails.repo,
      base: newBaseSha,
      head: newHeadSha,
    });

    diff = String(response.data);
  } else {
    console.log("Unsupported event:", process.env.GITHUB_EVENT_NAME);
    return;
  }

  if (!diff) {
    console.log("No diff found");
    return;
  }

  const parsedDiff = parseDiff(diff);

  const excludePatterns = core
    .getInput("exclude")
    .split(",")
    .map((s) => s.trim());

  const filteredDiff = parsedDiff.filter((file) => {
    return !excludePatterns.some((pattern) =>
      minimatch(file.to ?? "", pattern)
    );
  });
  
  console.log("calling ai to analyze code...");
  const comments = await analyzeCode(filteredDiff, prDetails);

  console.log(`AI suggested ${comments.length} comments.`);
  if (comments.length > 0) {
    await safeCreateReview(
      prDetails.owner,
      prDetails.repo,
      prDetails.pull_number,
      comments
    );
  }
}

main().catch((error) => {
  console.error("Error:", error);
  process.exit(1);
});
