/**
 * NanoClaw Agent Runner
 * Runs inside a container, receives config via stdin, outputs result to stdout
 *
 * Input protocol:
 *   Stdin: Full ContainerInput JSON (read until EOF)
 *   IPC:   Follow-up messages written as JSON files to /workspace/ipc/input/
 *          Files: {type:"message", text:"..."}.json
 *          Sentinel: /workspace/ipc/input/_close
 *
 * Stdout protocol:
 *   Each result is wrapped in OUTPUT_START_MARKER / OUTPUT_END_MARKER pairs.
 *   Multiple results may be emitted during a single long-lived session.
 */

import { execFile } from 'child_process';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

import { Codex, Thread } from '@openai/codex-sdk';
import type { ThreadOptions } from '@openai/codex-sdk';

interface ContainerInput {
  prompt: string;
  sessionId?: string;
  groupFolder: string;
  chatJid: string;
  isMain: boolean;
  isScheduledTask?: boolean;
  assistantName?: string;
  script?: string;
}

interface ContainerOutput {
  status: 'success' | 'error';
  result: string | null;
  newSessionId?: string;
  error?: string;
}

interface ScriptResult {
  wakeAgent: boolean;
  data?: unknown;
}

const IPC_INPUT_DIR = '/workspace/ipc/input';
const IPC_INPUT_CLOSE_SENTINEL = path.join(IPC_INPUT_DIR, '_close');
const IPC_POLL_MS = 500;
const SCRIPT_TIMEOUT_MS = 30_000;

const OUTPUT_START_MARKER = '---NANOCLAW_OUTPUT_START---';
const OUTPUT_END_MARKER = '---NANOCLAW_OUTPUT_END---';

function writeOutput(output: ContainerOutput): void {
  console.log(OUTPUT_START_MARKER);
  console.log(JSON.stringify(output));
  console.log(OUTPUT_END_MARKER);
}

function log(message: string): void {
  console.error(`[agent-runner] ${message}`);
}

async function readStdin(): Promise<string> {
  return new Promise((resolve, reject) => {
    let data = '';
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', (chunk) => {
      data += chunk;
    });
    process.stdin.on('end', () => resolve(data));
    process.stdin.on('error', reject);
  });
}

function appendConversationArchive(prompt: string, response: string): void {
  try {
    const conversationsDir = '/workspace/group/conversations';
    fs.mkdirSync(conversationsDir, { recursive: true });

    const day = new Date().toISOString().split('T')[0];
    const filePath = path.join(conversationsDir, `${day}.md`);

    const safePrompt =
      prompt.length > 5000 ? `${prompt.slice(0, 5000)}...` : prompt;
    const safeResponse =
      response.length > 5000 ? `${response.slice(0, 5000)}...` : response;

    const block = [
      `## ${new Date().toISOString()}`,
      '',
      '**User**',
      '',
      '```text',
      safePrompt,
      '```',
      '',
      '**Assistant**',
      '',
      safeResponse,
      '',
      '---',
      '',
    ].join('\n');

    fs.appendFileSync(filePath, block);
  } catch (err) {
    log(
      `Failed to archive conversation turn: ${err instanceof Error ? err.message : String(err)}`,
    );
  }
}

function shouldClose(): boolean {
  if (fs.existsSync(IPC_INPUT_CLOSE_SENTINEL)) {
    try {
      fs.unlinkSync(IPC_INPUT_CLOSE_SENTINEL);
    } catch {
      // ignore
    }
    return true;
  }
  return false;
}

function drainIpcInput(): string[] {
  try {
    fs.mkdirSync(IPC_INPUT_DIR, { recursive: true });
    const files = fs
      .readdirSync(IPC_INPUT_DIR)
      .filter((file) => file.endsWith('.json'))
      .sort();

    const messages: string[] = [];
    for (const file of files) {
      const filePath = path.join(IPC_INPUT_DIR, file);
      try {
        const data = JSON.parse(fs.readFileSync(filePath, 'utf-8')) as {
          type?: string;
          text?: string;
        };
        fs.unlinkSync(filePath);
        if (data.type === 'message' && data.text) {
          messages.push(data.text);
        }
      } catch (err) {
        log(
          `Failed to process input file ${file}: ${err instanceof Error ? err.message : String(err)}`,
        );
        try {
          fs.unlinkSync(filePath);
        } catch {
          // ignore
        }
      }
    }

    return messages;
  } catch (err) {
    log(`IPC drain error: ${err instanceof Error ? err.message : String(err)}`);
    return [];
  }
}

function waitForIpcMessage(): Promise<string | null> {
  return new Promise((resolve) => {
    const poll = () => {
      if (shouldClose()) {
        resolve(null);
        return;
      }

      const messages = drainIpcInput();
      if (messages.length > 0) {
        resolve(messages.join('\n'));
        return;
      }

      setTimeout(poll, IPC_POLL_MS);
    };

    poll();
  });
}

function readFirstExistingFile(paths: string[]): string | undefined {
  for (const candidate of paths) {
    if (fs.existsSync(candidate)) {
      return fs.readFileSync(candidate, 'utf-8');
    }
  }
  return undefined;
}

function buildInstructions(containerInput: ContainerInput): string | undefined {
  const groupInstructions = readFirstExistingFile([
    '/workspace/group/AGENTS.md',
    '/workspace/group/CLAUDE.md',
  ]);
  const globalInstructions = readFirstExistingFile([
    '/workspace/global/AGENTS.md',
    '/workspace/global/CLAUDE.md',
  ]);

  if (containerInput.isMain) {
    return groupInstructions?.trim();
  }

  if (!groupInstructions && !globalInstructions) {
    return undefined;
  }

  if (groupInstructions && globalInstructions) {
    return [
      groupInstructions.trim(),
      '',
      '---',
      'Global instructions:',
      globalInstructions.trim(),
    ].join('\n');
  }

  return (groupInstructions || globalInstructions)?.trim();
}

function buildThreadOptions(containerInput: ContainerInput): ThreadOptions {
  const additionalDirectories: string[] = [];

  if (containerInput.isMain && fs.existsSync('/workspace/project')) {
    additionalDirectories.push('/workspace/project');
  }
  if (!containerInput.isMain && fs.existsSync('/workspace/global')) {
    additionalDirectories.push('/workspace/global');
  }

  return {
    workingDirectory: '/workspace/group',
    skipGitRepoCheck: true,
    approvalPolicy: 'never',
    sandboxMode: 'danger-full-access',
    networkAccessEnabled: true,
    webSearchEnabled: true,
    additionalDirectories,
  };
}

function createCodex(containerInput: ContainerInput, mcpServerPath: string): Codex {
  const config: {
    instructions?: string;
    mcp_servers: Record<
      string,
      {
        command: string;
        args: string[];
        env: Record<string, string>;
      }
    >;
  } = {
    mcp_servers: {
      nanoclaw: {
        command: 'node',
        args: [mcpServerPath],
        env: {
          NANOCLAW_CHAT_JID: containerInput.chatJid,
          NANOCLAW_GROUP_FOLDER: containerInput.groupFolder,
          NANOCLAW_IS_MAIN: containerInput.isMain ? '1' : '0',
        },
      },
    },
  };

  const instructions = buildInstructions(containerInput);
  if (instructions) {
    config.instructions = instructions;
  }

  return new Codex({ config });
}

function createThread(
  codex: Codex,
  sessionId: string | undefined,
  threadOptions: ThreadOptions,
): Thread {
  return sessionId
    ? codex.resumeThread(sessionId, threadOptions)
    : codex.startThread(threadOptions);
}

async function runQuery(
  prompt: string,
  thread: Thread,
): Promise<{
  newSessionId?: string;
  closedDuringQuery: boolean;
  bufferedMessages: string[];
}> {
  let newSessionId = thread.id || undefined;
  let resultCount = 0;
  let closedDuringQuery = false;
  const bufferedMessages: string[] = [];

  const abortController = new AbortController();
  const pollTimer = setInterval(() => {
    if (shouldClose()) {
      closedDuringQuery = true;
      abortController.abort();
      return;
    }

    const pending = drainIpcInput();
    if (pending.length > 0) {
      bufferedMessages.push(...pending);
      log(`Buffered ${pending.length} IPC message(s) for next turn`);
    }
  }, IPC_POLL_MS);

  try {
    const { events } = await thread.runStreamed(prompt, {
      signal: abortController.signal,
    });

    for await (const event of events) {
      switch (event.type) {
        case 'thread.started':
          newSessionId = event.thread_id;
          log(`Thread initialized: ${newSessionId}`);
          break;
        case 'item.completed':
          if (event.item.type === 'agent_message') {
            resultCount++;
            const textResult = event.item.text?.trim() || '';
            log(`Result #${resultCount}: ${textResult.slice(0, 200)}`);
            appendConversationArchive(prompt, textResult);
            writeOutput({
              status: 'success',
              result: textResult || null,
              newSessionId,
            });
          }
          if (event.item.type === 'error') {
            throw new Error(event.item.message);
          }
          break;
        case 'turn.failed':
          throw new Error(event.error.message);
        case 'error':
          throw new Error(event.message);
        default:
          break;
      }
    }
  } catch (err) {
    if (!(closedDuringQuery && abortController.signal.aborted)) {
      throw err;
    }
    log('Turn aborted due to close sentinel');
  } finally {
    clearInterval(pollTimer);
  }

  log(
    `Query done. Results: ${resultCount}, closedDuringQuery: ${closedDuringQuery}`,
  );
  return { newSessionId, closedDuringQuery, bufferedMessages };
}

async function runScript(script: string): Promise<ScriptResult | null> {
  const scriptPath = '/tmp/task-script.sh';
  fs.writeFileSync(scriptPath, script, { mode: 0o755 });

  return new Promise((resolve) => {
    execFile(
      'bash',
      [scriptPath],
      {
        timeout: SCRIPT_TIMEOUT_MS,
        maxBuffer: 1024 * 1024,
        env: process.env,
      },
      (error, stdout, stderr) => {
        if (stderr) {
          log(`Script stderr: ${stderr.slice(0, 500)}`);
        }

        if (error) {
          log(`Script error: ${error.message}`);
          resolve(null);
          return;
        }

        const lines = stdout.trim().split('\n');
        const lastLine = lines[lines.length - 1];
        if (!lastLine) {
          log('Script produced no output');
          resolve(null);
          return;
        }

        try {
          const result = JSON.parse(lastLine);
          if (typeof result.wakeAgent !== 'boolean') {
            log(
              `Script output missing wakeAgent boolean: ${lastLine.slice(0, 200)}`,
            );
            resolve(null);
            return;
          }
          resolve(result as ScriptResult);
        } catch {
          log(`Script output is not valid JSON: ${lastLine.slice(0, 200)}`);
          resolve(null);
        }
      },
    );
  });
}

async function main(): Promise<void> {
  let containerInput: ContainerInput;

  try {
    const stdinData = await readStdin();
    containerInput = JSON.parse(stdinData) as ContainerInput;
    try {
      fs.unlinkSync('/tmp/input.json');
    } catch {
      // ignore
    }
    log(`Received input for group: ${containerInput.groupFolder}`);
  } catch (err) {
    writeOutput({
      status: 'error',
      result: null,
      error: `Failed to parse input: ${err instanceof Error ? err.message : String(err)}`,
    });
    process.exit(1);
  }

  const __dirname = path.dirname(fileURLToPath(import.meta.url));
  const mcpServerPath = path.join(__dirname, 'ipc-mcp-stdio.js');

  let sessionId = containerInput.sessionId;
  const threadOptions = buildThreadOptions(containerInput);
  const codex = createCodex(containerInput, mcpServerPath);
  let thread = createThread(codex, sessionId, threadOptions);

  fs.mkdirSync(IPC_INPUT_DIR, { recursive: true });
  try {
    fs.unlinkSync(IPC_INPUT_CLOSE_SENTINEL);
  } catch {
    // ignore
  }

  let prompt = containerInput.prompt;
  if (containerInput.isScheduledTask) {
    prompt =
      '[SCHEDULED TASK - The following message was sent automatically and is not coming directly from the user or group.]\n\n' +
      prompt;
  }

  const pending = drainIpcInput();
  if (pending.length > 0) {
    log(`Draining ${pending.length} pending IPC messages into initial prompt`);
    prompt += '\n' + pending.join('\n');
  }

  if (containerInput.script && containerInput.isScheduledTask) {
    log('Running task script...');
    const scriptResult = await runScript(containerInput.script);

    if (!scriptResult || !scriptResult.wakeAgent) {
      log(
        `Script decided not to wake agent: ${scriptResult ? 'wakeAgent=false' : 'script error/no output'}`,
      );
      writeOutput({
        status: 'success',
        result: null,
      });
      return;
    }

    prompt = `[SCHEDULED TASK]\n\nScript output:\n${JSON.stringify(
      scriptResult.data,
      null,
      2,
    )}\n\nInstructions:\n${containerInput.prompt}`;
  }

  try {
    while (true) {
      log(`Starting query (session: ${sessionId || 'new'})...`);

      let queryResult: {
        newSessionId?: string;
        closedDuringQuery: boolean;
        bufferedMessages: string[];
      };

      try {
        queryResult = await runQuery(prompt, thread);
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : String(err);

        // Resume can fail if the saved thread no longer exists. Recover once by
        // starting a fresh thread so old sessions do not brick the group.
        if (sessionId) {
          log(`Resume failed for session ${sessionId}: ${errorMessage}`);
          log('Starting a fresh session and retrying this prompt');
          sessionId = undefined;
          thread = createThread(codex, undefined, threadOptions);
          queryResult = await runQuery(prompt, thread);
        } else {
          throw err;
        }
      }

      if (queryResult.newSessionId) {
        sessionId = queryResult.newSessionId;
      }

      if (queryResult.closedDuringQuery) {
        log('Close sentinel consumed during query, exiting');
        break;
      }

      writeOutput({
        status: 'success',
        result: null,
        newSessionId: sessionId,
      });

      if (queryResult.bufferedMessages.length > 0) {
        prompt = queryResult.bufferedMessages.join('\n');
        log(
          `Continuing with ${queryResult.bufferedMessages.length} buffered message(s)`,
        );
        continue;
      }

      const immediate = drainIpcInput();
      if (immediate.length > 0) {
        prompt = immediate.join('\n');
        log(`Continuing with ${immediate.length} immediate message(s)`);
        continue;
      }

      log('Query ended, waiting for next IPC message...');
      const nextMessage = await waitForIpcMessage();
      if (nextMessage === null) {
        log('Close sentinel received, exiting');
        break;
      }

      log(`Got new message (${nextMessage.length} chars), starting new query`);
      prompt = nextMessage;
    }
  } catch (err) {
    const errorMessage = err instanceof Error ? err.message : String(err);
    log(`Agent error: ${errorMessage}`);
    writeOutput({
      status: 'error',
      result: null,
      newSessionId: sessionId,
      error: errorMessage,
    });
    process.exit(1);
  }
}

main();
