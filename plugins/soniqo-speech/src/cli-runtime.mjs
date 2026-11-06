import { spawn } from 'node:child_process';
import { constants, createWriteStream } from 'node:fs';
import { access, chmod, lstat, mkdir, open, readFile, readdir, realpath, writeFile } from 'node:fs/promises';
import { basename, dirname, isAbsolute, join, resolve, relative, sep, extname } from 'node:path';
import { randomUUID } from 'node:crypto';
import { Transform } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { SpeechRuntime, SpeechError, readLimited } from './runtime.mjs';
import { catalog, commands, validateArguments, buildArguments } from './cli-contract.mjs';

const PREVIEW_BYTES = 64 * 1024;
const OUTPUT_BYTES = 64 * 1024 * 1024;
const RESOURCE_BYTES = 25 * 1024 * 1024;
const ID = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
const MIME = { '.wav': 'audio/wav', '.mp3': 'audio/mpeg', '.json': 'application/json',
  '.jsonl': 'application/x-ndjson', '.txt': 'text/plain', '.rttm': 'text/plain',
  '.safetensors': 'application/octet-stream' };
const cleanText = (text) => text.replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, '');

async function privateDir(path) {
  await mkdir(path, { recursive: true, mode: 0o700 });
  const info = await lstat(path);
  if (!info.isDirectory() || info.isSymbolicLink()) throw new SpeechError('Select a real directory for local speech data.');
  await chmod(path, 0o700);
}

export class SpeechCLIRuntime extends SpeechRuntime {
  async findCliBinary() {
    const configured = this.env.SPEECH_CLI_BINARY;
    const candidates = configured ? [configured] : [
      ...(this.pluginRoot ? [join(this.pluginRoot, 'native', 'speech')] : []),
      '/opt/homebrew/bin/speech', '/usr/local/bin/speech',
      ...(this.env.PATH ?? '').split(':').filter(isAbsolute).map((path) => join(path, 'speech')),
    ];
    for (const candidate of candidates) {
      if (!isAbsolute(candidate)) continue;
      try { await access(candidate, constants.X_OK); return candidate; } catch {}
    }
    return undefined;
  }

  async status() {
    let locked = false;
    try { await lstat(this.lockPath); locked = true; } catch (error) { if (error.code !== 'ENOENT') throw error; }
    const binary = await this.findCliBinary();
    const cli = Boolean(binary);
    const bundled = Boolean(this.pluginRoot && binary === join(this.pluginRoot, 'native', 'speech'));
    return { ready: this.supported() && cli && !locked,
      platform_supported: this.supported(), runtime_installed: cli, inference_busy: locked,
      model_downloads: 'Required model files download from Hugging Face on first use and remain in the speech cache.',
      cli_runtime_installed: cli, cli_runtime_version: bundled ? catalog.runtime_version : 'external',
      cli_catalogue_version: catalog.runtime_version,
      cli_commands: [...commands.keys()],
      requirements: 'Apple Silicon, macOS 15+, Node.js 22+, and the bundled speech CLI or brew install speech.' };
  }

  async artifacts(jobDir, destinations) {
    const found = [];
    const visit = async (directory) => {
      if (found.length >= 256) return;
      for (const entry of await readdir(directory, { withFileTypes: true })) {
        const path = join(directory, entry.name);
        if (entry.isSymbolicLink()) continue;
        if (entry.isDirectory()) await visit(path);
        else if (entry.isFile()) found.push(path);
        if (found.length >= 256) break;
      }
    };
    await visit(jobDir);
    const listed = [];
    for (const path of found) {
      const info = await lstat(path);
      await chmod(path, 0o600);
      const components = relative(dirname(jobDir), path).split(sep).map(encodeURIComponent);
      listed.push({ path, ...(info.size <= RESOURCE_BYTES
        ? { uri: 'speech://files/' + components.join('/') } : {}),
        mime_type: MIME[extname(path).toLowerCase()] ?? 'application/octet-stream', bytes: info.size });
    }
    for (const destination of destinations.filter((item) => item.explicit)) {
      if (listed.some((item) => item.path === destination.path)) continue;
      try {
        const info = await lstat(destination.path);
        if (!info.isSymbolicLink()) listed.push({ path: destination.path,
          mime_type: info.isDirectory() ? 'inode/directory' : MIME[extname(destination.path)] ?? 'application/octet-stream',
          bytes: info.size });
      } catch {}
    }
    return listed;
  }

  async runCommand(command, args, signal) {
    if (command === 'voice-chat' && args.mcp_config && args.mcp_write_policy === undefined) {
      args = { ...args, mcp_write_policy: 'deny' };
    }
    const spec = validateArguments(command, args);
    if (!this.supported()) throw new SpeechError('Local speech requires Apple Silicon and macOS 15 or later.');
    const binary = await this.findCliBinary();
    if (!binary) throw new SpeechError('Install speech with Homebrew or set SPEECH_CLI_BINARY to its absolute path.');
    const owner = await this.acquireLock();
    let child;
    let exited;
    let outputPipeline;
    let abortListener;
    const failures = new AbortController();
    const control = args.execution ?? {};
    const activeSignal = AbortSignal.any([this.shutdown.signal, failures.signal,
      AbortSignal.timeout((control.timeout_seconds ?? 300) * 1000), ...(signal ? [signal] : [])]);
    const jobDir = join(this.dataDir, 'jobs', randomUUID());
    try {
      activeSignal.throwIfAborted();
      await privateDir(this.dataDir);
      await privateDir(jobDir);
      const cwd = control.working_directory ?? jobDir;
      if (!(await lstat(cwd)).isDirectory()) throw new SpeechError('The selected working directory does not exist.');
      const invocation = buildArguments(spec, args, jobDir, cwd);
      for (const destination of invocation.destinations.filter((item) => item.explicit)) {
          try {
            const existing = await lstat(destination.path);
            if (existing.isSymbolicLink()) throw new SpeechError('Select a real output path; symlink destinations are not overwritten.');
            if (!control.overwrite) throw new SpeechError('The output destination already exists. Choose a new path or explicitly enable execution.overwrite.');
          } catch (error) { if (error.code !== 'ENOENT') throw error; }
      }
      if (command === 'speak' && args.batch_file && args.output !== undefined) {
        const output = invocation.destinations.find((item) => !item.directory).path;
        const extension = extname(output) || '.wav';
        const stem = basename(output, extname(output)) + '_';
        let entries = [];
        try { entries = await readdir(dirname(output), { withFileTypes: true }); }
        catch (error) { if (error.code !== 'ENOENT') throw error; }
        for (const entry of entries) {
          if (!entry.name.startsWith(stem) || !entry.name.endsWith(extension)
              || !/^\d+$/.test(entry.name.slice(stem.length, -extension.length))) continue;
          if (entry.isSymbolicLink()) throw new SpeechError('A derived batch output is a symlink. Select a fresh output prefix.');
          if (!control.overwrite) throw new SpeechError('A derived batch output already exists. Choose a new prefix or explicitly enable execution.overwrite.');
        }
      }
      const stdoutPath = join(jobDir, 'stdout.txt');
      let preview = Buffer.alloc(0);
      let stderr = Buffer.alloc(0);
      let total = 0;
      const capture = new Transform({ transform(chunk, _encoding, callback) {
        total += chunk.length;
        if (total > OUTPUT_BYTES) return callback(new Error('CLI output size exceeded'));
        preview = Buffer.concat([preview, chunk]).subarray(-PREVIEW_BYTES);
        callback(null, chunk);
      } });
      child = spawn(binary, invocation.argv, {
        cwd, env: this.env, detached: true, stdio: ['pipe', 'pipe', 'pipe'],
      });
      exited = new Promise((ok) => {
        child.once('close', (code, signalName) => ok({ code, signal: signalName }));
        child.once('error', () => ok({ code: null, error: true }));
      });
      this.active = { child, exited };
      owner.child_pid = child.pid ?? null;
      await writeFile(join(this.lockPath, 'owner.json'), JSON.stringify(owner), { mode: 0o600 });
      child.stderr.on('data', (chunk) => { stderr = Buffer.concat([stderr, chunk]).subarray(-16384); });
      child.stdin.on('error', () => {});
      child.stdin.end(control.stdin_text ?? '');
      outputPipeline = pipeline(child.stdout, capture, createWriteStream(stdoutPath, { flags: 'wx', mode: 0o600 }));
      outputPipeline.catch(() => failures.abort());
      const aborted = new Promise((_ok, fail) => {
        abortListener = () => fail(new SpeechError('The speech CLI task was cancelled, timed out, or exceeded its output limit.'));
        activeSignal.addEventListener('abort', abortListener, { once: true });
        if (activeSignal.aborted) abortListener();
      });
      const exit = await Promise.race([exited, aborted]);
      await outputPipeline;
      if (exit.error) throw new SpeechError('The speech CLI could not start.');
      const output = cleanText(preview.toString('utf8'));
      const errors = cleanText(stderr.toString('utf8')).split('\n').filter((line) => /error:|Error:/.test(line)).slice(-5).join('\n');
      return { command, exit_code: exit.code ?? -1, output,
        output_truncated: total > PREVIEW_BYTES, working_directory: cwd,
        ...(exit.code !== 0 ? { error: errors || 'The CLI exited without completing the command.' } : {}),
        artifacts: await this.artifacts(jobDir, invocation.destinations) };
    } catch (error) {
      if (error instanceof SpeechError) throw error;
      if (activeSignal.aborted) throw new SpeechError('The speech CLI task was cancelled, timed out, or exceeded its output limit.');
      throw new SpeechError('The speech CLI task failed. Check the selected files, runtime, and available disk space.');
    } finally {
      if (abortListener) activeSignal.removeEventListener('abort', abortListener);
      try {
        if (child) await this.stopChild(child, exited);
        if (outputPipeline) await outputPipeline.catch(() => {});
      } finally { this.active = undefined; await this.releaseLock(owner); }
    }
  }

  async readArtifact(uri) {
    const parsed = new URL(uri);
    if (parsed.protocol !== 'speech:' || parsed.hostname !== 'files' || parsed.search || parsed.hash) throw new SpeechError('Unknown speech artifact.');
    const parts = parsed.pathname.slice(1).split('/').map(decodeURIComponent);
    if (!ID.test(parts[0]) || parts.length < 2 || parts.some((part) => !part || ['.', '..'].includes(part)
        || /[\\/\0]/.test(part))) throw new SpeechError('Invalid speech artifact path.');
    const jobPath = join(this.dataDir, 'jobs', parts[0]);
    if ((await lstat(jobPath)).isSymbolicLink()) throw new SpeechError('The job directory is a symlink.');
    const root = await realpath(jobPath);
    const path = join(root, ...parts.slice(1));
    const actual = await realpath(path);
    if (!actual.startsWith(root + sep)) throw new SpeechError('The artifact is outside its job directory.');
    const file = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK);
    try {
      const info = await file.stat();
      if (!info.isFile() || info.size > RESOURCE_BYTES) throw new SpeechError('This artifact is too large for inline transfer; use its local path.');
      const bytes = await readLimited({ body: file.createReadStream({
        start: 0, end: RESOURCE_BYTES, autoClose: false,
      }) }, RESOURCE_BYTES);
      return { bytes, mime_type: MIME[extname(path)] ?? 'application/octet-stream' };
    } finally { await file.close(); }
  }
}
