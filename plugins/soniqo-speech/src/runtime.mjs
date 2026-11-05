import { chmod, lstat, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { arch, homedir, platform, release } from 'node:os';
import { join, resolve } from 'node:path';
import { randomUUID } from 'node:crypto';

export class SpeechError extends Error {
  constructor(message) { super(message); this.name = 'SpeechError'; }
}

async function privateDirectory(path) {
  await mkdir(path, { recursive: true, mode: 0o700 });
  const info = await lstat(path);
  if (!info.isDirectory() || info.isSymbolicLink()) throw new SpeechError('The speech data directory must be a real directory.');
  await chmod(path, 0o700);
}

export async function readLimited(response, limit) {
  if (!response.body) throw new SpeechError('The speech runtime returned an empty response.');
  const parts = [];
  let length = 0;
  for await (const chunk of response.body) {
    length += chunk.length;
    if (length > limit) throw new SpeechError('The speech runtime response exceeded the size limit.');
    parts.push(Buffer.from(chunk));
  }
  return Buffer.concat(parts);
}

export class SpeechRuntime {
  constructor(options = {}) {
    this.env = options.env ?? process.env;
    this.pluginRoot = options.pluginRoot;
    this.dataDir = resolve(options.dataDir ?? this.env.SPEECH_PLUGIN_DATA_DIR
      ?? join(homedir(), 'Library', 'Application Support', 'Soniqo Speech'));
    this.system = options.system ?? { platform: platform(), arch: arch(), release: release() };
    this.lockDir = resolve(options.lockDir ?? join(homedir(), 'Library', 'Caches', 'Soniqo Speech'));
    this.lockPath = join(this.lockDir, 'inference.lock');
    this.shutdown = new AbortController();
    this.active = undefined;
  }

  supported() {
    return this.system.platform === 'darwin' && this.system.arch === 'arm64'
      && Number(this.system.release.split('.')[0]) >= 24;
  }

  async acquireLock() {
    await privateDirectory(this.lockDir);
    try { await mkdir(this.lockPath, { mode: 0o700 }); } catch (error) {
      if (error.code === 'EEXIST') throw new SpeechError('Another speech task is running or left a lock. Wait for it to finish; see the desktop plugin guide for crash recovery.');
      throw error;
    }
    const owner = { pid: process.pid, child_pid: null, token: randomUUID() };
    try {
      await writeFile(join(this.lockPath, 'owner.json'), JSON.stringify(owner), { mode: 0o600, flag: 'wx' });
    } catch (error) { await rm(this.lockPath, { recursive: true, force: true }); throw error; }
    return owner;
  }

  async releaseLock(owner) {
    const saved = JSON.parse(await readFile(join(this.lockPath, 'owner.json'), 'utf8'));
    if (saved.token === owner.token) await rm(this.lockPath, { recursive: true });
  }

  async stopChild(child, exited) {
    if (!child.pid) { await exited; return; }
    const terminate = (signal) => {
      try { process.kill(-child.pid, signal); } catch (error) { if (error.code !== 'ESRCH') throw error; }
    };
    // Descendants can survive the CLI parent, including configured MCP servers.
    terminate('SIGTERM');
    const timer = setTimeout(() => { terminate('SIGKILL'); }, 3000);
    timer.unref();
    try { await exited; }
    finally { clearTimeout(timer); terminate('SIGKILL'); }
  }

  async close() {
    this.shutdown.abort();
    const active = this.active;
    if (active) await this.stopChild(active.child, active.exited);
  }
}
