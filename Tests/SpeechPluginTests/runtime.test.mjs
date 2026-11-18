import { test } from 'node:test';
import assert from 'node:assert/strict';
import { access, lstat, mkdir, mkdtemp, readFile, rm, symlink, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { readLimited } from '../../plugins/soniqo-speech/src/runtime.mjs';
import { SpeechCLIRuntime } from '../../plugins/soniqo-speech/src/cli-runtime.mjs';

async function fixture(t) {
  const dir = await mkdtemp(join(tmpdir(), 'speech-runtime-test-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const runtime = new SpeechCLIRuntime({
    dataDir: join(dir, 'data'), lockDir: join(dir, 'locks'),
    env: { SPEECH_CLI_BINARY: process.execPath },
    system: { platform: 'darwin', arch: 'arm64', release: '24.0.0' },
  });
  t.after(() => runtime.close());
  return { dir, runtime };
}

test('bounds streamed artifact reads even if the file grows while being read', async () => {
  await assert.rejects(() => readLimited(new Response(new Uint8Array(100)), 50), /size limit/);
});

test('status reports missing runtimes and unsupported platforms without creating data or loading models', async (t) => {
  const { runtime, dir } = await fixture(t);
  assert.equal((await runtime.status()).ready, true);
  await assert.rejects(() => access(runtime.dataDir));
  await assert.rejects(() => access(runtime.lockDir));
  runtime.system.platform = 'linux';
  assert.equal((await runtime.status()).ready, false);
  await assert.rejects(() => runtime.runCommand('kokoro', { text: 'Hello.' }), /Apple Silicon/);
  runtime.system.platform = 'darwin';
  runtime.env.SPEECH_CLI_BINARY = join(dir, 'missing');
  assert.equal((await runtime.status()).runtime_installed, false);
  await assert.rejects(() => runtime.runCommand('kokoro', { text: 'Hello.' }), /Install speech/);
});

test('different output directories share one exclusive inference lock', async (t) => {
  const { runtime } = await fixture(t);
  const other = new SpeechCLIRuntime({ dataDir: join(runtime.dataDir, 'other'),
    lockDir: runtime.lockDir, env: runtime.env, system: runtime.system });
  const owner = await runtime.acquireLock();
  try {
    assert.equal((await other.status()).inference_busy, true);
    await assert.rejects(() => other.runCommand('kokoro', { text: 'Hi.' }), /Another speech task/);
    assert.equal((await lstat(runtime.lockPath)).mode & 0o777, 0o700);
    assert.equal((await lstat(join(runtime.lockPath, 'owner.json'))).mode & 0o777, 0o600);
  } finally { await runtime.releaseLock(owner); }
  assert.equal((await other.status()).inference_busy, false);
});

test('a stale lock requires recovery and a foreign owner cannot release it', async (t) => {
  const { runtime } = await fixture(t);
  const owner = await runtime.acquireLock();
  await runtime.releaseLock({ token: 'different-owner' });
  assert.equal(JSON.parse(await readFile(join(runtime.lockPath, 'owner.json'))).token, owner.token);
  await assert.rejects(() => runtime.acquireLock(), /left a lock/);
  await runtime.releaseLock(owner);
});

test('a symlinked lock directory cannot redirect writes into another directory', async (t) => {
  const { runtime, dir } = await fixture(t);
  const target = join(dir, 'unrelated'); await mkdir(target);
  const marker = join(target, 'keep'); await writeFile(marker, 'unchanged');
  await symlink(target, runtime.lockDir);
  await assert.rejects(() => runtime.acquireLock(), /real directory/);
  assert.equal(await readFile(marker, 'utf8'), 'unchanged');
  await assert.rejects(() => access(join(target, 'inference.lock')));
});
