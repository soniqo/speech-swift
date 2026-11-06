import { test } from 'node:test';
import assert from 'node:assert/strict';
import { access, lstat } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';
import { homedir, platform, arch } from 'node:os';
import { inspectWav } from './helpers.mjs';
import { SpeechCLIRuntime } from '../../plugins/soniqo-speech/src/cli-runtime.mjs';
import { connectPlugin } from './client.mjs';

const enabled = process.env.SPEECH_PLUGIN_E2E === '1' && platform() === 'darwin' && arch() === 'arm64';
const entry = process.env.SPEECH_PLUGIN_E2E_ENTRY
  ?? fileURLToPath(new URL('../../plugins/soniqo-speech/dist/server.cjs', import.meta.url));

test('E2ESpeechPluginCLI: mirrored Kokoro and Parakeet commands run through the packaged CLI', {
  skip: !enabled, timeout: 360000,
}, async (t) => {
  const dataDir = process.env.SPEECH_PLUGIN_E2E_DATA_DIR
    ?? join(homedir(), 'Library', 'Application Support', 'Soniqo Speech');
  const runtime = new SpeechCLIRuntime({ dataDir, pluginRoot: resolve(dirname(entry), '..') });
  assert.equal((await runtime.status()).ready, true);
  if (process.env.SPEECH_PLUGIN_E2E_ENTRY && !process.env.SPEECH_CLI_BINARY) {
    assert.equal(await runtime.findCliBinary(), resolve(dirname(entry), '../native/speech'));
  }
  const root = resolve(dirname(entry), '..');
  const client = await connectPlugin(root, { SPEECH_PLUGIN_DATA_DIR: dataDir });
  t.after(() => client.close());
  const generated = await client.callTool({
    name: 'speech_kokoro', arguments: { cli: { text: 'The quick brown fox jumps over the lazy dog.', voice: 'af_heart', language: 'en' } },
  }, undefined, { timeout: 300000, resetTimeoutOnProgress: true });
  assert.notEqual(generated.isError, true, JSON.stringify(generated.content));
  assert.equal(generated.structuredContent.exit_code, 0);
  const audio = generated.structuredContent.artifacts.find((item) => item.mime_type === 'audio/wav');
  assert.ok(audio);
  assert.equal((await lstat(audio.path)).mode & 0o777, 0o600);
  const resource = await client.readResource({ uri: audio.uri });
  const metadata = inspectWav(Buffer.from(resource.contents[0].blob, 'base64'));
  assert.equal(metadata.sample_rate, 24000);
  assert.ok(metadata.duration_seconds >= 1 && metadata.duration_seconds <= 15);
  assert.equal(resource.contents[0].blob, generated.content.find((part) => part.type === 'audio').data);
  const transcript = await client.callTool({
    name: 'speech_transcribe', arguments: { cli: { audio_file: audio.path, engine: 'parakeet' } },
  }, undefined, { timeout: 300000, resetTimeoutOnProgress: true });
  assert.notEqual(transcript.isError, true, JSON.stringify(transcript.content));
  assert.equal(transcript.structuredContent.exit_code, 0);
  assert.match(transcript.structuredContent.output.toLowerCase(), /quick brown fox/);
  assert.match(transcript.structuredContent.output.toLowerCase(), /lazy dog/);
  await assert.rejects(() => access(runtime.lockPath));
  t.diagnostic(JSON.stringify({ cli_commands: ['kokoro', 'transcribe'], audio_path: audio.path,
    transcript_matched: true, artifacts: generated.structuredContent.artifacts.length }));
});
