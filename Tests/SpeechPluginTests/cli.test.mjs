import { test } from 'node:test';
import assert from 'node:assert/strict';
import { chmod, mkdtemp, readFile, writeFile, rm, access, symlink, realpath } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { SpeechCLIRuntime } from '../../plugins/soniqo-speech/src/cli-runtime.mjs';
import { catalog, commands, validateArguments, buildArguments } from '../../plugins/soniqo-speech/src/cli-contract.mjs';
import { wav } from './helpers.mjs';

async function fixture(t, options = {}) {
  const dir = await mkdtemp(join(tmpdir(), 'speech-cli-test-'));
  const binary = join(dir, 'speech');
  const log = join(dir, 'argv.json');
  const descendantLog = join(dir, 'descendant-pid');
  const heartbeat = join(dir, 'heartbeat');
  const descendantSource = 'const fs = require("node:fs"); process.on("SIGTERM", () => {}); '
    + 'fs.writeFileSync(' + JSON.stringify(heartbeat) + ', "ready"); '
    + 'setInterval(() => fs.appendFileSync(' + JSON.stringify(heartbeat) + ', "."), 20);';
  const bytes = wav();
  const source = [
    '#!' + process.execPath,
    "import { writeFile, mkdir, truncate, rename } from 'node:fs/promises';",
    "import { dirname } from 'node:path';",
    "import { spawn } from 'node:child_process';",
    "import { access } from 'node:fs/promises';",
    'const argv=process.argv.slice(2);',
    'let stdin=""; for await(const chunk of process.stdin) stdin+=chunk;',
    'await writeFile(' + JSON.stringify(log + '.tmp') + ',JSON.stringify({argv,stdin,cwd:process.cwd(),pid:process.pid}));await rename(' + JSON.stringify(log + '.tmp') + ',' + JSON.stringify(log) + ');',
    ...(options.descendant ? [
      'const descendant = spawn(process.execPath, ["-e", ' + JSON.stringify(descendantSource) + '], { stdio: "ignore" });',
      'descendant.unref(); await writeFile(' + JSON.stringify(descendantLog) + ', String(descendant.pid));',
      'for(let i=0;i<200;i++){try{await access(' + JSON.stringify(heartbeat) + ');break;}catch{await new Promise(r=>setTimeout(r,10));}}',
    ] : []),
    'if (' + Boolean(options.pause) + ') await new Promise(r=>setTimeout(r,10000));',
    'if (' + Boolean(options.large) + ') process.stdout.write("x".repeat(100000));',
    'const output=argv.find(x=>x.startsWith("--output="));',
    'if(output){const path=output.slice("--output=".length);await mkdir(dirname(path),{recursive:true});await writeFile(path,Buffer.from(' + JSON.stringify(bytes.toString('base64')) + ',"base64"));if(' + Boolean(options.largeArtifact) + ')await truncate(path,26*1024*1024);}',
    'console.log("Result: known CLI output");',
    ...(options.failure ? ['console.error("Error: invalid selected input");process.exitCode=2;'] : []),
  ].join('\n');
  await writeFile(binary, source); await chmod(binary, 0o700);
  const runtime = new SpeechCLIRuntime({
    dataDir: join(dir, 'data'), lockDir: join(dir, 'locks'),
    env: { ...process.env, SPEECH_CLI_BINARY: binary },
    system: { platform: 'darwin', arch: 'arm64', release: '24.0.0' },
  });
  t.after(async () => {
    await runtime.close();
    if (options.descendant) {
      try { process.kill(Number(await readFile(descendantLog)), 'SIGKILL'); } catch {}
    }
    await rm(dir, { recursive: true, force: true });
  });
  return { dir, runtime, log, heartbeat };
}

test('catalogue exposes every CLI command independently and preserves every argument', () => {
  assert.equal(catalog.commands.length, 33);
  assert.equal(catalog.commands.reduce((total, spec) => total + spec.arguments.length, 0), 392);
  assert.equal(new Set(catalog.commands.map((spec) => spec.tool)).size, 33);
  for (const name of ['transcribe', 'speak', 'diarize', 'align', 'voice-chat', 'compose', 'restore', 'avatar-motion', 'gliner classify', 'gliner extract', 'clef decide', 'help']) {
    assert.ok(commands.has(name));
  }
  assert.ok(commands.get('speak').arguments.some((arg) => arg.key === 'voxcpm2_variant'));
  assert.ok(commands.get('speak').arguments.some((arg) => arg.key === 'voice_sample'));
});

test('argv construction prevents positional and option-value injection', () => {
  const args = { text: '--help; $(touch should-not-exist)', voice: '--verbose' };
  const spec = validateArguments('kokoro', args);
  const { argv } = buildArguments(spec, args, '/private/job');
  assert.ok(argv.includes('--voice=--verbose'));
  assert.deepEqual(argv.slice(-2), ['--', args.text]);
  assert.throws(() => validateArguments('arbitrary-shell', {}), /not in/);
  assert.throws(() => validateArguments('kokoro', { text: 'hi', shell: 'sh' }), /Unknown argument/);
});

test('nested CLI operations preserve the reviewed subcommand path and argument boundaries', () => {
  const args = { text: '--help; literal text', labels: 'reminder,other', variant: 'int8' };
  const { argv } = buildArguments(validateArguments('gliner classify', args), args, '/private/job');
  assert.deepEqual(argv.slice(0, 2), ['gliner', 'classify']);
  assert.ok(argv.includes('--labels=reminder,other'));
  assert.deepEqual(argv.slice(-2), ['--', args.text]);
  const clef = { request: 'Turn off the lights', offline: true };
  assert.deepEqual(buildArguments(validateArguments('clef decide', clef), clef, '/private/job').argv,
    ['clef', 'decide', '--offline', '--', clef.request]);
});

test('repeating options and native numeric flags retain their argument boundaries', () => {
  const args = { audio_file: '/selected.wav', keywords: ['hey soniqo', '--help'], beam: 4 };
  const { argv } = buildArguments(validateArguments('wake', args), args, '/private/job');
  assert.ok(argv.includes('--keywords=hey soniqo'));
  assert.ok(argv.includes('--keywords=--help'));
  assert.ok(argv.includes('--beam=4'));
  assert.deepEqual(argv.slice(-2), ['--', '/selected.wav']);
});

test('microphone and nested servers require explicit opt-ins', () => {
  assert.throws(() => validateArguments('voice-chat', {}), /Microphone/);
  assert.throws(() => validateArguments('voice-chat', { execution: { allow_microphone: true } }), /max_seconds/);
  assert.doesNotThrow(() => validateArguments('voice-chat', { max_seconds: 30, execution: { allow_microphone: true } }));
  assert.throws(() => validateArguments('voice-chat', { input: '/a.wav', mcp_config: '/selected.json' }), /configured MCP/);
  assert.doesNotThrow(() => validateArguments('voice-chat', {
    input: '/a.wav', mcp_config: '/selected.json', execution: { allow_external_tools: true },
  }));
});

test('executes fixed command without a shell and returns the real output and artifact', async (t) => {
  const { runtime, log } = await fixture(t);
  const result = await runtime.runCommand('kokoro', { text: 'literal $(touch never)', voice: 'af_bella' });
  assert.equal(result.exit_code, 0);
  assert.match(result.output, /known CLI output/);
  const record = JSON.parse(await readFile(log));
  assert.equal(record.argv[0], 'kokoro');
  assert.deepEqual(record.argv.slice(-2), ['--', 'literal $(touch never)']);
  assert.ok(record.argv.includes('--voice=af_bella'));
  const audio = result.artifacts.find((item) => item.mime_type === 'audio/wav');
  assert.ok(audio);
  assert.deepEqual((await runtime.readArtifact(audio.uri)).bytes, wav());
  const stdout = result.artifacts.find((item) => item.path.endsWith('stdout.txt'));
  assert.match((await runtime.readArtifact(stdout.uri)).bytes.toString(), /known CLI output/);
  await assert.rejects(() => access(runtime.lockPath));
});

test('supports selected working directories and stdin', async (t) => {
  const { runtime, log, dir } = await fixture(t);
  await runtime.runCommand('translate', { to: 'de', execution: { working_directory: dir, stdin_text: 'Hello.' } });
  const record = JSON.parse(await readFile(log));
  assert.equal(record.cwd, await realpath(dir));
  assert.equal(record.stdin, 'Hello.');
  assert.ok(record.argv.includes('--to=de'));
});

test('refuses existing output destinations unless overwrite was explicitly enabled', async (t) => {
  const { runtime, dir } = await fixture(t);
  const path = join(dir, 'existing.wav'); await writeFile(path, 'original');
  await assert.rejects(() => runtime.runCommand('kokoro', { text: 'hi', output: path }), /already exists/);
  assert.equal(await readFile(path, 'utf8'), 'original');
  const result = await runtime.runCommand('kokoro', { text: 'hi', output: path, execution: { overwrite: true } });
  assert.equal(result.exit_code, 0);
  assert.deepEqual(await readFile(path), wav());
});

test('never follows output symlinks even when overwriting was requested', async (t) => {
  const { runtime, dir } = await fixture(t);
  const target = join(dir, 'private.wav'); await writeFile(target, 'keep');
  const output = join(dir, 'linked.wav'); await symlink(target, output);
  await assert.rejects(() => runtime.runCommand('kokoro', {
    text: 'hi', output, execution: { overwrite: true },
  }), /symlink/);
  assert.equal(await readFile(target, 'utf8'), 'keep');
});

test('batch synthesis protects derived output names even when the base path does not exist', async (t) => {
  const { runtime, dir, log } = await fixture(t);
  const output = join(dir, 'batch.wav');
  const derived = join(dir, 'batch_0.wav');
  await writeFile(derived, 'original');
  await assert.rejects(() => runtime.runCommand('speak', {
    batch_file: '/selected-batch.txt', output,
  }), /derived batch output already exists/);
  assert.equal(await readFile(derived, 'utf8'), 'original');
  await assert.rejects(() => access(log));
  await runtime.runCommand('speak', {
    batch_file: '/selected-batch.txt', output, execution: { overwrite: true },
  });
  const target = join(dir, 'private.wav'); await writeFile(target, 'keep');
  await rm(output);
  await symlink(target, join(dir, 'batch_1.wav'));
  await assert.rejects(() => runtime.runCommand('speak', {
    batch_file: '/selected-batch.txt', output, execution: { overwrite: true },
  }), /derived batch output is a symlink/);
  assert.equal(await readFile(target, 'utf8'), 'keep');
});

test('nested MCP writes default to denied while preserving explicit policies', async (t) => {
  const { runtime, log } = await fixture(t);
  await runtime.runCommand('voice-chat', {
    input: '/selected.wav', mcp_config: '/selected-config.json',
    execution: { allow_external_tools: true },
  });
  assert.ok(JSON.parse(await readFile(log)).argv.includes('--mcp-write-policy=deny'));
  await runtime.runCommand('voice-chat', {
    input: '/selected.wav', mcp_config: '/selected-config.json', mcp_write_policy: 'allow',
    execution: { allow_external_tools: true },
  });
  assert.ok(JSON.parse(await readFile(log)).argv.includes('--mcp-write-policy=allow'));
});

test('preserves full CLI output locally when the returned preview is truncated', async (t) => {
  const { runtime } = await fixture(t, { large: true });
  const result = await runtime.runCommand('transcribe', { audio_file: '/selected.wav' });
  assert.equal(result.output_truncated, true);
  assert.match(result.output, /known CLI output/);
  const file = result.artifacts.find((item) => item.path.endsWith('stdout.txt'));
  assert.ok(file.bytes > 100000);
});

test('large generated artifacts provide local paths without unreadable resource links', async (t) => {
  const { runtime } = await fixture(t, { largeArtifact: true });
  const result = await runtime.runCommand('kokoro', { text: 'hi' });
  const audio = result.artifacts.find((item) => item.mime_type === 'audio/wav');
  assert.equal(audio.bytes, 26 * 1024 * 1024);
  assert.equal(audio.uri, undefined);
  await access(audio.path);
});

test('returns native failure status instead of fabricated success', async (t) => {
  const { runtime } = await fixture(t, { failure: true });
  const result = await runtime.runCommand('transcribe', { audio_file: '/selected.wav' });
  assert.equal(result.exit_code, 2);
  assert.match(result.error, /invalid selected input/);
});

test('cancels the CLI process and releases the shared inference lock', async (t) => {
  const { runtime, log } = await fixture(t, { pause: true });
  const controller = new AbortController();
  const pending = runtime.runCommand('transcribe', { audio_file: '/selected.wav' }, controller.signal);
  const rejected = assert.rejects(() => pending, /cancelled/);
  for (let i = 0; i < 100; i++) { try { await access(log); break; } catch { await delay(10); } }
  const { pid } = JSON.parse(await readFile(log));
  controller.abort(); await rejected;
  assert.throws(() => process.kill(pid, 0), { code: 'ESRCH' });
  await assert.rejects(() => access(runtime.lockPath));
});

test('cleans up descendants even after the CLI parent has already exited', async (t) => {
  const { runtime, heartbeat } = await fixture(t, { descendant: true });
  await runtime.runCommand('transcribe', { audio_file: '/selected.wav' });
  await delay(50);
  const stopped = await readFile(heartbeat, 'utf8');
  await delay(150);
  assert.equal(await readFile(heartbeat, 'utf8'), stopped, 'A descendant survived the completed CLI task');
  await assert.rejects(() => access(runtime.lockPath));
});

test('resource paths cannot traverse job roots or follow external symlinks', async (t) => {
  const { runtime, dir } = await fixture(t);
  const result = await runtime.runCommand('kokoro', { text: 'hi' });
  const audio = result.artifacts.find((item) => item.mime_type === 'audio/wav');
  const outside = join(dir, 'private.txt'); await writeFile(outside, 'private');
  const link = join(result.working_directory, 'external.wav');
  await symlink(outside, link);
  const uri = audio.uri.replace('output.wav', 'external.wav');
  await assert.rejects(() => runtime.readArtifact(uri), /outside/);
  await assert.rejects(() => runtime.readArtifact('speech://files/../../private'));
});
