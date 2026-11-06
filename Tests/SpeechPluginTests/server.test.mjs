import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { catalog } from '../../plugins/soniqo-speech/src/cli-contract.mjs';
import { connectPlugin } from './client.mjs';

const entry = fileURLToPath(new URL('../../plugins/soniqo-speech/dist/server.cjs', import.meta.url));

export async function connect(t, env = {}) {
  const dir = await mkdtemp(join(tmpdir(), 'speech-plugin-protocol-'));
  const root = resolve(dirname(entry), '..');
  const client = await connectPlugin(root, { SPEECH_PLUGIN_DATA_DIR: dir, ...env });
  t.after(async () => { await client.close(); await rm(dir, { recursive: true, force: true }); });
  return { client, dir, stderr: client.stderr };
}

test('bundled server negotiates MCP and exposes focused tools with explicit annotations', async (t) => {
  const { client, stderr } = await connect(t);
  assert.equal(client.getServerVersion().name, 'soniqo-speech');
  const { tools } = await client.listTools();
  assert.deepEqual(tools.map((tool) => tool.name).sort(),
    ['get_speech_status', ...catalog.commands.map((spec) => spec.tool)].sort());
  for (const tool of tools) {
    assert.equal(typeof tool.annotations.readOnlyHint, 'boolean');
    assert.equal(typeof tool.annotations.openWorldHint, 'boolean');
    assert.equal(typeof tool.annotations.destructiveHint, 'boolean');
    assert.equal(tool.inputSchema.additionalProperties, false);
    assert.ok(tool.outputSchema);
  }
  assert.equal(tools.find((tool) => tool.name === 'speech_transcribe').annotations.destructiveHint, false);
  assert.equal(tools.find((tool) => tool.name === 'speech_diarize').annotations.destructiveHint, false);
  assert.equal(tools.find((tool) => tool.name === 'speech_kokoro').annotations.destructiveHint, true);
  assert.equal(tools.find((tool) => tool.name === 'speech_voice_chat').annotations.destructiveHint, true);
  const transcription = tools.find((tool) => tool.name === 'speech_transcribe').inputSchema;
  assert.equal(transcription.properties.cli.properties.audio_file.type, 'string');
  assert.equal(transcription.properties.cli.properties.window.type, 'number');
  assert.equal(transcription.properties.cli.properties.audio_file.anyOf, undefined);
  assert.equal(transcription.properties.context, undefined);
  assert.equal(transcription.properties.cli.properties.context.type, 'string');
  const result = await client.callTool({ name: 'get_speech_status', arguments: {} });
  assert.equal(result.isError, undefined);
  assert.equal(typeof result.structuredContent.ready, 'boolean');
  assert.equal(result.structuredContent.cli_commands.length, 33);
  assert.equal(stderr(), '');
});

test('desktop discovery budgets retain every typed CLI operation', async (t) => {
  const { client } = await connect(t);
  const { tools } = await client.listTools();
  const namespaceDescription = 'This tool is part of plugin `Soniqo Speech`.\n\n' + client.getInstructions();
  let total = 0;
  for (const tool of tools) {
    const bytes = Buffer.byteLength(JSON.stringify({
      type: 'namespace', name: 'mcp__speech', description: namespaceDescription,
      tools: [{ type: 'function', name: tool.name, description: tool.description,
        parameters: tool.inputSchema, strict: false }],
    }));
    assert.ok(bytes <= 7500, tool.name + ' must leave headroom below the 8,000-byte per-tool limit: ' + bytes);
    total += bytes;
    const spec = catalog.commands.find((item) => item.tool === tool.name);
    if (!spec?.arguments.length) continue;
    assert.deepEqual(Object.keys(tool.inputSchema.properties.cli.properties).sort(),
      spec.arguments.map((arg) => arg.key).sort(), tool.name + ' must retain every declared CLI argument');
    assert.deepEqual((tool.inputSchema.properties.cli.required ?? []).sort(),
      spec.arguments.filter((arg) => arg.required).map((arg) => arg.key).sort());
  }
  assert.ok(total <= 58000, 'Combined definitions must leave headroom below 64,000 bytes: ' + total);
  t.diagnostic('Desktop tool definitions: ' + total + ' bytes across ' + tools.length + ' tools.');
});

test('MCP schema validation rejects unknown fields, invalid limits, and missing required arguments', async (t) => {
  const { client } = await connect(t);
  for (const args of [
    { cli: { text: 'Hello.' }, execution: { timeout_seconds: 0 } },
    { cli: { text: 'Hello.', list_voices: 'yes' } },
    { cli: { text: 'Hello.', executable: '/bin/sh' } },
    { cli: { text: 'x'.repeat(1024 * 1024 + 1) } },
  ]) {
    const result = await client.callTool({ name: 'speech_kokoro', arguments: args });
    assert.equal(result.isError, true);
  }
  const noFile = await client.callTool({ name: 'speech_transcribe', arguments: {} });
  assert.equal(noFile.isError, true);
  const numericPath = await client.callTool({ name: 'speech_transcribe', arguments: { cli: { audio_file: 123 } } });
  assert.equal(numericPath.isError, true);
});

test('MCP failures return errors without fabricated transcripts or audio', async (t) => {
  const { client } = await connect(t, { SPEECH_CLI_BINARY: '/missing/speech' });
  const status = await client.callTool({ name: 'get_speech_status', arguments: {} });
  assert.equal(status.structuredContent.runtime_installed, false);
  const result = await client.callTool({ name: 'speech_kokoro', arguments: { cli: { text: 'Hello.' } } });
  assert.equal(result.isError, true);
  assert.equal(result.structuredContent, undefined);
  assert.ok(result.content.every((item) => item.type === 'text'));
  await assert.rejects(() => client.readResource({ uri: 'speech://files/../../private' }));
});

test('MCP exposes nested operations independently and answers group help without a runtime', async (t) => {
  const { client } = await connect(t, { SPEECH_CLI_BINARY: '/missing/speech' });
  const { tools } = await client.listTools();
  for (const name of ['speech_gliner_classify', 'speech_gliner_extract', 'speech_clef_decide']) {
    assert.ok(tools.some((tool) => tool.name === name));
  }
  for (const name of ['speech_gliner', 'speech_clef', 'speech_help']) {
    assert.equal(tools.find((tool) => tool.name === name).annotations.readOnlyHint, true);
  }
  const group = await client.callTool({ name: 'speech_gliner', arguments: {} });
  assert.notEqual(group.isError, true);
  assert.deepEqual(JSON.parse(group.structuredContent.output).subcommands, ['gliner classify', 'gliner extract']);
  const help = await client.callTool({ name: 'speech_help', arguments: { cli: { subcommands: ['gliner', 'classify'] } } });
  assert.notEqual(help.isError, true);
  assert.equal(JSON.parse(help.structuredContent.output).command, 'gliner classify');
  const speak = await client.callTool({ name: 'speech_help', arguments: { cli: { subcommands: ['speak'] } } });
  assert.notEqual(speak.isError, true);
  assert.deepEqual(JSON.parse(speak.structuredContent.output), catalog.commands.find((spec) => spec.command === 'speak'));
});
