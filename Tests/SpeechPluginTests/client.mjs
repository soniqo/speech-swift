import { Client } from '../../plugins/soniqo-speech/node_modules/@modelcontextprotocol/sdk/dist/esm/client/index.js';
import { StdioClientTransport } from '../../plugins/soniqo-speech/node_modules/@modelcontextprotocol/sdk/dist/esm/client/stdio.js';
import { readFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';

export async function connectPlugin(root, env = {}) {
  const declaration = JSON.parse(await readFile(join(root, 'mcp.json'))).mcpServers.speech;
  const transport = new StdioClientTransport({
    command: resolve(root, declaration.command), args: declaration.args,
    cwd: root, stderr: 'pipe', env: {
      ...process.env, SPEECH_PLUGIN_NODE: process.execPath, ...env,
    },
  });
  let diagnostics = '';
  transport.stderr?.on('data', (chunk) => { diagnostics += chunk; });
  const client = new Client({ name: 'speech-plugin-tests', version: '1.0.0' });
  try { await client.connect(transport); }
  catch (error) { await client.close(); throw error; }
  client.stderr = () => diagnostics;
  return client;
}
