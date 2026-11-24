import { McpServer, ResourceTemplate } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { McpError, ErrorCode } from '@modelcontextprotocol/sdk/types.js';
import { z } from 'zod';
import { dirname, resolve } from 'node:path';
import { SpeechError } from './runtime.mjs';
import { SpeechCLIRuntime } from './cli-runtime.mjs';
import { catalog, commands } from './cli-contract.mjs';

const version = typeof SPEECH_PLUGIN_VERSION === 'string' ? SPEECH_PLUGIN_VERSION : '0.1.0';
const runtime = new SpeechCLIRuntime({ pluginRoot: resolve(dirname(process.argv[1]), '..') });
const server = new McpServer({ name: 'soniqo-speech', version }, {
  instructions: 'Use selected inputs. Require consent for playback, recording, overwrites and nested servers.',
});

async function runTool(extra, operation) {
  let progress = 0;
  const token = extra?._meta?.progressToken;
  const timer = token === undefined ? undefined : setInterval(() => {
    extra.sendNotification({
      method: 'notifications/progress',
      params: { progressToken: token, progress: ++progress, message: 'Local speech processing is running.' },
    }).catch(() => {});
  }, 5000);
  timer?.unref();
  try {
    return await operation();
  } catch (error) {
    return { isError: true, content: [{ type: 'text',
      text: error instanceof SpeechError ? error.message : 'Local speech processing failed. Check the runtime and available disk space.' }] };
  } finally { if (timer) clearInterval(timer); }
}

server.registerTool('get_speech_status', {
  title: 'Check local speech setup',
  description: 'Check Apple Silicon and CLI runtime availability and list supported commands without loading or downloading models.',
  inputSchema: z.object({}).strict(),
  outputSchema: z.object({
    ready: z.boolean(), platform_supported: z.boolean(), runtime_installed: z.boolean(),
    inference_busy: z.boolean(), requirements: z.string(),
    model_downloads: z.string(),
    cli_runtime_installed: z.boolean(), cli_runtime_version: z.string(),
    cli_catalogue_version: z.string(), cli_commands: z.array(z.string()),
  }),
  annotations: { readOnlyHint: true, destructiveHint: false, openWorldHint: false },
}, async (_args, extra) => runTool(extra, async () => {
  const status = await runtime.status();
  return { structuredContent: status, content: [{ type: 'text', text: JSON.stringify(status) }] };
}));

const executionSchema = z.object({
  timeout_seconds: z.number().min(1).max(1800).optional(),
  working_directory: z.string().optional(),
  stdin_text: z.string().max(1024 * 1024).optional(),
  overwrite: z.boolean().optional().describe('Explicit consent to replace existing outputs.'),
  allow_microphone: z.boolean().optional().describe('Explicit recording consent; max_seconds required.'),
  allow_external_tools: z.boolean().optional().describe('Explicit consent to start selected MCP servers.'),
}).strict();

const artifactSchema = z.object({
  path: z.string(), uri: z.string().optional(), mime_type: z.string(), bytes: z.number(),
});
const cliOutputSchema = z.object({
  command: z.string(), exit_code: z.number(), output: z.string(), output_truncated: z.boolean(),
  working_directory: z.string(), error: z.string().optional(), artifacts: z.array(artifactSchema),
});

function brief(text, limit = 56) {
  const value = text.replace(/\s+/g, ' ').trim();
  return value.length <= limit ? value : value.slice(0, limit - 1).replace(/\s+\S*$/, '') + '…';
}

const prominentArguments = new Set(['text', 'engine', 'input', 'audio_file', 'output', 'output_dir', 'model', 'model_id',
  'voice_sample', 'voice_sample_text', 'ref_audio', 'ref_text', 'play', 'mcp_config', 'mcp_write_policy']);

for (const spec of catalog.commands) {
  const cliShape = {};
  for (const arg of spec.arguments) {
    let field = arg.type === 'boolean' ? z.boolean()
      : arg.choices ? z.enum(arg.choices)
      : arg.type === 'number' ? z.number().finite()
      : z.string().refine((value) => value.length <= 1024 * 1024, 'CLI values are limited to 1 MiB.');
    if (arg.repeating) field = z.array(field).max(1000);
    // Keep every typed argument visible within desktop discovery budgets; full help stays in the catalogue.
    if (arg.description && (arg.required || arg.kind === 'positional' || prominentArguments.has(arg.key))) {
      field = field.describe(arg.key === 'mcp_write_policy'
        ? 'Default deny; allow requires explicit external-write consent.' : brief(arg.description));
    }
    cliShape[arg.key] = arg.required ? field : field.optional();
  }
  const help = spec.command === 'help' || Boolean(spec.subcommands?.length);
  const shape = {};
  if (Object.keys(cliShape).length) {
    const cli = z.object(cliShape).strict();
    shape.cli = spec.arguments.some((arg) => arg.required) ? cli : cli.optional();
  }
  if (!help) shape.execution = executionSchema.optional();
  server.registerTool(spec.tool, {
    title: 'speech ' + spec.command,
    description: brief(spec.description, 160) + (help ? ' Reads bundled help without inference.'
      : ' Local CLI; flags in cli. Full flag help: speech_help.'),
    inputSchema: z.object(shape).strict(), outputSchema: cliOutputSchema,
    annotations: {
      readOnlyHint: help,
      destructiveHint: !help && spec.arguments.some((arg) =>
        ['output', 'output_dir', 'debug_dir', 'mcp_config'].includes(arg.key)),
      openWorldHint: !help,
    },
  }, async (args, extra) => runTool(extra, async () => {
    const { cli, execution } = args;
    args = { ...cli, ...(execution === undefined ? {} : { execution }) };
    let result;
    if (help) {
      const names = spec.command === 'help' ? args.subcommands ?? [] : spec.path;
      const selected = names.length ? commands.get(names.join(' ')) : undefined;
      if (names.length && !selected) throw new SpeechError('Select one command path from the speech catalogue.');
      result = { command: spec.command, exit_code: 0, output: JSON.stringify(selected ?? {
        runtime_version: catalog.runtime_version,
        commands: catalog.commands.map((command) => ({ command: command.command, description: command.description })),
      }, null, 2), output_truncated: false, working_directory: '', artifacts: [] };
    } else {
      result = await runtime.runCommand(spec.command, args, extra.signal);
    }
    const content = [{ type: 'text', text: result.output || result.error || 'The speech command completed.' }];
    for (const artifact of result.artifacts) {
      if (artifact.uri && artifact.bytes) content.push({
        type: 'resource_link', uri: artifact.uri, name: artifact.path.split('/').pop(),
        mimeType: artifact.mime_type, description: 'CLI output saved at ' + artifact.path,
      });
    }
    const inline = result.artifacts.find((item) => item.uri && item.mime_type === 'audio/wav'
      && item.bytes > 0 && item.bytes <= 2 * 1024 * 1024);
    if (inline) {
      const audio = await runtime.readArtifact(inline.uri);
      content.push({ type: 'audio', data: audio.bytes.toString('base64'), mimeType: 'audio/wav' });
    }
    return { ...(result.exit_code !== 0 ? { isError: true } : {}), structuredContent: result, content };
  }));
}

server.registerResource('cli-output', new ResourceTemplate('speech://files/{job}/{+path}', { list: undefined }), {
  title: 'Speech CLI artifact', description: 'Read an output created in a managed speech job directory.',
}, async (uri) => {
  try {
    const result = await runtime.readArtifact(uri.href);
    return { contents: [{ uri: uri.href, mimeType: result.mime_type, blob: result.bytes.toString('base64') }] };
  } catch { throw new McpError(ErrorCode.InvalidParams, 'The speech artifact is unavailable or exceeds the inline transfer limit.'); }
});

async function shutdown() {
  await runtime.close();
  await server.close();
}

process.once('SIGINT', () => { shutdown().catch(() => { process.exitCode = 1; }); });
process.once('SIGTERM', () => { shutdown().catch(() => { process.exitCode = 1; }); });
server.server.onclose = () => { runtime.close().catch(() => { process.exitCode = 1; }); };

server.connect(new StdioServerTransport()).catch(() => {
  console.error('Could not start the local speech tools.');
  process.exitCode = 1;
});
