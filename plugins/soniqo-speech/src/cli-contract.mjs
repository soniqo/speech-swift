import catalog from './cli-catalog.json' with { type: 'json' };
import { isAbsolute, resolve, basename } from 'node:path';
import { SpeechError } from './runtime.mjs';

export { catalog };
export const commands = new Map(catalog.commands.map((spec) => [spec.command, spec]));

function valueString(value) {
  if (!['string', 'number'].includes(typeof value) || (typeof value === 'number' && !Number.isFinite(value))) {
    throw new SpeechError('CLI values must be strings or finite numbers.');
  }
  const text = String(value);
  if (text.includes('\0') || text.length > 1024 * 1024) throw new SpeechError('The CLI value is invalid or too long.');
  return text;
}

export function validateArguments(command, args) {
  const spec = commands.get(command);
  if (!spec) throw new SpeechError('This speech command is not in the reviewed command catalogue.');
  const known = new Set(spec.arguments.map((arg) => arg.key));
  for (const key of Object.keys(args)) {
    if (key !== 'execution' && !known.has(key)) throw new SpeechError('Unknown argument: ' + key);
  }
  for (const arg of spec.arguments) {
    const value = args[arg.key];
    if (value === undefined) {
      if (arg.required) throw new SpeechError('Missing required argument: ' + arg.key);
      continue;
    }
    if (arg.type === 'boolean') {
      if (typeof value !== 'boolean') throw new SpeechError(arg.key + ' must be a boolean.');
    } else if (arg.repeating) {
      if (!Array.isArray(value) || value.length > 1000) throw new SpeechError(arg.key + ' must be an array.');
      for (const item of value) valueString(item);
    } else {
      valueString(value);
      if (arg.choices && !arg.choices.includes(value)) throw new SpeechError('Unsupported value for ' + arg.key);
    }
  }
  const execution = args.execution ?? {};
  const allowed = new Set(['timeout_seconds', 'working_directory', 'stdin_text', 'allow_microphone', 'allow_external_tools', 'overwrite']);
  for (const key of Object.keys(execution)) if (!allowed.has(key)) throw new SpeechError('Unknown execution option: ' + key);
  if (execution.timeout_seconds !== undefined && (!Number.isFinite(execution.timeout_seconds)
      || execution.timeout_seconds < 1 || execution.timeout_seconds > 1800)) {
    throw new SpeechError('The execution timeout must be between 1 and 1,800 seconds.');
  }
  if (execution.working_directory !== undefined && (typeof execution.working_directory !== 'string'
      || !isAbsolute(execution.working_directory) || execution.working_directory.includes('\0'))) {
    throw new SpeechError('The working directory must be an absolute local path.');
  }
  for (const key of ['allow_microphone', 'allow_external_tools', 'overwrite']) {
    if (execution[key] !== undefined && typeof execution[key] !== 'boolean') throw new SpeechError(key + ' must be a boolean.');
  }
  if (execution.stdin_text !== undefined) {
    if (typeof execution.stdin_text !== 'string') throw new SpeechError('stdin_text must be a string.');
    valueString(execution.stdin_text);
  }
  if (command === 'voice-chat' && !args.input) {
    if (execution.allow_microphone !== true) throw new SpeechError('Microphone capture requires explicit user intent and execution.allow_microphone=true; otherwise supply input.');
    if (!Number.isFinite(Number(args.max_seconds)) || Number(args.max_seconds) <= 0 || Number(args.max_seconds) > 300) {
      throw new SpeechError('Live voice chat requires max_seconds between 1 and 300.');
    }
  }
  if (command === 'voice-chat' && args.mcp_config && execution.allow_external_tools !== true) {
    throw new SpeechError('Starting configured MCP servers requires explicit user intent and execution.allow_external_tools=true.');
  }
  return spec;
}

export function buildArguments(spec, args, jobDir, workingDirectory = jobDir) {
  const values = { ...args };
  const destinations = [];
  for (const arg of spec.arguments) {
    if (!['output', 'output_dir', 'debug_dir'].includes(arg.key)) continue;
    const fallback = arg.key === 'output' ? basename(arg.default ?? (spec.command + '.wav'))
      : arg.key === 'debug_dir' ? 'debug' : 'output';
    const path = args[arg.key] === undefined ? resolve(jobDir, fallback) : resolve(workingDirectory, valueString(args[arg.key]));
    values[arg.key] = path;
    destinations.push({ path, directory: arg.key !== 'output', explicit: args[arg.key] !== undefined });
  }
  const argv = [...spec.path];
  const positionals = [];
  for (const arg of spec.arguments) {
    const value = values[arg.key];
    if (value === undefined) continue;
    if (arg.kind === 'positional') {
      positionals.push(...(arg.repeating ? value : [value]).map(valueString));
    } else if (arg.type === 'boolean') {
      if (value) argv.push(arg.flag);
    } else {
      for (const item of arg.repeating ? value : [value]) argv.push(arg.flag + '=' + valueString(item));
    }
  }
  if (positionals.length) argv.push('--', ...positionals);
  return { argv, destinations };
}
