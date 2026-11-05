import { build } from 'esbuild';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const root = fileURLToPath(new URL('../', import.meta.url));
const manifest = JSON.parse(await readFile(resolve(root, 'plugin.json'), 'utf8'));
const result = await build({
  entryPoints: [resolve(root, 'src/server.mjs')],
  outfile: resolve(root, 'dist/server.cjs'),
  bundle: true,
  platform: 'node',
  target: 'node22',
  format: 'cjs',
  legalComments: 'eof',
  metafile: true,
  define: { SPEECH_PLUGIN_VERSION: JSON.stringify(manifest.version) },
});
await mkdir(resolve(root, 'dist'), { recursive: true });
await writeFile(resolve(root, 'dist/cli-catalog.json'), await readFile(resolve(root, 'src/cli-catalog.json')));
const bases = new Set();
for (const input of Object.keys(result.metafile.inputs).filter((path) => path.includes('node_modules/'))) {
  let base = dirname(resolve(input));
  while (base !== dirname(base)) {
    try {
      const metadata = JSON.parse(await readFile(resolve(base, 'package.json'), 'utf8'));
      if (metadata.name) { bases.add(base); break; }
    } catch {}
    base = dirname(base);
  }
}
const notices = await Promise.all([...bases].sort().map(async (base) => {
  const pkg = JSON.parse(await readFile(resolve(base, 'package.json'), 'utf8'));
  if (!pkg.name) throw new Error('Bundled dependency has no package name: ' + base);
  let license;
  for (const filename of ['LICENSE', 'LICENSE.md', 'LICENSE.txt', 'license', 'license.md']) {
    try { license = await readFile(resolve(base, filename), 'utf8'); break; } catch {}
  }
  if (!license) throw new Error('Missing license for ' + pkg.name);
  return pkg.name + ' ' + pkg.version + '\n' + license;
}));
await writeFile(resolve(root, 'dist/THIRD_PARTY_NOTICES.txt'), notices.join('\n\n'));
