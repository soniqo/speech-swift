#!/usr/bin/env python3
"""Build Apple's fixed NumPy input bridge inside a Core ML conversion venv.

The stable coremltools 9.0 bridge can release NumPy references without the GIL
on newer macOS versions. This installs the upstream fix, leaving the 9.0
converter and the system Python environment intact.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile

REVISION = "a207b1a3452eafe02f2d870f486287834af2c82d"
REPOSITORY = "https://github.com/apple/coremltools.git"


def validate_environment(prefix, base_prefix, package):
    prefix, base_prefix, package = map(lambda value: Path(value).resolve(), [prefix, base_prefix, package])
    if prefix == base_prefix or not package.is_relative_to(prefix):
        raise ValueError("Run this helper inside an isolated venv containing coremltools 9.0")


def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def installed_bridge_receipt():
    spec = importlib.util.find_spec('coremltools')
    if spec is None:
        raise ValueError('Install coremltools==9.0 first')
    package = Path(spec.origin).resolve().parent
    path = package / '_inkling_native_bridge.json'
    if not path.is_file():
        raise ValueError('Run scripts/prepare_inkling_coreml_bridge.py before Core ML prediction checks')
    receipt = json.loads(path.read_text())
    if receipt.get('upstream_revision') != REVISION or receipt.get('binary_sha256') != digest(package / 'libcoremlpython.so'):
        raise ValueError('Core ML Python bridge does not match its pinned build receipt')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path,
                        default=Path(tempfile.gettempdir()) / f'speech-swift-coremltools-{REVISION[:12]}')
    parser.add_argument('--build-dir', type=Path)
    args = parser.parse_args()
    spec = importlib.util.find_spec('coremltools')
    if spec is None or importlib.metadata.version('coremltools') != '9.0':
        parser.error('Install coremltools==9.0 in this venv first')
    package = Path(spec.origin).resolve().parent
    validate_environment(sys.prefix, sys.base_prefix, package)
    if sys.platform != 'darwin' or sys.version_info[:2] != (3, 12):
        parser.error('This tested bridge setup requires macOS and Python 3.12')
    cmake = shutil.which('cmake')
    if cmake is None:
        parser.error('Install CMake before building the bridge')
    source = args.source_dir.expanduser().resolve()
    if not source.exists():
        subprocess.run(['git', 'clone', '--depth', '1', '--filter=blob:none', '--no-checkout', REPOSITORY, str(source)], check=True)
        subprocess.run(['git', '-C', str(source), 'fetch', '--depth', '1', 'origin', REVISION], check=True)
        subprocess.run(['git', '-C', str(source), 'checkout', '--detach', REVISION], check=True)
    actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != REVISION or subprocess.run(['git', '-C', str(source), 'diff', '--quiet', 'HEAD']).returncode:
        parser.error('The source must be the unchanged pinned upstream revision')
    build = (args.build_dir or source / 'build-bridge').expanduser().resolve()
    subprocess.run([cmake, '-S', str(source), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
        '-DCMAKE_OSX_DEPLOYMENT_TARGET=15.0', '-DCMAKE_POLICY_VERSION_MINIMUM=3.5', '-DOVERWRITE_PB_SOURCE=0',
        '-DPYTHON_EXECUTABLE:FILEPATH=' + sys.executable,
        '-DPYTHON_INCLUDE_DIR:PATH=' + sysconfig.get_config_var('INCLUDEPY'),
        '-DPYTHON_LIBRARY:FILEPATH=' + str(Path(sysconfig.get_config_var('LIBDIR')) / sysconfig.get_config_var('LDLIBRARY'))], check=True)
    subprocess.run([cmake, '--build', str(build), '--target', 'coremlpython', '-j', '2'], check=True)
    binary, target = source / 'coremltools/libcoremlpython.so', package / 'libcoremlpython.so'
    original_hash = digest(target)
    backup = target.with_name('libcoremlpython.so.original-' + original_hash[:12])
    if not backup.exists():
        shutil.copy2(target, backup)
    temporary = target.with_name('libcoremlpython.so.new')
    shutil.copy2(binary, temporary)
    os.replace(temporary, target)
    receipt = {'upstream_repository': REPOSITORY, 'upstream_revision': REVISION,
        'upstream_fix': 'https://github.com/apple/coremltools/pull/2829',
        'coremltools': '9.0', 'python': sys.version.split()[0],
        'binary_sha256': digest(target), 'previous_binary_sha256': original_hash,
        'cmake': subprocess.check_output([cmake, '--version'], text=True).splitlines()[0]}
    (package / '_inkling_native_bridge.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
