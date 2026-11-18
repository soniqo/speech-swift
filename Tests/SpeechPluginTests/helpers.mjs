export function wav({ seconds = 0.25, rate = 16000, channels = 1 } = {}) {
  const size = Math.floor(seconds * rate) * channels * 2;
  const bytes = Buffer.alloc(44 + size);
  bytes.write('RIFF', 0); bytes.writeUInt32LE(bytes.length - 8, 4); bytes.write('WAVE', 8);
  bytes.write('fmt ', 12); bytes.writeUInt32LE(16, 16); bytes.writeUInt16LE(1, 20);
  bytes.writeUInt16LE(channels, 22); bytes.writeUInt32LE(rate, 24);
  bytes.writeUInt32LE(rate * channels * 2, 28); bytes.writeUInt16LE(channels * 2, 32);
  bytes.writeUInt16LE(16, 34); bytes.write('data', 36); bytes.writeUInt32LE(size, 40);
  for (let i = 44; i < bytes.length; i += 2) bytes.writeInt16LE(Math.sin(i / 9) * 1000, i);
  return bytes;
}


export function inspectWav(bytes) {
  if (bytes.length < 44 || bytes.toString('ascii', 0, 4) !== 'RIFF'
      || bytes.toString('ascii', 8, 12) !== 'WAVE') {
    throw new Error('Select a PCM16 WAV audio file.');
  }
  const end = bytes.readUInt32LE(4) + 8;
  if (end !== bytes.length) throw new Error('The WAV file is incomplete or has trailing data.');
  let format;
  let audioBytes;
  let offset = 12;
  while (offset + 8 <= end) {
    const kind = bytes.toString('ascii', offset, offset + 4);
    const size = bytes.readUInt32LE(offset + 4);
    const start = offset + 8;
    if (start + size > end) throw new Error('The WAV file has an incomplete chunk.');
    if (kind === 'fmt ') {
      if (format || size < 16) throw new Error('The WAV format chunk is invalid.');
      format = {
        encoding: bytes.readUInt16LE(start),
        channels: bytes.readUInt16LE(start + 2),
        sample_rate: bytes.readUInt32LE(start + 4),
        byteRate: bytes.readUInt32LE(start + 8),
        blockAlign: bytes.readUInt16LE(start + 12),
        bits: bytes.readUInt16LE(start + 14),
      };
    } else if (kind === 'data') {
      if (audioBytes !== undefined) throw new Error('Use a WAV with one audio data chunk.');
      audioBytes = size;
    }
    offset = start + size + (size % 2);
  }
  if (!format || audioBytes === undefined || audioBytes === 0 || offset !== end
      || format.encoding !== 1 || format.bits !== 16 || ![1, 2].includes(format.channels)
      || format.sample_rate < 8000 || format.sample_rate > 192000
      || format.blockAlign !== format.channels * 2
      || format.byteRate !== format.sample_rate * format.blockAlign
      || audioBytes % format.blockAlign !== 0) {
    throw new Error('Use a nonempty 16-bit PCM WAV with one or two channels.');
  }
  const duration_seconds = audioBytes / format.byteRate;
  if (duration_seconds > 300) {
    throw new Error('Split recordings into WAV clips of at most five minutes.');
  }
  return { sample_rate: format.sample_rate, channels: format.channels, duration_seconds };
}
