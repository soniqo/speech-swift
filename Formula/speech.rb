class Speech < Formula
  desc "AI speech models for Apple Silicon — ASR, TTS, speech-to-speech"
  homepage "https://github.com/soniqo/speech-swift"
  url "https://github.com/soniqo/speech-swift/releases/download/v0.0.11/audio-macos-arm64.tar.gz"
  sha256 "51c5ab89fd78a4a97fcc84fec4c329d26ac9f395096c97ce761ff3ba8fb952e0"
  license "Apache-2.0"

  depends_on arch: :arm64
  depends_on :macos

  def install
    libexec.install "audio", "audio-server", "mlx.metallib"
    libexec.install "Qwen3Speech_KokoroTTS.bundle"
    bin.write_exec_script libexec/"audio"
    bin.write_exec_script libexec/"audio-server"
  end

  test do
    assert_match "AI speech models", shell_output("#{bin}/audio --help")
    assert_match "HTTP API server", shell_output("#{bin}/audio-server --help")
  end
end
