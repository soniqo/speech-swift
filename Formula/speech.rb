class Speech < Formula
  desc "AI speech models for Apple Silicon — ASR, TTS, speech-to-speech"
  homepage "https://github.com/soniqo/speech-swift"
  url "https://github.com/soniqo/speech-swift/releases/download/v0.0.8/audio-macos-arm64.tar.gz"
  sha256 "22b549125fd4c7b1c32df18fa67a2e94262261837b96aba04ec5e5399785673a"
  license "Apache-2.0"

  depends_on arch: :arm64
  depends_on :macos

  def install
    libexec.install "audio", "mlx.metallib"
    bin.write_exec_script libexec/"audio"
  end

  test do
    assert_match "AI speech models", shell_output("#{bin}/audio --help")
  end
end
