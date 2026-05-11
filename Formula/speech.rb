class Speech < Formula
  desc "AI speech models for Apple Silicon — ASR, TTS, speech-to-speech"
  homepage "https://github.com/soniqo/speech-swift"
  url "https://github.com/soniqo/speech-swift/releases/download/v0.0.13/audio-macos-arm64.tar.gz"
  sha256 "5dd67bf8fb7c298d197e098ae5202efe72eeea84785696a7d1c885f55594d4f6"
  license "Apache-2.0"

  depends_on arch: :arm64
  depends_on :macos

  def install
    # `speech`/`speech-server` are canonical; `audio`/`audio-server` are
    # deprecated aliases retained for one release cycle. Older release tarballs
    # only carry the `audio` names, so install whichever is present.
    %w[speech speech-server audio audio-server mlx.metallib].each do |f|
      libexec.install f if File.exist?(f)
    end
    libexec.install "Qwen3Speech_KokoroTTS.bundle" if File.exist?("Qwen3Speech_KokoroTTS.bundle")

    %w[speech speech-server audio audio-server].each do |name|
      bin.write_exec_script libexec/name if (libexec/name).exist?
    end
  end

  test do
    primary = (bin/"speech").exist? ? "speech" : "audio"
    server  = (bin/"speech-server").exist? ? "speech-server" : "audio-server"
    assert_match "AI speech models", shell_output("#{bin}/#{primary} --help")
    assert_match "HTTP API server", shell_output("#{bin}/#{server} --help")
  end
end
