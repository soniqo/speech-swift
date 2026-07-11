import AudioCLILib
import Foundation

// Backward-compat: when invoked via the deprecated `audio` binary name, emit a one-line
// deprecation notice. Inferred from argv[0] so each binary self-identifies.
if let argv0 = CommandLine.arguments.first,
   (argv0 as NSString).lastPathComponent == "audio" {
    FileHandle.standardError.write(Data(
        "warning: `audio` is a deprecated alias and will be removed in a future release — use `speech` instead.\n".utf8
    ))
}

AudioCLI.main()
