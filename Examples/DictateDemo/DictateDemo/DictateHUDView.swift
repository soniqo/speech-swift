import SwiftUI

/// Floating HUD window showing live transcription.
struct DictateHUDView: View {
    @Bindable var viewModel: DictateViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Header
            HStack {
                Circle()
                    .fill(viewModel.isRecording ? .red : .gray)
                    .frame(width: 8, height: 8)
                Text(viewModel.isRecording ? "Listening..." : "Paused")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                if viewModel.isRecording {
                    // Audio level indicator
                    RoundedRectangle(cornerRadius: 2)
                        .fill(.green.opacity(0.7))
                        .frame(width: CGFloat(viewModel.audioLevel * 100), height: 4)
                        .animation(.easeOut(duration: 0.1), value: viewModel.audioLevel)
                }
            }

            // Transcript
            if viewModel.fullText.isEmpty {
                Text("Speak to transcribe...")
                    .foregroundStyle(.tertiary)
                    .font(.body)
            } else {
                ScrollView {
                    HStack {
                        Text(committedPart)
                            + Text(partialPart)
                        Spacer()
                    }
                }
                .frame(maxHeight: 200)
            }
        }
        .padding()
        .frame(width: 350, alignment: .leading)
        .background(.ultraThinMaterial)
    }

    private var committedPart: AttributedString {
        var s = AttributedString(viewModel.committedText)
        s.font = .system(.body, design: .rounded)
        return s
    }

    private var partialPart: AttributedString {
        guard !viewModel.partialText.isEmpty else { return AttributedString() }
        let prefix = viewModel.committedText.isEmpty ? "" : " "
        var s = AttributedString(prefix + viewModel.partialText)
        s.font = .system(.body, design: .rounded)
        s.foregroundColor = .secondary
        return s
    }
}
