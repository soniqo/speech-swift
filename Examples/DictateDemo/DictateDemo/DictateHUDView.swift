import SwiftUI

struct DictateHUDView: View {
    @Bindable var viewModel: DictateViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Header
            HStack {
                Circle()
                    .fill(statusColor)
                    .frame(width: 8, height: 8)
                Text(statusText)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                if viewModel.isRecording {
                    Text("\(viewModel.wordCount) words")
                        .font(.caption2).monospacedDigit()
                        .foregroundStyle(.tertiary)
                }
            }

            // Transcript
            if viewModel.sentences.isEmpty && viewModel.partialText.isEmpty {
                Text("Speak to transcribe...")
                    .foregroundStyle(.tertiary)
                    .font(.body)
            } else {
                ScrollViewReader { proxy in
                    ScrollView {
                        VStack(alignment: .leading, spacing: 4) {
                            ForEach(Array(viewModel.sentences.enumerated()), id: \.offset) { i, sentence in
                                Text(sentence)
                                    .font(.system(.body, design: .rounded))
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .id(i)
                            }
                            if !viewModel.partialText.isEmpty {
                                Text(viewModel.partialText)
                                    .font(.system(.body, design: .rounded))
                                    .foregroundStyle(.secondary)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .id("partial")
                            }
                        }
                    }
                    .frame(maxHeight: 250)
                    .onChange(of: viewModel.sentences.count) {
                        withAnimation {
                            proxy.scrollTo(viewModel.sentences.count - 1, anchor: .bottom)
                        }
                    }
                    .onChange(of: viewModel.partialText) {
                        withAnimation {
                            proxy.scrollTo("partial", anchor: .bottom)
                        }
                    }
                }
            }
        }
        .padding()
        .frame(width: 380, alignment: .leading)
        .background(.ultraThinMaterial)
    }

    private var statusColor: Color {
        if !viewModel.isRecording { return .gray }
        return viewModel.isSpeechActive ? .green : .orange
    }

    private var statusText: String {
        if !viewModel.isRecording { return "Paused" }
        return viewModel.isSpeechActive ? "Listening..." : "Waiting for speech..."
    }
}
