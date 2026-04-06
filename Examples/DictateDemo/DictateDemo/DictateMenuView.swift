import SwiftUI

struct DictateMenuView: View {
    @Bindable var viewModel: DictateViewModel

    var body: some View {
        VStack(spacing: 8) {
            if viewModel.isLoading {
                Text(viewModel.loadingStatus)
                    .font(.caption)
            } else if !viewModel.modelLoaded {
                Button("Load Models") {
                    Task { await viewModel.loadModels() }
                }
            } else {
                Button {
                    viewModel.toggleRecording()
                } label: {
                    HStack {
                        Image(systemName: viewModel.isRecording ? "stop.circle.fill" : "mic.circle.fill")
                            .foregroundStyle(viewModel.isRecording ? .red : .accentColor)
                        Text(viewModel.isRecording ? "Stop Dictation" : "Start Dictation")
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .buttonStyle(.plain)
                .padding(.horizontal)

                if viewModel.isRecording {
                    Text(viewModel.isSpeechActive ? "Speech detected" : "Listening...")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .padding(.horizontal)
                }

                if !viewModel.fullText.isEmpty {
                    Divider()
                    Text(viewModel.fullText)
                        .font(.system(.body, design: .rounded))
                        .lineLimit(8)
                        .frame(maxWidth: 300, alignment: .leading)
                        .padding(.horizontal)

                    HStack {
                        Button("Copy") { viewModel.copyToClipboard() }
                        Button("Paste") { viewModel.pasteToFrontApp() }
                        Spacer()
                        Button("Clear") { viewModel.clearText() }
                    }
                    .padding(.horizontal)
                }
            }

            if let error = viewModel.errorMessage {
                Text(error).font(.caption).foregroundStyle(.red).padding(.horizontal)
            }

            Divider()
            Button("Quit") { NSApplication.shared.terminate(nil) }
                .padding(.horizontal)
        }
        .padding(.vertical, 8)
        .frame(minWidth: 280)
    }
}
