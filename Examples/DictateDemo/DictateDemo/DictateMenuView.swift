import SwiftUI

/// Menu bar dropdown with controls.
struct DictateMenuView: View {
    @Bindable var viewModel: DictateViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Status
            if viewModel.isLoading {
                HStack(spacing: 8) {
                    ProgressView()
                        .controlSize(.small)
                    Text(viewModel.loadingStatus)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal)
            } else if !viewModel.modelLoaded {
                Button("Load Model") {
                    Task { await viewModel.loadModel() }
                }
                .padding(.horizontal)
            } else {
                // Record button
                Button {
                    viewModel.toggleRecording()
                } label: {
                    HStack {
                        Image(systemName: viewModel.isRecording ? "stop.circle.fill" : "mic.circle.fill")
                            .foregroundStyle(viewModel.isRecording ? .red : .accentColor)
                        Text(viewModel.isRecording ? "Stop Dictation" : "Start Dictation")
                    }
                }
                .keyboardShortcut("d", modifiers: [.command, .shift])
                .padding(.horizontal)

                // Live transcript preview
                if !viewModel.fullText.isEmpty {
                    Divider()

                    Text(viewModel.fullText)
                        .font(.system(.body, design: .rounded))
                        .lineLimit(5)
                        .padding(.horizontal)
                        .frame(maxWidth: 300, alignment: .leading)

                    HStack(spacing: 8) {
                        Button("Copy") {
                            viewModel.copyToClipboard()
                        }
                        .keyboardShortcut("c", modifiers: [.command])

                        Button("Paste to App") {
                            viewModel.pasteToFrontApp()
                        }
                        .keyboardShortcut("v", modifiers: [.command, .shift])

                        Spacer()

                        Button("Clear") {
                            viewModel.clearText()
                        }
                    }
                    .padding(.horizontal)
                }
            }

            if let error = viewModel.errorMessage {
                Text(error)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .padding(.horizontal)
            }

            Divider()

            Button("Quit") {
                NSApplication.shared.terminate(nil)
            }
            .keyboardShortcut("q")
            .padding(.horizontal)
        }
        .padding(.vertical, 8)
        .frame(minWidth: 280)
        .task {
            if !viewModel.modelLoaded {
                await viewModel.loadModel()
            }
        }
    }
}
