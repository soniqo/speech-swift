#if canImport(SwiftUI)
import SwiftUI

/// A scrolling transcript that distinguishes committed (final) lines from the
/// in-progress partial line currently being recognized. Backend-agnostic — feed
/// it any data you derive from your ASR model's output.
///
/// ```swift
/// TranscriptionView(
///     finals: store.finalLines,
///     currentPartial: store.partialLine
/// )
/// ```
///
/// Pair with ``TranscriptionStore`` if you want a ready-made `@Observable`
/// model wired up to handle finals/partials updates.
public struct TranscriptionView: View {
    public let finals: [String]
    public let currentPartial: String?
    public let placeholder: String
    public let autoScroll: Bool

    public init(
        finals: [String],
        currentPartial: String? = nil,
        placeholder: String = "Listening…",
        autoScroll: Bool = true
    ) {
        self.finals = finals
        self.currentPartial = currentPartial
        self.placeholder = placeholder
        self.autoScroll = autoScroll
    }

    public var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 8) {
                    if finals.isEmpty && (currentPartial?.isEmpty ?? true) {
                        Text(placeholder)
                            .foregroundStyle(.secondary)
                            .italic()
                            .id(Self.placeholderID)
                    }

                    ForEach(Array(finals.enumerated()), id: \.offset) { index, line in
                        Text(line)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .id(Self.finalID(index))
                    }

                    if let partial = currentPartial, !partial.isEmpty {
                        Text(partial)
                            .foregroundStyle(.secondary)
                            .italic()
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .id(Self.partialID)
                    }
                }
                .padding(.horizontal, 4)
            }
            .onChange(of: finals.count) { _, _ in
                guard autoScroll else { return }
                scrollToBottom(proxy: proxy)
            }
            .onChange(of: currentPartial ?? "") { _, _ in
                guard autoScroll else { return }
                scrollToBottom(proxy: proxy)
            }
        }
        .accessibilityLabel("Transcript")
    }

    private func scrollToBottom(proxy: ScrollViewProxy) {
        let target: String
        if currentPartial?.isEmpty == false {
            target = Self.partialID
        } else if !finals.isEmpty {
            target = Self.finalID(finals.count - 1)
        } else {
            target = Self.placeholderID
        }
        withAnimation(.easeOut(duration: 0.2)) {
            proxy.scrollTo(target, anchor: .bottom)
        }
    }

    private static let placeholderID = "speechui.transcript.placeholder"
    private static let partialID = "speechui.transcript.partial"
    private static func finalID(_ index: Int) -> String { "speechui.transcript.final.\(index)" }
}

/// Optional `@Observable` model for driving ``TranscriptionView``. Wires up
/// finals + partial state and exposes a single `apply(text:isFinal:)` entry
/// point so callers can plug any backend's output into it without coupling.
///
/// ```swift
/// let store = TranscriptionStore()
///
/// for await partial in model.transcribeStream(audio: samples, sampleRate: 16000) {
///     store.apply(text: partial.text, isFinal: partial.isFinal)
/// }
/// ```
@Observable
@MainActor
public final class TranscriptionStore {
    public private(set) var finalLines: [String] = []
    public private(set) var currentPartial: String? = nil

    public init() {}

    /// Apply a transcript update from any streaming ASR backend.
    /// - Parameters:
    ///   - text: The transcript text. May be a partial in-progress line or a
    ///     committed final.
    ///   - isFinal: When `true`, `text` is moved to ``finalLines`` and the
    ///     in-progress partial is cleared. When `false`, ``currentPartial`` is
    ///     updated and finals are untouched.
    public func apply(text: String, isFinal: Bool) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if isFinal {
            if !trimmed.isEmpty {
                finalLines.append(trimmed)
            }
            currentPartial = nil
        } else {
            currentPartial = trimmed.isEmpty ? nil : trimmed
        }
    }

    /// Clear all transcript state.
    public func reset() {
        finalLines.removeAll()
        currentPartial = nil
    }
}
#endif
