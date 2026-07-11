import SwiftUI
import UIKit

@MainActor
@Observable
final class BenchViewModel {
    var status: String = "Starting…"
    var rows: [BenchRow] = []
    private var started = false

    func start() {
        guard !started else { return }
        started = true
        // Run the whole suite off the main thread; hop back to update UI.
        Task.detached(priority: .userInitiated) {
            let core = BenchmarkCore()
            await core.run(
                onStatus: { s in Task { @MainActor in self.status = s } },
                onRow: { r in Task { @MainActor in self.rows.append(r) } }
            )
            await MainActor.run {
                self.status = "Done — \(self.rows.count) rows · results.json written"
            }
        }
    }
}

struct ContentView: View {
    @State private var vm = BenchViewModel()

    var body: some View {
        NavigationStack {
            List {
                Section("Status") {
                    Text(vm.status)
                        .font(.footnote.monospaced())
                        .foregroundStyle(.secondary)
                }
                Section("Results (RTF lower = faster)") {
                    if vm.rows.isEmpty {
                        Text("Running…").foregroundStyle(.secondary)
                    }
                    ForEach(vm.rows) { r in
                        VStack(alignment: .leading, spacing: 3) {
                            Text(r.label).font(.subheadline).bold()
                            Text(r.model)
                                .font(.caption2.monospaced())
                                .foregroundStyle(.secondary)
                            if let e = r.error {
                                Text(e).font(.caption2).foregroundStyle(.red).lineLimit(4)
                            } else {
                                Text("\(r.metric)  ·  \(Int(r.peakMB)) MB")
                                    .font(.callout.monospaced())
                                    .foregroundStyle(.orange)
                            }
                        }
                        .padding(.vertical, 2)
                    }
                }
            }
            .navigationTitle("CoreML Bench · iPhone")
        }
        .task { vm.start() }
        .onAppear { UIApplication.shared.isIdleTimerDisabled = true }
    }
}
