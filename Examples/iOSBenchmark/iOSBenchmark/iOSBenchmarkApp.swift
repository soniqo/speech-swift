import SwiftUI
import Darwin

@main
struct iOSBenchmarkApp: App {
    init() {
        // Unbuffered stdout so devicectl --console streams [BENCH] lines live
        // (a piped stdout is block-buffered by default and hides progress).
        setvbuf(stdout, nil, _IONBF, 0)
    }
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
