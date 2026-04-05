import SwiftUI

@main
struct DictateDemoApp: App {
    @State private var viewModel = DictateViewModel()

    var body: some Scene {
        MenuBarExtra {
            DictateMenuView(viewModel: viewModel)
        } label: {
            Image(systemName: viewModel.isRecording ? "mic.fill" : "mic")
        }
        .menuBarExtraStyle(.window)

        Window("Dictate", id: "dictate-hud") {
            DictateHUDView(viewModel: viewModel)
        }
        .windowStyle(.hiddenTitleBar)
        .windowResizability(.contentSize)
        .defaultPosition(.topTrailing)
    }
}
