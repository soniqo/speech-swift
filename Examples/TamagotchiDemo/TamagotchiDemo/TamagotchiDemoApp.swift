import SwiftUI

@main
struct TamagotchiDemoApp: App {
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            CompanionChatView()
        }
    }
}
