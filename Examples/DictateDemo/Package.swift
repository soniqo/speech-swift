// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "DictateDemo",
    platforms: [.macOS("15.0")],
    dependencies: [
        .package(path: "../.."),
    ],
    targets: [
        .executableTarget(
            name: "DictateDemo",
            dependencies: [
                .product(name: "ParakeetStreamingASR", package: "speech-swift"),
                .product(name: "AudioCommon", package: "speech-swift"),
            ],
            path: "DictateDemo",
            exclude: ["DictateDemo.entitlements", "Info.plist"]
        ),
    ]
)
