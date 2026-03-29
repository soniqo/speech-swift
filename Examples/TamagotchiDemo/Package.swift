// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "TamagotchiDemo",
    platforms: [.iOS(.v17), .macOS(.v14)],
    dependencies: [
        .package(path: "../.."),
    ],
    targets: [
        .executableTarget(
            name: "TamagotchiDemo",
            dependencies: [
                .product(name: "Qwen3Chat", package: "qwen3-asr-swift"),
                .product(name: "KokoroTTS", package: "qwen3-asr-swift"),
                .product(name: "ParakeetASR", package: "qwen3-asr-swift"),
                .product(name: "SpeechVAD", package: "qwen3-asr-swift"),
                .product(name: "SpeechCore", package: "qwen3-asr-swift"),
                .product(name: "AudioCommon", package: "qwen3-asr-swift"),
            ],
            path: "TamagotchiDemo",
            exclude: ["Info.plist"]
        ),
    ]
)
