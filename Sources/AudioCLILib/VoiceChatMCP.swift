import ArgumentParser
import CoreFoundation
import Foundation
import MCP

#if canImport(Darwin)
import Darwin
#endif

#if canImport(System)
import System
#else
import SystemPackage
#endif

enum VoiceChatMCPError: Error, CustomStringConvertible, Sendable {
    case invalidConfiguration(String)
    case serverFailed(String)
    case toolCall(String)
    case clarificationRequired(field: String, message: String)
    case timedOut(String)

    var description: String {
        switch self {
        case .invalidConfiguration(let message):
            return "invalid MCP configuration: \(message)"
        case .serverFailed(let message):
            return "MCP server failed: \(message)"
        case .toolCall(let message):
            return "invalid MCP tool call: \(message)"
        case .clarificationRequired(let field, let message):
            return "MCP tool needs clarification for \(field): \(message)"
        case .timedOut(let operation):
            return "MCP \(operation) timed out"
        }
    }
}

public enum VoiceChatMCPWritePolicy: String, Codable, CaseIterable,
    ExpressibleByArgument, Sendable
{
    case deny
    case confirm
    case allow
}

struct VoiceChatMCPConfiguration: Decodable, Sendable, Equatable {
    struct Server: Decodable, Sendable, Equatable {
        enum Adapter: String, Decodable, Sendable, Equatable {
            case appleRemindersEventKit = "apple-reminders-eventkit"
        }

        let command: String
        let args: [String]
        let env: [String: String]
        let workingDirectory: String?
        let enabledTools: [String]
        let readOnlyTools: [String]
        let adapter: Adapter?

        private enum CodingKeys: String, CodingKey {
            case command, args, env, workingDirectory, enabledTools, readOnlyTools
            case adapter
        }

        init(
            command: String,
            args: [String] = [],
            env: [String: String] = [:],
            workingDirectory: String? = nil,
            enabledTools: [String],
            readOnlyTools: [String] = [],
            adapter: Adapter? = nil
        ) {
            self.command = command
            self.args = args
            self.env = env
            self.workingDirectory = workingDirectory
            self.enabledTools = enabledTools
            self.readOnlyTools = readOnlyTools
            self.adapter = adapter
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            command = try container.decode(String.self, forKey: .command)
            args = try container.decodeIfPresent([String].self, forKey: .args) ?? []
            env = try container.decodeIfPresent(
                [String: String].self, forKey: .env) ?? [:]
            workingDirectory = try container.decodeIfPresent(
                String.self, forKey: .workingDirectory)
            enabledTools = try container.decodeIfPresent(
                [String].self, forKey: .enabledTools) ?? []
            readOnlyTools = try container.decodeIfPresent(
                [String].self, forKey: .readOnlyTools) ?? []
            adapter = try container.decodeIfPresent(
                Adapter.self, forKey: .adapter)
        }
    }

    let mcpServers: [String: Server]

    static func load(from url: URL) throws -> Self {
        let fileSize = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize
        guard (fileSize ?? 0) <= 1_048_576 else {
            throw VoiceChatMCPError.invalidConfiguration(
                "configuration exceeds 1 MiB")
        }
        let data = try Data(contentsOf: url)
        guard data.count <= 1_048_576 else {
            throw VoiceChatMCPError.invalidConfiguration(
                "configuration exceeds 1 MiB")
        }
        let configuration = try JSONDecoder().decode(Self.self, from: data)
        guard !configuration.mcpServers.isEmpty else {
            throw VoiceChatMCPError.invalidConfiguration(
                "mcpServers must contain at least one server")
        }
        return configuration
    }

    func selectedServers(names: [String]) throws -> [(String, Server)] {
        let selectedNames = names.isEmpty
            ? mcpServers.keys.sorted()
            : Array(Set(names)).sorted()
        let unknown = selectedNames.filter { mcpServers[$0] == nil }
        guard unknown.isEmpty else {
            throw VoiceChatMCPError.invalidConfiguration(
                "unknown server names: \(unknown.joined(separator: ", "))")
        }

        return try selectedNames.map { name in
            let server = mcpServers[name]!
            guard !server.command.trimmingCharacters(
                in: .whitespacesAndNewlines).isEmpty else {
                throw VoiceChatMCPError.invalidConfiguration(
                    "server \(name) has an empty command")
            }
            guard !server.command.hasPrefix("-") else {
                throw VoiceChatMCPError.invalidConfiguration(
                    "server \(name) command cannot begin with a dash")
            }
            guard !server.enabledTools.isEmpty else {
                throw VoiceChatMCPError.invalidConfiguration(
                    "server \(name) must explicitly list enabledTools")
            }
            guard server.enabledTools.allSatisfy({ tool in
                !tool.isEmpty && tool.unicodeScalars.allSatisfy(\.isASCII)
            }) else {
                throw VoiceChatMCPError.invalidConfiguration(
                    "server \(name) enabledTools must be non-empty ASCII names")
            }
            guard Set(server.enabledTools).count == server.enabledTools.count else {
                throw VoiceChatMCPError.invalidConfiguration(
                    "server \(name) contains duplicate enabledTools")
            }
            let unknownReadOnly = Set(server.readOnlyTools)
                .subtracting(server.enabledTools)
            guard unknownReadOnly.isEmpty else {
                throw VoiceChatMCPError.invalidConfiguration(
                    "server \(name) marks disabled tools read-only: "
                        + unknownReadOnly.sorted().joined(separator: ", "))
            }
            if server.adapter == .appleRemindersEventKit {
                let supported = Set([
                    "create_reminder", "list_reminders", "update_reminder",
                ])
                let unsupported = Set(server.enabledTools)
                    .subtracting(supported)
                guard unsupported.isEmpty else {
                    throw VoiceChatMCPError.invalidConfiguration(
                        "server \(name) uses unsupported Apple Reminders aliases: "
                            + unsupported.sorted().joined(separator: ", "))
                }
                guard !server.enabledTools.contains("update_reminder")
                        || server.enabledTools.contains("list_reminders") else {
                    throw VoiceChatMCPError.invalidConfiguration(
                        "server \(name) must enable list_reminders when update_reminder is enabled")
                }
                guard !server.readOnlyTools.contains("create_reminder") else {
                    throw VoiceChatMCPError.invalidConfiguration(
                        "server \(name) cannot mark create_reminder read-only")
                }
                guard !server.readOnlyTools.contains("update_reminder") else {
                    throw VoiceChatMCPError.invalidConfiguration(
                        "server \(name) cannot mark update_reminder read-only")
                }
            }
            return (name, server)
        }
    }
}

struct VoiceChatMCPTool: Sendable, Equatable {
    enum Access: String, Sendable, Equatable {
        case read
        case write
    }

    let serverName: String
    let name: String
    let description: String
    let inputSchemaJSON: String
    let access: Access
}

/// Session-scoped exact references for provider identifiers that are expensive
/// for the native function head to copy token by token. This table does not
/// inspect user speech or resolve names: the model receives a short identifier
/// in a tool result and must return that same identifier in a later tool call.
struct VoiceChatMCPOpaqueReferenceTable: Sendable, Equatable {
    private(set) var providerIDByReference: [String: String] = [:]
    private var referenceByProviderID: [String: String] = [:]
    private var nextReference = 1

    mutating func modelFacingRecords(
        _ records: [[String: Any]]
    ) -> [[String: Any]] {
        records.map { record in
            guard let providerID = record["id"] as? String,
                  !providerID.isEmpty else { return record }
            var exposed = record
            exposed["id"] = reference(for: providerID)
            return exposed
        }
    }

    func providerID(for reference: String) -> String? {
        providerIDByReference[reference]
    }

    static func isManagedReference(_ value: String) -> Bool {
        guard value.first == "r", value.count > 1 else { return false }
        return value.dropFirst().allSatisfy(\.isNumber)
    }

    private mutating func reference(for providerID: String) -> String {
        if let existing = referenceByProviderID[providerID] {
            return existing
        }
        var candidate: String
        repeat {
            candidate = "r\(nextReference)"
            nextReference += 1
        } while providerIDByReference[candidate] != nil
        providerIDByReference[candidate] = providerID
        referenceByProviderID[providerID] = candidate
        return candidate
    }
}

protocol VoiceChatMCPToolExecuting: Sendable {
    func tool(named name: String) async -> VoiceChatMCPTool?
    func callTool(name: String, argumentsJSON: String) async throws -> String
}

actor VoiceChatMCPRuntime: VoiceChatMCPToolExecuting {
    private struct Connection {
        let name: String
        let client: Client
        let transport: StdioTransport
        let process: Process
        let inputPipe: Pipe
        let outputPipe: Pipe
        let errorPipe: Pipe
    }

    private struct ToolRoute {
        enum Adapter: Sendable, Equatable {
            case passthrough
            case appleRemindersEventKit
        }

        let tool: VoiceChatMCPTool
        let client: Client
        let providerToolName: String
        let adapter: Adapter
    }

    private let timeoutSeconds: Double
    /// The same clock snapshot and timezone embedded in the model prompt.
    /// Keeping them session-scoped prevents "tomorrow" from moving if a live
    /// conversation crosses midnight before the function call is emitted.
    private let referenceDate: Date
    private let calendar: Calendar
    private var connections: [Connection] = []
    private var routes: [String: ToolRoute] = [:]
    private var appleReminderReferences = VoiceChatMCPOpaqueReferenceTable()
    private init(
        timeoutSeconds: Double,
        referenceDate: Date,
        calendar: Calendar
    ) {
        self.timeoutSeconds = timeoutSeconds
        self.referenceDate = referenceDate
        self.calendar = calendar
    }

    static func start(
        configurationURL: URL,
        selectedServerNames: [String],
        timeoutSeconds: Double,
        referenceDate: Date = Date(),
        calendar: Calendar = .current
    ) async throws -> VoiceChatMCPRuntime {
        let configuration = try VoiceChatMCPConfiguration.load(
            from: configurationURL)
        let selected = try configuration.selectedServers(
            names: selectedServerNames)
        let enabledCount = selected.reduce(0) { $0 + $1.1.enabledTools.count }
        guard enabledCount <= 5 else {
            throw VoiceChatMCPError.invalidConfiguration(
                "VoiceChat supports at most five enabled tools per session; found \(enabledCount)")
        }
        guard timeoutSeconds.isFinite, timeoutSeconds > 0,
              timeoutSeconds <= 120 else {
            throw VoiceChatMCPError.invalidConfiguration(
                "timeout must be between 0 and 120 seconds")
        }

        let runtime = VoiceChatMCPRuntime(
            timeoutSeconds: timeoutSeconds,
            referenceDate: referenceDate,
            calendar: calendar)
        do {
            for (name, server) in selected {
                try await runtime.connect(name: name, configuration: server)
            }
            return runtime
        } catch {
            await runtime.shutdown()
            throw error
        }
    }

    func availableTools() -> [VoiceChatMCPTool] {
        routes.values.map(\.tool).sorted { $0.name < $1.name }
    }

    func availableToolsJSON() throws -> String {
        let definitions: [[String: Any]] = try availableTools().map { tool in
            try Self.modelFacingToolDefinition(tool)
        }
        let json = try compactASCIIJSON(definitions)
        guard json.utf8.count <= 48_000 else {
            throw VoiceChatMCPError.invalidConfiguration(
                "enabled tool definitions exceed 48 KiB")
        }
        return json
    }

    /// Canonical reminder aliases are intentionally package-visible for
    /// schema regression tests. Production installation still verifies that
    /// the pinned provider exposes `reminders_tasks` before using them.
    static func appleEventKitAliasTool(
        serverName: String,
        alias: String,
        readOnly: Bool
    ) throws -> VoiceChatMCPTool {
        let description: String
        let schema: String
        switch alias {
        case "list_reminders":
            description = "List active reminders across every reminder list, including names, lists, due dates, and completion state"
            schema = #"{"type":"object","properties":{"search":{"type":"string","description":"Optional reminder name search; omit to list every active reminder"}},"additionalProperties":false}"#
        case "create_reminder":
            description = "Create a new reminder. Name is the only required field; omit list to use the system default and omit every optional field the user did not specify"
            schema = #"{"type":"object","properties":{"name":{"type":"string","description":"Reminder title spoken by the user"},"list":{"type":"string","description":"Optional exact reminder-list name"},"body":{"type":"string","description":"Optional note text"},"due_date":{"type":"string","description":"Optional local date and time as YYYY-MM-DD HH:mm; omit unless the user specified both"},"priority":{"type":"number","description":"Optional 0, 1, 5, or 9; omit unless explicitly requested"}},"required":["name"],"additionalProperties":false}"#
        case "update_reminder":
            description = "Update one clearly identified reminder using the short id returned by list_reminders. If the user has not identified which reminder to change, ask instead of choosing one. Set completed true to remove it from active reminders. Include only fields that should change"
            schema = #"{"type":"object","properties":{"id":{"type":"string","description":"Short reminder id returned by list_reminders"},"name":{"type":"string","description":"Optional new name"},"list":{"type":"string","description":"Optional exact reminder-list name"},"body":{"type":"string","description":"Optional replacement note text"},"due_date":{"type":"string","description":"Optional local date and time as YYYY-MM-DD HH:mm"},"priority":{"type":"number","description":"Optional 0, 1, 5, or 9"},"completed":{"type":"boolean","description":"Set true to remove this reminder from active reminders"}},"required":["id"],"additionalProperties":false}"#
        default:
            throw VoiceChatMCPError.invalidConfiguration(
                "unsupported Apple Reminders alias: \(alias)")
        }
        return VoiceChatMCPTool(
            serverName: serverName,
            name: alias,
            description: description,
            inputSchemaJSON: schema,
            access: readOnly ? .read : .write)
    }

    /// Present provider schemas in the compact form consumed by the trained
    /// function head. Required arguments remain required: the runtime does not
    /// infer tool intent or synthesize missing values from the transcript.
    static func modelFacingToolDefinition(
        _ tool: VoiceChatMCPTool
    ) throws -> [String: Any] {
        let schemaData = Data(tool.inputSchemaJSON.utf8)
        guard let schema = try JSONSerialization.jsonObject(with: schemaData)
            as? [String: Any] else {
            throw VoiceChatMCPError.invalidConfiguration(
                "tool \(tool.name) input schema must be a JSON object")
        }
        return [
            "name": tool.name,
            "description": asciiSanitized(
                tool.description, maximumCharacters: 520),
            "parameters": schema,
        ]
    }

    func tool(named name: String) -> VoiceChatMCPTool? {
        routes[name]?.tool
    }

    func callTool(name: String, argumentsJSON: String) async throws -> String {
        guard let route = routes[name] else {
            throw VoiceChatMCPError.toolCall(
                "tool \(name) is not enabled")
        }
        let providerCall: VoiceChatFunctionCall
        switch route.adapter {
        case .passthrough:
            providerCall = VoiceChatFunctionCall(
                name: route.providerToolName,
                argumentsJSON: argumentsJSON)
        case .appleRemindersEventKit:
            let providerArgumentsJSON = try Self.appleEventKitProviderArguments(
                toolName: name,
                modelArgumentsJSON: argumentsJSON,
                references: appleReminderReferences)
            providerCall = try Self.appleEventKitProviderCall(
                toolName: name,
                argumentsJSON: providerArgumentsJSON,
                referenceDate: referenceDate,
                calendar: calendar)
        }
        let response = try await callProvider(
            route: route,
            publicToolName: name,
            providerCall: providerCall)
        return response
    }

    private func callProvider(
        route: ToolRoute,
        publicToolName: String,
        providerCall: VoiceChatFunctionCall
    ) async throws -> String {
        let data = Data(providerCall.argumentsJSON.utf8)
        let arguments = try JSONDecoder().decode(
            [String: Value].self, from: data)
        let result: (content: [Tool.Content], isError: Bool?)
        do {
            result = try await withMCPTimeout(
                seconds: timeoutSeconds,
                operation: "tool \(publicToolName)"
            ) {
                try await route.client.callTool(
                    name: providerCall.name, arguments: arguments)
            }
        } catch {
            if case VoiceChatMCPError.timedOut = error {
                await shutdown()
            }
            throw error
        }

        var textParts: [String] = []
        for content in result.content {
            switch content {
            case .text(let text, _, _):
                textParts.append(text)
            case .image(_, let mimeType, _, _):
                textParts.append("[image \(mimeType)]")
            case .audio(_, let mimeType, _, _):
                textParts.append("[audio \(mimeType)]")
            case .resource:
                textParts.append("[resource]")
            case .resourceLink(let uri, let name, _, _, _, _):
                textParts.append("[resource \(name): \(uri)]")
            }
        }
        let joined = textParts.joined(separator: "\n")
        let adapted: String
        if !(result.isError ?? false),
           route.adapter == .appleRemindersEventKit {
            adapted = try Self.appleEventKitCanonicalResult(
                toolName: publicToolName,
                providerText: joined)
        } else {
            adapted = joined
        }
        let limit = route.adapter == .appleRemindersEventKit ? 6_000 : 1_500
        let bounded = String(adapted.prefix(limit))
        let modelResult: Any
        if !(result.isError ?? false),
           route.adapter == .appleRemindersEventKit,
           publicToolName == "list_reminders",
           let data = adapted.data(using: .utf8),
           let records = try? JSONSerialization.jsonObject(with: data)
                as? [[String: Any]] {
            // Keep flattened reminder records as JSON values. Encoding the
            // array inside a JSON string adds escapes and function-response
            // tokens without carrying any additional model-visible meaning.
            // Replace long EventKit UUIDs with exact session references so a
            // subsequent native update call does not spend dozens of function
            // tokens copying an opaque provider implementation detail.
            modelResult = appleReminderReferences.modelFacingRecords(records)
        } else {
            modelResult = bounded
        }
        let succeeded = !(result.isError ?? false)
        var envelope: [String: Any] = ["ok": succeeded]
        // The native call immediately precedes its response in the function
        // channel, so repeating the tool name in a successful Apple Reminders
        // result carries no information. Keep it on failures, where naming the
        // failed operation helps the model recover, and on arbitrary MCP
        // adapters whose response contracts are not controlled here.
        if !succeeded || route.adapter != .appleRemindersEventKit {
            envelope["tool"] = publicToolName
        }
        // The native function history already contains every create/update
        // argument. A successful EventKit write therefore needs only the
        // provider's success bit; replaying its prose duplicates those values
        // through both the 11B and EAR-TTS caches. Reads and arbitrary MCP
        // tools retain their complete bounded result.
        let isCompactReminderWrite = succeeded
            && route.adapter == .appleRemindersEventKit
            && ["create_reminder", "update_reminder"]
                .contains(publicToolName)
        if !isCompactReminderWrite {
            envelope[(result.isError ?? false) ? "error" : "result"] = modelResult
        }
        return try compactASCIIJSON(envelope)
    }

    func shutdown() async {
        let active = connections
        connections.removeAll()
        routes.removeAll()
        appleReminderReferences = VoiceChatMCPOpaqueReferenceTable()
        for connection in active {
            await connection.client.disconnect()
            connection.errorPipe.fileHandleForReading.readabilityHandler = nil
            connection.inputPipe.fileHandleForWriting.closeFile()
            connection.outputPipe.fileHandleForReading.closeFile()
            connection.errorPipe.fileHandleForReading.closeFile()
            await Self.terminateAndWait(connection.process)
        }
    }

    private func connect(
        name: String,
        configuration: VoiceChatMCPConfiguration.Server
    ) async throws {
        let process = Process()
        let inputPipe = Pipe()
        let outputPipe = Pipe()
        let errorPipe = Pipe()

        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = [configuration.command] + configuration.args
        process.standardInput = inputPipe
        process.standardOutput = outputPipe
        process.standardError = errorPipe
        if let directory = configuration.workingDirectory {
            process.currentDirectoryURL = URL(fileURLWithPath: directory)
        }
        process.environment = Self.restrictedProcessEnvironment(
            parent: ProcessInfo.processInfo.environment,
            configured: configuration.env)
        errorPipe.fileHandleForReading.readabilityHandler = { handle in
            if handle.availableData.isEmpty {
                handle.readabilityHandler = nil
            }
        }

        do {
            try process.run()
        } catch {
            errorPipe.fileHandleForReading.readabilityHandler = nil
            throw VoiceChatMCPError.serverFailed(
                "could not start \(name): \(error)")
        }

        let transport = StdioTransport(
            input: FileDescriptor(
                rawValue: outputPipe.fileHandleForReading.fileDescriptor),
            output: FileDescriptor(
                rawValue: inputPipe.fileHandleForWriting.fileDescriptor))
        let client = Client(
            name: "speech-swift-voicechat",
            version: "1.0.0")
        do {
            _ = try await withMCPTimeout(
                // A pinned npx server may compile its small EventKit helper
                // on first use. Keep that one-time startup allowance separate
                // from the per-tool timeout used during a live conversation.
                seconds: max(timeoutSeconds, 60),
                operation: "server initialization"
            ) {
                try await client.connect(transport: transport)
            }

            var discovered: [Tool] = []
            var cursor: String?
            repeat {
                let pageCursor = cursor
                let page = try await withMCPTimeout(
                    seconds: timeoutSeconds,
                    operation: "tool discovery"
                ) {
                    try await client.listTools(cursor: pageCursor)
                }
                discovered.append(contentsOf: page.tools)
                cursor = page.nextCursor
            } while cursor != nil

            var discoveredByName: [String: Tool] = [:]
            for tool in discovered {
                guard discoveredByName[tool.name] == nil else {
                    throw VoiceChatMCPError.serverFailed(
                        "server \(name) returned duplicate tool \(tool.name)")
                }
                discoveredByName[tool.name] = tool
            }
            if configuration.adapter == .appleRemindersEventKit {
                try installAppleRemindersEventKitAliases(
                    serverName: name,
                    configuration: configuration,
                    discoveredByName: discoveredByName,
                    client: client)
            } else {
                let missing = configuration.enabledTools.filter {
                    discoveredByName[$0] == nil
                }
                guard missing.isEmpty else {
                    throw VoiceChatMCPError.invalidConfiguration(
                        "server \(name) does not expose enabled tools: "
                            + missing.joined(separator: ", "))
                }

                for toolName in configuration.enabledTools {
                    guard routes[toolName] == nil else {
                        throw VoiceChatMCPError.invalidConfiguration(
                            "duplicate tool name across MCP servers: \(toolName)")
                    }
                    let discoveredTool = discoveredByName[toolName]!
                    let schema = try String(
                        decoding: JSONEncoder.sorted.encode(
                            discoveredTool.inputSchema),
                        as: UTF8.self)
                    guard schema.utf8.count <= 8_192 else {
                        throw VoiceChatMCPError.invalidConfiguration(
                            "tool \(toolName) input schema exceeds 8 KiB")
                    }
                    let definition = VoiceChatMCPTool(
                        serverName: name,
                        name: toolName,
                        description: asciiSanitized(
                            discoveredTool.description ?? toolName,
                            maximumCharacters: 280),
                        inputSchemaJSON: asciiEscapedJSON(schema),
                        access: configuration.readOnlyTools.contains(toolName)
                            ? .read : .write)
                    routes[toolName] = ToolRoute(
                        tool: definition,
                        client: client,
                        providerToolName: toolName,
                        adapter: .passthrough)
                }
            }
            connections.append(Connection(
                name: name,
                client: client,
                transport: transport,
                process: process,
                inputPipe: inputPipe,
                outputPipe: outputPipe,
                errorPipe: errorPipe))
        } catch {
            await client.disconnect()
            errorPipe.fileHandleForReading.readabilityHandler = nil
            inputPipe.fileHandleForWriting.closeFile()
            outputPipe.fileHandleForReading.closeFile()
            errorPipe.fileHandleForReading.closeFile()
            await Self.terminateAndWait(process)
            throw VoiceChatMCPError.serverFailed(
                "\(name): \(error); server stderr was omitted")
        }
    }

    /// MCP servers are child processes owned by this runtime. Give them a
    /// short graceful-exit window after stdio closes, then ensure shutdown
    /// cannot leave a wedged server behind indefinitely.
    private static func terminateAndWait(_ process: Process) async {
        guard process.isRunning else { return }
        process.terminate()
        for _ in 0 ..< 25 {
            guard process.isRunning else { return }
            try? await Task.sleep(nanoseconds: 20_000_000)
        }
        #if canImport(Darwin)
        if process.isRunning {
            _ = Darwin.kill(process.processIdentifier, SIGKILL)
        }
        #endif
        for _ in 0 ..< 25 {
            guard process.isRunning else { return }
            try? await Task.sleep(nanoseconds: 20_000_000)
        }
    }

    private func installAppleRemindersEventKitAliases(
        serverName: String,
        configuration: VoiceChatMCPConfiguration.Server,
        discoveredByName: [String: Tool],
        client: Client
    ) throws {
        let requiredProviderTools = Set(["reminders_tasks"])
        let missingProviderTools = requiredProviderTools.filter {
            discoveredByName[$0] == nil
        }
        guard missingProviderTools.isEmpty else {
            throw VoiceChatMCPError.invalidConfiguration(
                "server \(serverName) does not expose the pinned EventKit reminder tools: "
                    + missingProviderTools.sorted().joined(separator: ", "))
        }

        for alias in configuration.enabledTools {
            guard routes[alias] == nil else {
                throw VoiceChatMCPError.invalidConfiguration(
                    "duplicate tool name across MCP servers: \(alias)")
            }
            let providerToolName: String
            providerToolName = "reminders_tasks"
            let definition = try Self.appleEventKitAliasTool(
                serverName: serverName,
                alias: alias,
                readOnly: configuration.readOnlyTools.contains(alias))
            routes[alias] = ToolRoute(
                tool: definition,
                client: client,
                providerToolName: providerToolName,
                adapter: .appleRemindersEventKit)
        }
    }

    static func appleEventKitProviderCall(
        toolName: String,
        argumentsJSON: String,
        referenceDate: Date = Date(),
        calendar: Calendar = .current
    ) throws -> VoiceChatFunctionCall {
        guard let data = argumentsJSON.data(using: .utf8),
              let arguments = try JSONSerialization.jsonObject(with: data)
                as? [String: Any] else {
            throw VoiceChatMCPError.toolCall(
                "arguments for \(toolName) must be a JSON object")
        }

        let providerName: String
        let providerArguments: [String: Any]
        switch toolName {
        case "list_reminders":
            providerName = "reminders_tasks"
            var read: [String: Any] = [
                "action": "read",
                "showCompleted": false,
            ]
            if let search = nonEmptyString(arguments["search"]) {
                read["search"] = search
            }
            providerArguments = read
        case "create_reminder":
            providerName = "reminders_tasks"
            var create: [String: Any] = [
                "action": "create",
                "title": try requiredString(
                    "name", in: arguments, toolName: toolName),
            ]
            if let list = nonEmptyString(arguments["list"]) {
                create["targetList"] = list
            }
            if let body = nonEmptyString(arguments["body"]) {
                create["note"] = body
            }
            if let dueDate = nonEmptyString(arguments["due_date"]) {
                guard let normalized = normalizedEventKitDueDate(
                    dueDate,
                    referenceDate: referenceDate,
                    calendar: calendar)
                else {
                    throw VoiceChatMCPError.clarificationRequired(
                        field: "due_date",
                        message: "state an exact date and time")
                }
                create["dueDate"] = normalized
            }
            if let value = try appleEventKitPriority(
                arguments["priority"], toolName: toolName)
            {
                create["priority"] = value
            }
            providerArguments = create
        case "update_reminder":
            providerName = "reminders_tasks"
            var update: [String: Any] = [
                "action": "update",
                "id": try requiredString("id", in: arguments, toolName: toolName),
            ]
            var hasMutation = false
            if let name = nonEmptyString(arguments["name"]) {
                update["title"] = name
                hasMutation = true
            }
            if let list = nonEmptyString(arguments["list"]) {
                update["targetList"] = list
                hasMutation = true
            }
            if let body = arguments["body"] as? String {
                update["note"] = String(body.prefix(1_000))
                hasMutation = true
            }
            if let dueDate = nonEmptyString(arguments["due_date"]) {
                guard let normalized = normalizedEventKitDueDate(
                    dueDate,
                    referenceDate: referenceDate,
                    calendar: calendar)
                else {
                    throw VoiceChatMCPError.clarificationRequired(
                        field: "due_date",
                        message: "state an exact date and time")
                }
                update["dueDate"] = normalized
                hasMutation = true
            }
            if let value = try appleEventKitPriority(
                arguments["priority"], toolName: toolName)
            {
                update["priority"] = value
                hasMutation = true
            }
            if let completed = arguments["completed"] as? Bool {
                update["completed"] = completed
                hasMutation = true
            }
            guard hasMutation else {
                throw VoiceChatMCPError.toolCall(
                    "update_reminder requires at least one field to change")
            }
            providerArguments = update
        default:
            throw VoiceChatMCPError.toolCall(
                "unsupported Apple Reminders alias: \(toolName)")
        }
        return VoiceChatFunctionCall(
            name: providerName,
            argumentsJSON: try compactASCIIJSON(providerArguments))
    }

    static func appleEventKitProviderArguments(
        toolName: String,
        modelArgumentsJSON: String,
        references: VoiceChatMCPOpaqueReferenceTable
    ) throws -> String {
        guard toolName == "update_reminder" else {
            return modelArgumentsJSON
        }
        guard let data = modelArgumentsJSON.data(using: .utf8),
              var arguments = try JSONSerialization.jsonObject(with: data)
                as? [String: Any] else {
            throw VoiceChatMCPError.toolCall(
                "arguments for update_reminder must be a JSON object")
        }
        guard let reference = nonEmptyString(arguments["id"]) else {
            return modelArgumentsJSON
        }
        if let providerID = references.providerID(
            for: reference)
        {
            arguments["id"] = providerID
        } else if VoiceChatMCPOpaqueReferenceTable.isManagedReference(
            reference)
        {
            throw VoiceChatMCPError.toolCall(
                "update_reminder id is no longer available; list reminders again")
        }
        return try compactASCIIJSON(arguments)
    }

    static func appleEventKitCanonicalResult(
        toolName: String,
        providerText: String
    ) throws -> String {
        switch toolName {
        case "list_reminders":
            let reminders = appleEventKitReminderRecords(providerText)
            if reminders.isEmpty,
               !providerText.localizedCaseInsensitiveContains("no reminder"),
               !providerText.localizedCaseInsensitiveContains("total: 0") {
                throw VoiceChatMCPError.toolCall(
                    "the EventKit provider returned an unexpected list_reminders response")
            }
            return try compactASCIIJSON(Array(reminders.prefix(24)))
        case "create_reminder", "update_reminder":
            return providerText
        default:
            throw VoiceChatMCPError.toolCall(
                "unsupported Apple Reminders alias: \(toolName)")
        }
    }

    /// Convert the provider's bounded Markdown to a compact, flattened record
    /// array. The caller replaces provider IDs with session references so a
    /// subsequent native update call can identify the exact reminder without
    /// runtime name matching.
    private static func appleEventKitReminderRecords(
        _ providerText: String
    ) -> [[String: Any]] {
        var records: [[String: Any]] = []
        var current: [String: Any]?

        func flush() {
            if let current, current["name"] != nil { records.append(current) }
            current = nil
        }

        for rawLine in providerText.split(
            separator: "\n", omittingEmptySubsequences: false
        ) {
            let line = String(rawLine)
            if line.hasPrefix("- [ ] ") || line.hasPrefix("- [x] ") {
                flush()
                var title = String(line.dropFirst(6))
                // The provider may append known emoji metadata markers to the
                // title. Do not use a generic non-ASCII boundary here because
                // reminder names themselves may contain accented or non-Latin
                // text.
                let metadataMarkers = [" 🔄", " 📍", " 🏷", " 📋"]
                if let marker = metadataMarkers.compactMap({
                    title.range(of: $0)?.lowerBound
                }).min() {
                    title = String(title[..<marker])
                }
                guard let name = boundedProviderName(title) else { continue }
                current = [
                    "name": name,
                ]
                continue
            }
            guard current != nil else { continue }
            let fields: [(prefix: String, key: String)] = [
                ("  - List: ", "list"),
                ("  - ID: ", "id"),
                ("  - Due: ", "due_date"),
            ]
            for field in fields where line.hasPrefix(field.prefix) {
                let value = String(line.dropFirst(field.prefix.count))
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if !value.isEmpty {
                    current?[field.key] = String(value.prefix(160))
                }
                break
            }
        }
        flush()
        return records
    }

    private static func requiredString(
        _ key: String,
        in arguments: [String: Any],
        toolName: String
    ) throws -> String {
        guard let value = nonEmptyString(arguments[key]) else {
            throw VoiceChatMCPError.toolCall(
                "\(toolName) requires a non-empty \(key)")
        }
        return value
    }

    private static func nonEmptyString(_ value: Any?) -> String? {
        guard let string = value as? String else { return nil }
        let trimmed = string.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    private static func normalizedEventKitDueDate(
        _ value: String,
        referenceDate: Date,
        calendar: Calendar
    ) -> String? {
        let value = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty else { return nil }

        let output = DateFormatter()
        output.calendar = calendar
        output.locale = Locale(identifier: "en_US_POSIX")
        output.timeZone = calendar.timeZone
        output.dateFormat = "yyyy-MM-dd HH:mm:ss"

        // Models commonly emit RFC 3339 even when a schema shows a local
        // timestamp. Respect the encoded offset and convert it to EventKit's
        // local representation instead of rejecting an otherwise exact time.
        let iso8601 = ISO8601DateFormatter()
        for options: ISO8601DateFormatter.Options in [
            [.withInternetDateTime, .withFractionalSeconds],
            [.withInternetDateTime],
        ] {
            iso8601.formatOptions = options
            if let date = iso8601.date(from: value) {
                return output.string(from: date)
            }
        }

        let formatter = DateFormatter()
        formatter.calendar = calendar
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = calendar.timeZone
        formatter.isLenient = false
        let formats = [
            "MMMM d, yyyy 'at' h:mm a",
            "MMMM d, yyyy 'at' h a",
            "MMMM d, yyyy h:mm a",
            "MMMM d, yyyy h a",
            "MMM d, yyyy 'at' h:mm a",
            "MMM d, yyyy 'at' h a",
            "MMM d, yyyy h:mm a",
            "MMM d, yyyy h a",
            "yyyy-MM-dd h:mm a",
            "yyyy-MM-dd h a",
            "yyyy-MM-dd HH:mm",
            "yyyy-MM-dd HH:mm:ss",
            "yyyy-MM-dd'T'HH:mm:ss",
            "yyyy-MM-dd'T'HH:mm:ss.SSS",
            "yyyy-MM-dd HH:mm:ss ZZZZZ",
            "yyyy-MM-dd",
        ]
        for format in formats {
            formatter.dateFormat = format
            guard let date = formatter.date(from: value) else { continue }
            if format == "yyyy-MM-dd" {
                let dateOnly = DateFormatter()
                dateOnly.calendar = calendar
                dateOnly.locale = Locale(identifier: "en_US_POSIX")
                dateOnly.timeZone = calendar.timeZone
                dateOnly.dateFormat = "yyyy-MM-dd"
                return dateOnly.string(from: date)
            }
            return output.string(from: date)
        }

        // Relative dates are accepted only when the model supplied an exact
        // clock time. This is argument normalization, not transcript parsing:
        // ambiguous values such as "tomorrow morning" still require the model
        // to ask the user rather than guessing an hour.
        let lowercased = value.lowercased()
        let relativePrefixes: [(prefix: String, dayOffset: Int)] = [
            ("tomorrow at ", 1),
            ("tomorrow ", 1),
            ("today at ", 0),
            ("today ", 0),
        ]
        for relative in relativePrefixes where lowercased.hasPrefix(
            relative.prefix)
        {
            let clock = String(value.dropFirst(relative.prefix.count))
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let clockFormatter = DateFormatter()
            clockFormatter.calendar = calendar
            clockFormatter.locale = Locale(identifier: "en_US_POSIX")
            clockFormatter.timeZone = calendar.timeZone
            clockFormatter.isLenient = false
            for clockFormat in ["h:mm a", "h a", "HH:mm", "HH:mm:ss"] {
                clockFormatter.dateFormat = clockFormat
                guard let parsedClock = clockFormatter.date(from: clock) else {
                    continue
                }
                let components = calendar.dateComponents(
                    [.hour, .minute, .second], from: parsedClock)
                guard let day = calendar.date(
                    byAdding: .day,
                    value: relative.dayOffset,
                    to: calendar.startOfDay(for: referenceDate)),
                    let date = calendar.date(
                        bySettingHour: components.hour ?? 0,
                        minute: components.minute ?? 0,
                        second: components.second ?? 0,
                        of: day)
                else { return nil }
                return output.string(from: date)
            }
        }
        return nil
    }

    private static func appleEventKitPriority(
        _ rawValue: Any?,
        toolName: String
    ) throws -> Int? {
        guard let rawValue else { return nil }
        guard let number = rawValue as? NSNumber,
              CFGetTypeID(number) != CFBooleanGetTypeID(),
              number.doubleValue.isFinite,
              number.doubleValue.rounded() == number.doubleValue else {
            throw VoiceChatMCPError.toolCall(
                "\(toolName) priority must be an integer")
        }
        let value = number.intValue
        guard [0, 1, 5, 9].contains(value) else {
            throw VoiceChatMCPError.toolCall(
                "\(toolName) priority must be 0, 1, 5, or 9")
        }
        return value
    }

    private static func boundedProviderName(_ value: String) -> String? {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        return String(trimmed.prefix(120))
    }

    static func restrictedProcessEnvironment(
        parent: [String: String],
        configured: [String: String]
    ) -> [String: String] {
        let inheritedKeys = [
            "PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE",
        ]
        var environment: [String: String] = [:]
        for key in inheritedKeys {
            if let value = parent[key] { environment[key] = value }
        }
        if environment["PATH"] == nil {
            environment["PATH"] = "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
        }
        environment.merge(configured) { _, value in value }
        return environment
    }
}

struct VoiceChatFunctionCall: Sendable, Equatable {
    let name: String
    let argumentsJSON: String

    var fingerprint: String { name + "\n" + argumentsJSON }

    static func parse(_ raw: String) throws -> Self {
        var payload = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if let start = payload.range(of: "<TOOLCALL>"),
           let end = payload.range(
               of: "</TOOLCALL>", range: start.upperBound ..< payload.endIndex)
        {
            payload = String(payload[start.upperBound ..< end.lowerBound])
        }
        guard let data = payload.data(using: .utf8) else {
            throw VoiceChatMCPError.toolCall("payload is not UTF-8")
        }
        let root = try JSONSerialization.jsonObject(with: data)
        let calls: [[String: Any]]
        if let array = root as? [[String: Any]] {
            calls = array
        } else if let object = root as? [String: Any] {
            calls = [object]
        } else {
            throw VoiceChatMCPError.toolCall(
                "payload must be one JSON call object")
        }
        guard calls.count == 1,
              let name = calls[0]["name"] as? String,
              !name.isEmpty else {
            throw VoiceChatMCPError.toolCall(
                "exactly one named tool call is supported")
        }

        var arguments: Any = calls[0]["arguments"] ?? [String: Any]()
        if let string = arguments as? String,
           let stringData = string.data(using: .utf8),
           let decoded = try? JSONSerialization.jsonObject(with: stringData) {
            arguments = decoded
        }
        guard arguments is [String: Any] else {
            throw VoiceChatMCPError.toolCall(
                "arguments for \(name) must be a JSON object")
        }
        let argumentsJSON = try compactASCIIJSON(arguments)
        return Self(name: name, argumentsJSON: argumentsJSON)
    }
}

struct VoiceChatMCPAction: Sendable, Equatable {
    let responseJSON: String
    let requireAssistantReplyBeforeNextFunctionCall: Bool

    init(
        responseJSON: String,
        requireAssistantReplyBeforeNextFunctionCall: Bool = false
    ) {
        self.responseJSON = responseJSON
        self.requireAssistantReplyBeforeNextFunctionCall =
            requireAssistantReplyBeforeNextFunctionCall
    }
}

struct VoiceChatMCPToolActivity: Sendable, Equatable {
    enum State: String, Sendable, Equatable {
        case running
        case completed
        case needsInput = "needs input"
        case failed
    }

    let name: String
    let state: State
    let elapsedMilliseconds: Double

    init(
        name: String,
        state: State,
        elapsedMilliseconds: Double = 0
    ) {
        self.name = name
        self.state = state
        self.elapsedMilliseconds = elapsedMilliseconds
    }
}

struct VoiceChatMCPToolRuntimeStatus: Sendable, Equatable {
    let executing: Bool
    let name: String?
    let activity: VoiceChatMCPToolActivity?
}

actor VoiceChatMCPToolCoordinator {
    private enum PendingWriteStage: Sendable, Equatable {
        case awaitingConfirmationPrompt
        case confirmationPromptActive
        case awaitingModelDecision
    }

    private struct PendingWrite: Sendable {
        let call: VoiceChatFunctionCall
        var stage: PendingWriteStage
        var observedUserSpeech: Bool
    }

    private let executor: any VoiceChatMCPToolExecuting
    private let writePolicy: VoiceChatMCPWritePolicy
    private var activeToolExecutions = 0
    private var activeToolName: String?
    private var activeToolStartedAtNanoseconds: UInt64?
    private var lastToolActivity: VoiceChatMCPToolActivity?
    private var pendingWrite: PendingWrite?
    private var lastExecutedWriteFingerprint: String?
    private var freshUserSpeechSinceLastWrite = true

    init(
        executor: any VoiceChatMCPToolExecuting,
        writePolicy: VoiceChatMCPWritePolicy
    ) {
        self.executor = executor
        self.writePolicy = writePolicy
    }

    /// Handle only calls emitted by VoiceChat's native function channel.
    ///
    /// The coordinator deliberately does not inspect the user transcript,
    /// classify intent, infer arguments, repair dates, or match reminder names.
    /// Its responsibilities are protocol parsing, enabled-tool enforcement,
    /// write authorization, execution, and result delivery.
    func handleFunctionCall(
        _ rawCall: String
    ) async -> VoiceChatMCPAction {
        do {
            let call = try VoiceChatFunctionCall.parse(rawCall)
            guard let tool = await executor.tool(named: call.name) else {
                return responseAction([
                    "ok": false,
                    "status": "tool_not_enabled",
                    "tool": call.name,
                ])
            }
            if let validationError = Self.validate(
                call: call, against: tool.inputSchemaJSON)
            {
                return responseAction([
                    "ok": false,
                    "status": "invalid_arguments",
                    "tool": call.name,
                    "error": validationError,
                ])
            }

            if tool.access == .read {
                return executionAction(for: await execute(call))
            }

            if lastExecutedWriteFingerprint == call.fingerprint,
               !freshUserSpeechSinceLastWrite
            {
                return responseAction([
                    "ok": false,
                    "status": "duplicate_suppressed",
                    "tool": call.name,
                ])
            }

            switch writePolicy {
            case .deny:
                pendingWrite = nil
                return responseAction([
                    "ok": false,
                    "status": "write_denied",
                    "tool": call.name,
                ])
            case .allow:
                pendingWrite = nil
                return await executeWrite(call)
            case .confirm:
                if let pendingWrite,
                   pendingWrite.call.fingerprint == call.fingerprint {
                    if pendingWrite.stage == .awaitingModelDecision,
                       pendingWrite.observedUserSpeech {
                        self.pendingWrite = nil
                        return await executeWrite(call)
                    }
                    return confirmationRequiredAction(for: call)
                }
                pendingWrite = PendingWrite(
                    call: call,
                    stage: .awaitingConfirmationPrompt,
                    observedUserSpeech: false)
                return confirmationRequiredAction(for: call)
            }
        } catch {
            return responseAction([
                "ok": false,
                "status": "invalid_tool_call",
                "error": String(describing: error),
            ])
        }
    }

    /// A confirmation is a model decision, not transcript keyword matching.
    /// The first complete text turn after the pending result is the model's
    /// confirmation question. A later text turn means the model chose not to
    /// repeat the identical call, so the pending write expires.
    func observeModelTextTurnStart() {
        guard var pendingWrite else { return }
        switch pendingWrite.stage {
        case .awaitingConfirmationPrompt:
            pendingWrite.stage = .confirmationPromptActive
            self.pendingWrite = pendingWrite
        case .confirmationPromptActive:
            break
        case .awaitingModelDecision:
            self.pendingWrite = nil
        }
    }

    /// Only speech captured after the confirmation question has finished may
    /// authorize a repeated native write call. Without this boundary, acoustic
    /// echo from the assistant's own prompt can satisfy the activity gate.
    func observeModelTextTurnEnd() {
        guard var pendingWrite,
              pendingWrite.stage == .confirmationPromptActive else { return }
        pendingWrite.stage = .awaitingModelDecision
        pendingWrite.observedUserSpeech = false
        self.pendingWrite = pendingWrite
    }

    /// Track only whether fresh acoustic activity occurred after a confirmation
    /// prompt or completed write. No words or phrases are inspected.
    func observeUserActivity(rnntIsBlank: Bool?) {
        if rnntIsBlank == false {
            freshUserSpeechSinceLastWrite = true
            if var pendingWrite,
               pendingWrite.stage == .awaitingModelDecision {
                pendingWrite.observedUserSpeech = true
                self.pendingWrite = pendingWrite
            }
        }
    }

    func isExecutingTool() -> Bool {
        activeToolExecutions > 0
    }

    func executingToolName() -> String? {
        activeToolName
    }

    func toolActivity() -> VoiceChatMCPToolActivity? {
        if let activity = lastToolActivity,
           activity.state == .running,
           let started = activeToolStartedAtNanoseconds {
            return VoiceChatMCPToolActivity(
                name: activity.name,
                state: .running,
                elapsedMilliseconds: Self.elapsedMilliseconds(since: started))
        }
        return lastToolActivity
    }

    func runtimeStatus() -> VoiceChatMCPToolRuntimeStatus {
        VoiceChatMCPToolRuntimeStatus(
            executing: activeToolExecutions > 0,
            name: activeToolName,
            activity: toolActivity())
    }

    private func executeWrite(
        _ call: VoiceChatFunctionCall
    ) async -> VoiceChatMCPAction {
        let response = await execute(call)
        if Self.responseSucceeded(response) {
            lastExecutedWriteFingerprint = call.fingerprint
            freshUserSpeechSinceLastWrite = false
        }
        return executionAction(for: response)
    }

    private func execute(_ call: VoiceChatFunctionCall) async -> String {
        let started = DispatchTime.now().uptimeNanoseconds
        activeToolExecutions += 1
        activeToolName = call.name
        activeToolStartedAtNanoseconds = started
        lastToolActivity = VoiceChatMCPToolActivity(
            name: call.name,
            state: .running)
        defer {
            activeToolExecutions -= 1
            if activeToolExecutions == 0 {
                activeToolName = nil
                activeToolStartedAtNanoseconds = nil
            }
        }
        do {
            let response = try await executor.callTool(
                name: call.name,
                argumentsJSON: call.argumentsJSON)
            lastToolActivity = VoiceChatMCPToolActivity(
                name: call.name,
                state: Self.responseSucceeded(response)
                    ? .completed
                    : .failed,
                elapsedMilliseconds: Self.elapsedMilliseconds(since: started))
            return response
        } catch {
            let clarification: (field: String, message: String)?
            if let mcpError = error as? VoiceChatMCPError,
               case .clarificationRequired(let field, let message) = mcpError
            {
                clarification = (field, message)
            } else {
                clarification = nil
            }
            lastToolActivity = VoiceChatMCPToolActivity(
                name: call.name,
                state: clarification == nil ? .failed : .needsInput,
                elapsedMilliseconds: Self.elapsedMilliseconds(since: started))
            if let clarification {
                return (try? compactASCIIJSON([
                    "ok": false,
                    "tool": call.name,
                    "clarification_required": clarification.field,
                    "error": clarification.message,
                ])) ?? #"{"ok":false,"error":"more information required"}"#
            }
            return (try? compactASCIIJSON([
                "ok": false,
                "tool": call.name,
                "error": Self.modelVisibleError(error),
            ])) ?? #"{"ok":false,"error":"tool failed"}"#
        }
    }

    /// Keep model-facing failures actionable but short. Provider/debug detail
    /// belongs in host logs; replaying it through the trained function channel
    /// costs one 11B + EAR-TTS cache position per token and can expose details
    /// that are irrelevant to the spoken recovery.
    private static func modelVisibleError(_ error: Error) -> String {
        if let error = error as? VoiceChatMCPError {
            switch error {
            case .toolCall(let message):
                return String(message.prefix(120))
            case .clarificationRequired(_, let message):
                return String(message.prefix(120))
            case .timedOut:
                return "timed out"
            case .serverFailed:
                return "provider failed"
            case .invalidConfiguration:
                return "configuration error"
            }
        }
        return "tool failed"
    }

    private func responseAction(
        _ object: [String: Any]
    ) -> VoiceChatMCPAction {
        let response = (try? compactASCIIJSON(object))
            ?? #"{"ok":false,"executed":false,"status":"serialization_failed"}"#
        return VoiceChatMCPAction(
            responseJSON: response,
            requireAssistantReplyBeforeNextFunctionCall: true)
    }

    /// A provider or protocol failure must become a spoken assistant turn
    /// before the native function head can retry. Otherwise the checkpoint can
    /// immediately emit the same call again, creating an invisible tool loop
    /// that leaves the live session waiting forever. This gate observes only
    /// the structured result and model BOS token; it does not inspect speech.
    private func executionAction(for response: String) -> VoiceChatMCPAction {
        VoiceChatMCPAction(
            responseJSON: response,
            requireAssistantReplyBeforeNextFunctionCall:
                !Self.responseSucceeded(response))
    }

    private func confirmationRequiredAction(
        for call: VoiceChatFunctionCall
    ) -> VoiceChatMCPAction {
        let response = (try? compactASCIIJSON([
            "ok": false,
            "executed": false,
            "confirmation_required": true,
            "status": "awaiting_user_confirmation",
            "tool": call.name,
        ])) ?? #"{"ok":false,"executed":false,"status":"serialization_failed"}"#
        return VoiceChatMCPAction(
            responseJSON: response,
            requireAssistantReplyBeforeNextFunctionCall: true)
    }

    private static func responseSucceeded(_ responseJSON: String) -> Bool {
        guard let data = responseJSON.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data)
                as? [String: Any] else { return false }
        return object["ok"] as? Bool == true
    }

    /// Validate the small JSON-Schema subset used by MCP tool definitions.
    /// This is structural validation only; it never compares arguments with
    /// transcript text or infers values.
    private static func validate(
        call: VoiceChatFunctionCall,
        against schemaJSON: String
    ) -> String? {
        guard let argumentsData = call.argumentsJSON.data(using: .utf8),
              let arguments = try? JSONSerialization.jsonObject(
                with: argumentsData) as? [String: Any] else {
            return "arguments must be a JSON object"
        }
        guard let schemaData = schemaJSON.data(using: .utf8),
              let schema = try? JSONSerialization.jsonObject(
                with: schemaData) as? [String: Any] else {
            return "tool schema is invalid"
        }
        let properties = schema["properties"] as? [String: Any] ?? [:]
        let required = schema["required"] as? [String] ?? []
        for key in required {
            guard let value = arguments[key], !(value is NSNull) else {
                return "missing required argument: \(key)"
            }
            if let string = value as? String,
               string.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return "required argument is empty: \(key)"
            }
        }
        for (key, value) in arguments {
            guard let property = properties[key] as? [String: Any],
                  let type = property["type"] as? String else { continue }
            let valid: Bool
            switch type {
            case "string":
                valid = value is String
            case "boolean":
                valid = value is Bool
            case "number":
                valid = value is NSNumber
                    && CFGetTypeID(value as! NSNumber) != CFBooleanGetTypeID()
            case "integer":
                if let number = value as? NSNumber,
                   CFGetTypeID(number) != CFBooleanGetTypeID() {
                    valid = number.doubleValue.rounded() == number.doubleValue
                } else {
                    valid = false
                }
            case "array":
                valid = value is [Any]
            case "object":
                valid = value is [String: Any]
            default:
                valid = true
            }
            if !valid {
                return "argument \(key) must be \(type)"
            }
        }
        return nil
    }

    private static func elapsedMilliseconds(since started: UInt64) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000
    }
}

private extension JSONEncoder {
    static var sorted: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return encoder
    }
}

private func compactASCIIJSON(_ object: Any) throws -> String {
    let data = try JSONSerialization.data(
        withJSONObject: object,
        options: [.sortedKeys, .withoutEscapingSlashes])
    return asciiEscapedJSON(String(decoding: data, as: UTF8.self))
}

private func asciiSanitized(
    _ input: String,
    maximumCharacters: Int
) -> String {
    let bounded = input.prefix(maximumCharacters)
    return String(bounded.unicodeScalars.map { scalar in
        if scalar.isASCII, scalar.value >= 0x20 { return Character(String(scalar)) }
        return "?"
    })
}

private func asciiEscapedJSON(_ input: String) -> String {
    var output = ""
    output.reserveCapacity(input.utf8.count)
    for scalar in input.unicodeScalars {
        if scalar.value == 0x3C {
            output += "\\u003C"
        } else if scalar.value == 0x3E {
            output += "\\u003E"
        } else if scalar.value == 0x26 {
            output += "\\u0026"
        } else if scalar.isASCII {
            output.unicodeScalars.append(scalar)
        } else if scalar.value <= 0xFFFF {
            output += String(format: "\\u%04X", scalar.value)
        } else {
            let value = scalar.value - 0x10000
            let high = 0xD800 + (value >> 10)
            let low = 0xDC00 + (value & 0x3FF)
            output += String(format: "\\u%04X\\u%04X", high, low)
        }
    }
    return output
}

private final class VoiceChatMCPTimeoutResolution<T>: @unchecked Sendable {
    private enum Source {
        case operation
        case timeout
        case cancellation
    }

    private let lock = NSLock()
    private var continuation: CheckedContinuation<T, Error>?
    private var pendingResult: Result<T, Error>?
    private var operationTask: Task<Void, Never>?
    private var timeoutTask: Task<Void, Never>?
    private var resolved = false

    func install(_ continuation: CheckedContinuation<T, Error>) {
        lock.lock()
        if let pendingResult {
            lock.unlock()
            continuation.resume(with: pendingResult)
            return
        }
        self.continuation = continuation
        lock.unlock()
    }

    func installTasks(
        operation: Task<Void, Never>,
        timeout: Task<Void, Never>
    ) {
        lock.lock()
        if resolved {
            lock.unlock()
            operation.cancel()
            timeout.cancel()
            return
        }
        operationTask = operation
        timeoutTask = timeout
        lock.unlock()
    }

    func resolveFromOperation(_ result: Result<T, Error>) {
        resolve(result, source: .operation)
    }

    func resolveFromTimeout(operation: String) {
        resolve(
            .failure(VoiceChatMCPError.timedOut(operation)),
            source: .timeout)
    }

    func cancel() {
        resolve(.failure(CancellationError()), source: .cancellation)
    }

    private func resolve(_ result: Result<T, Error>, source: Source) {
        lock.lock()
        guard !resolved else {
            lock.unlock()
            return
        }
        resolved = true
        let tasksToCancel: [Task<Void, Never>]
        switch source {
        case .operation:
            tasksToCancel = [timeoutTask].compactMap { $0 }
        case .timeout:
            tasksToCancel = [operationTask].compactMap { $0 }
        case .cancellation:
            tasksToCancel = [operationTask, timeoutTask].compactMap { $0 }
        }
        operationTask = nil
        timeoutTask = nil
        if let continuation {
            self.continuation = nil
            lock.unlock()
            tasksToCancel.forEach { $0.cancel() }
            continuation.resume(with: result)
        } else {
            pendingResult = result
            lock.unlock()
            tasksToCancel.forEach { $0.cancel() }
        }
    }
}

func withMCPTimeout<T: Sendable>(
    seconds: Double,
    operation: String,
    body: @escaping @Sendable () async throws -> T
) async throws -> T {
    let resolution = VoiceChatMCPTimeoutResolution<T>()
    return try await withTaskCancellationHandler {
        try await withCheckedThrowingContinuation { continuation in
            resolution.install(continuation)
            let operationTask = Task {
                do {
                    resolution.resolveFromOperation(
                        .success(try await body()))
                } catch {
                    resolution.resolveFromOperation(.failure(error))
                }
            }
            let timeoutTask = Task {
                do {
                    try await Task.sleep(
                        nanoseconds: UInt64(seconds * 1_000_000_000))
                    resolution.resolveFromTimeout(operation: operation)
                } catch {
                    // The winning operation or caller canceled this timer.
                }
            }
            resolution.installTasks(
                operation: operationTask,
                timeout: timeoutTask)
        }
    } onCancel: {
        resolution.cancel()
    }
}
