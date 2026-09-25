import Foundation

/// A JSON Schema compiled into the node graph the byte matcher walks.
///
/// Nodes live in one flat array and refer to each other by index, so matcher frames are plain
/// values (cheap to copy, hashable) and the "any JSON value" node can refer to itself.
final class JSONSchemaGrammar: @unchecked Sendable {
    enum Node {
        /// Canonical encodings of the admissible values (`enum`, `const`, booleans, `null`).
        case literals([[UInt8]])
        case string(StringSpec)
        case integer(min: Int64?, max: Int64?)
        case number
        case array(items: Int32, minItems: Int, maxItems: Int?)
        case object(ObjectSpec)
        /// Any of these nodes (`anyOf`, a `type` list, `nullable`, an unconstrained value).
        case union([Int32])
    }

    struct StringSpec {
        var minLength: Int
        var maxLength: Int?
        var pattern: JSONPattern?
    }

    struct Property {
        /// The key as JSON string content (escaped, without quotes).
        let key: [UInt8]
        let node: Int32
        let required: Bool
    }

    struct ObjectSpec {
        /// Declared properties in writing order: `required` order first, then the rest in schema
        /// order.
        var properties: [Property]
        var minProperties: Int
        var maxProperties: Int?
        /// Value schema for arbitrary keys; set only on an object with no declared properties.
        var freeValues: Int32?
        /// `requiredFrom[i]` = number of required properties at index `i` or later.
        var requiredFrom: [Int]
    }

    private(set) var nodes: [Node] = []
    private(set) var root: Int32 = 0
    /// A free JSON string, used for arbitrary object keys.
    private(set) var freeString: Int32 = 0
    private var anyNode: Int32?

    init(schema text: String) throws {
        let json = try OrderedJSON.parse(text)
        freeString = add(.string(StringSpec(minLength: 0, maxLength: nil, pattern: nil)))
        root = try compile(json, path: "#")
    }

    private func add(_ node: Node) -> Int32 {
        nodes.append(node)
        return Int32(nodes.count - 1)
    }

    /// Any JSON value: a free object, array, string, number, boolean or null.
    private func any() -> Int32 {
        if let anyNode { return anyNode }
        let index = add(.union([]))
        anyNode = index
        let object = add(.object(ObjectSpec(properties: [], minProperties: 0, maxProperties: nil,
                                            freeValues: index, requiredFrom: [0])))
        let array = add(.array(items: index, minItems: 0, maxItems: nil))
        let scalars = add(.literals([Array("true".utf8), Array("false".utf8), Array("null".utf8)]))
        nodes[Int(index)] = .union([object, array, freeString, add(.number), scalars])
        return index
    }

    private static let annotations: Set<String> = [
        "title", "description", "default", "examples", "$schema", "$id", "$comment",
        "deprecated", "readOnly", "writeOnly",
    ]
    private static let handled: Set<String> = [
        "type", "enum", "const", "properties", "required", "additionalProperties", "items",
        "minItems", "maxItems", "uniqueItems", "minLength", "maxLength", "pattern", "minimum",
        "maximum", "exclusiveMinimum", "exclusiveMaximum", "anyOf", "nullable", "minProperties",
        "maxProperties",
    ]

    private func compile(_ schema: OrderedJSON, path: String) throws -> Int32 {
        switch schema {
        case .bool(true): return any()
        case .bool(false): throw ChatResponseFormatError.unsupported(keyword: "false", path: path)
        case .object: break
        default: throw ChatResponseFormatError.invalidSchema("schema at \(path) is not an object")
        }
        guard case .object(let members) = schema else { fatalError("unreachable") }
        for m in members where !Self.annotations.contains(m.key) && !Self.handled.contains(m.key) {
            throw ChatResponseFormatError.unsupported(keyword: m.key, path: path)
        }
        if case .bool(true)? = schema["uniqueItems"] {
            throw ChatResponseFormatError.unsupported(keyword: "uniqueItems", path: path)
        }

        let node: Int32
        if let anyOf = schema["anyOf"] {
            let constraints = members.filter {
                !Self.annotations.contains($0.key) && $0.key != "anyOf" && $0.key != "nullable"
            }
            if let other = constraints.first {
                // Sibling constraints would have to hold in every branch at once.
                throw ChatResponseFormatError.unsupported(keyword: "\(other.key) beside anyOf", path: path)
            }
            guard case .array(let branches) = anyOf, !branches.isEmpty else {
                throw ChatResponseFormatError.invalidSchema("anyOf at \(path) must be a non-empty array")
            }
            var compiled: [Int32] = []
            for (i, b) in branches.enumerated() {
                compiled.append(try compile(b, path: "\(path)/anyOf/\(i)"))
            }
            node = add(.union(compiled))
        } else if let values = try enumValues(schema, path: path) {
            let types = try typeNames(schema, path: path)
            for v in values where types != nil {
                let t = v.typeName
                guard types!.contains(t) || (t == "integer" && types!.contains("number")) else {
                    throw ChatResponseFormatError.invalidSchema(
                        "enum value \(String(decoding: v.canonicalBytes, as: UTF8.self)) at \(path) does not match its type")
                }
            }
            node = add(.literals(values.map(\.canonicalBytes)))
        } else {
            var types = try typeNames(schema, path: path) ?? inferredTypes(schema)
            if types.isEmpty { types = ["any"] }
            var compiled: [Int32] = []
            for t in types { compiled.append(try compileType(t, schema, path: path)) }
            node = compiled.count == 1 ? compiled[0] : add(.union(compiled))
        }

        if case .bool(true)? = schema["nullable"] {
            return add(.union([node, add(.literals([Array("null".utf8)]))]))
        }
        return node
    }

    private func enumValues(_ schema: OrderedJSON, path: String) throws -> [OrderedJSON]? {
        if let c = schema["const"] { return [c] }
        guard let e = schema["enum"] else { return nil }
        guard case .array(let values) = e, !values.isEmpty else {
            throw ChatResponseFormatError.invalidSchema("enum at \(path) must be a non-empty array")
        }
        guard values.count <= 64 else {
            throw ChatResponseFormatError.unsupported(keyword: "enum (more than 64 values)", path: path)
        }
        return values
    }

    private func typeNames(_ schema: OrderedJSON, path: String) throws -> [String]? {
        switch schema["type"] {
        case nil: return nil
        case .string(let t)?: return [t]
        case .array(let items)?:
            return try items.map {
                guard case .string(let t) = $0 else {
                    throw ChatResponseFormatError.invalidSchema("type list at \(path) must hold names")
                }
                return t
            }
        default: throw ChatResponseFormatError.invalidSchema("type at \(path) must be a name or a list")
        }
    }

    private func inferredTypes(_ schema: OrderedJSON) -> [String] {
        func has(_ keys: [String]) -> Bool { keys.contains { schema[$0] != nil } }
        var types: [String] = []
        if has(["properties", "required", "additionalProperties", "minProperties", "maxProperties"]) {
            types.append("object")
        }
        if has(["items", "minItems", "maxItems"]) { types.append("array") }
        if has(["minLength", "maxLength", "pattern"]) { types.append("string") }
        if has(["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"]) { types.append("number") }
        return types
    }

    private func nonNegativeInt(_ schema: OrderedJSON, _ key: String, path: String) throws -> Int? {
        guard let v = schema[key] else { return nil }
        guard case .number(_, let d) = v, d >= 0, d.rounded() == d, d < 1e9 else {
            throw ChatResponseFormatError.invalidSchema("\(key) at \(path) must be a non-negative integer")
        }
        return Int(d)
    }

    private func compileType(_ type: String, _ schema: OrderedJSON, path: String) throws -> Int32 {
        switch type {
        case "any": return any()
        case "null": return add(.literals([Array("null".utf8)]))
        case "boolean": return add(.literals([Array("true".utf8), Array("false".utf8)]))
        case "string":
            let minLength = try nonNegativeInt(schema, "minLength", path: path) ?? 0
            let maxLength = try nonNegativeInt(schema, "maxLength", path: path)
            if let maxLength, maxLength < minLength {
                throw ChatResponseFormatError.invalidSchema("maxLength < minLength at \(path)")
            }
            var pattern: JSONPattern?
            if let p = schema["pattern"] {
                guard case .string(let source) = p else {
                    throw ChatResponseFormatError.invalidSchema("pattern at \(path) must be a string")
                }
                pattern = try JSONPattern.compile(source, path: path)
            }
            return add(.string(StringSpec(minLength: minLength, maxLength: maxLength, pattern: pattern)))
        case "integer":
            func bound(_ key: String) throws -> Int64? {
                guard let v = schema[key] else { return nil }
                guard case .number(_, let d) = v, abs(d) < 1e18 else {
                    throw ChatResponseFormatError.invalidSchema("\(key) at \(path) must be a number")
                }
                switch key {
                case "minimum": return Int64(d.rounded(.up))
                case "maximum": return Int64(d.rounded(.down))
                case "exclusiveMinimum": return Int64(d.rounded(.down)) + 1
                default: return Int64(d.rounded(.up)) - 1
                }
            }
            let lows = try ["minimum", "exclusiveMinimum"].compactMap { try bound($0) }
            let highs = try ["maximum", "exclusiveMaximum"].compactMap { try bound($0) }
            let lo = lows.max(), hi = highs.min()
            if let lo, let hi, lo > hi {
                throw ChatResponseFormatError.invalidSchema("integer range at \(path) is empty")
            }
            return add(.integer(min: lo, max: hi))
        case "number":
            for key in ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"] where schema[key] != nil {
                throw ChatResponseFormatError.unsupported(keyword: "\(key) on a number", path: path)
            }
            return add(.number)
        case "array":
            let minItems = try nonNegativeInt(schema, "minItems", path: path) ?? 0
            let maxItems = try nonNegativeInt(schema, "maxItems", path: path)
            if let maxItems, maxItems < minItems {
                throw ChatResponseFormatError.invalidSchema("maxItems < minItems at \(path)")
            }
            let items: Int32
            switch schema["items"] {
            case nil: items = any()
            case .array?: throw ChatResponseFormatError.unsupported(keyword: "items (tuple form)", path: path)
            case let s?: items = try compile(s, path: "\(path)/items")
            }
            return add(.array(items: items, minItems: minItems, maxItems: maxItems))
        case "object":
            return try compileObject(schema, path: path)
        default:
            throw ChatResponseFormatError.invalidSchema("unknown type '\(type)' at \(path)")
        }
    }

    private func compileObject(_ schema: OrderedJSON, path: String) throws -> Int32 {
        var declared: [(name: String, schema: OrderedJSON)] = []
        switch schema["properties"] {
        case nil: break
        case .object(let members)?: declared = members.map { ($0.key, $0.value) }
        default: throw ChatResponseFormatError.invalidSchema("properties at \(path) must be an object")
        }
        var required: [String] = []
        switch schema["required"] {
        case nil: break
        case .array(let names)?:
            for n in names {
                guard case .string(let s) = n else {
                    throw ChatResponseFormatError.invalidSchema("required at \(path) must list names")
                }
                if !required.contains(s) { required.append(s) }
            }
        default: throw ChatResponseFormatError.invalidSchema("required at \(path) must be an array")
        }
        for name in required where !declared.contains(where: { $0.name == name }) {
            throw ChatResponseFormatError.unsupported(
                keyword: "required property '\(name)' without a schema in properties", path: path)
        }
        let minProperties = try nonNegativeInt(schema, "minProperties", path: path) ?? 0
        let maxProperties = try nonNegativeInt(schema, "maxProperties", path: path)

        var freeValues: Int32?
        switch schema["additionalProperties"] {
        case nil, .bool(true)?:
            // Only declared properties are written, which every such schema accepts; an object
            // with none declared takes arbitrary keys.
            if declared.isEmpty { freeValues = any() }
        case .bool(false)?: break
        case let s?:
            guard declared.isEmpty else {
                throw ChatResponseFormatError.unsupported(
                    keyword: "additionalProperties schema beside properties", path: path)
            }
            freeValues = try compile(s, path: "\(path)/additionalProperties")
        }

        let ordered = required.compactMap { name in declared.first { $0.name == name } }
            + declared.filter { !required.contains($0.name) }
        var properties: [Property] = []
        for p in ordered {
            let node = try compile(p.schema, path: "\(path)/properties/\(p.name)")
            properties.append(Property(key: JSONEncoding.escape(p.name), node: node,
                                       required: required.contains(p.name)))
        }
        guard properties.count <= 64 else {
            throw ChatResponseFormatError.unsupported(keyword: "properties (more than 64)", path: path)
        }
        if let maxProperties {
            if required.count > maxProperties {
                throw ChatResponseFormatError.invalidSchema("more required properties than maxProperties at \(path)")
            }
            if maxProperties < minProperties {
                throw ChatResponseFormatError.invalidSchema("maxProperties < minProperties at \(path)")
            }
        }
        if freeValues == nil, minProperties > properties.count {
            throw ChatResponseFormatError.invalidSchema("minProperties exceeds the declared properties at \(path)")
        }
        var requiredFrom = Array(repeating: 0, count: properties.count + 1)
        for i in stride(from: properties.count - 1, through: 0, by: -1) {
            requiredFrom[i] = requiredFrom[i + 1] + (properties[i].required ? 1 : 0)
        }
        return add(.object(ObjectSpec(properties: properties, minProperties: minProperties,
                                      maxProperties: maxProperties, freeValues: freeValues,
                                      requiredFrom: requiredFrom)))
    }
}
