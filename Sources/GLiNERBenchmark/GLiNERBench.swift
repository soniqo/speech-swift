import Foundation
import GLiNER
import MLX
import Darwin

struct Case: Codable {
    let text: String
    let action: String?
    let entities: [String:[String]]?
}
struct Row: Codable {
    let text: String
    let task: String
    let milliseconds: [Double]
    let choices: [GLiNERChoice]?
    let entities: [String:[GLiNERSpan]]?
    let correct: Bool
}
struct MemoryPoint: Codable {
    let phase: String
    let residentBytes: UInt64
    let physicalFootprintBytes: UInt64
    let mlxActiveBytes: Int
    let mlxCacheBytes: Int
    let mlxPeakActiveBytes: Int
}
func memoryPoint(_ phase: String) -> MemoryPoint {
    var info = mach_task_basic_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info_data_t>.size / MemoryLayout<integer_t>.size)
    let status = withUnsafeMutablePointer(to: &info) { ptr in ptr.withMemoryRebound(to: integer_t.self,capacity:Int(count)) {
        task_info(mach_task_self_,task_flavor_t(MACH_TASK_BASIC_INFO),$0,&count)
    }}
    return MemoryPoint(phase:phase,residentBytes:status == KERN_SUCCESS ? info.resident_size : 0,physicalFootprintBytes:footprint(),mlxActiveBytes:Memory.activeMemory,mlxCacheBytes:Memory.cacheMemory,mlxPeakActiveBytes:Memory.peakMemory)
}
final class FootprintSampler: @unchecked Sendable {
    private let lock = NSLock()
    private let timer = DispatchSource.makeTimerSource(queue:DispatchQueue(label:"gliner.memory-sample"))
    private var peak: UInt64 = 0
    init() {
        timer.schedule(deadline:.now(),repeating:.milliseconds(20))
        timer.setEventHandler { [weak self] in
            let value = footprint()
            guard let self else { return }
            self.lock.lock(); self.peak = max(self.peak,value); self.lock.unlock()
        }
        timer.resume()
    }
    func stop() -> UInt64 {
        timer.cancel()
        lock.lock(); defer { lock.unlock() }
        return max(peak,footprint())
    }
    deinit { timer.cancel() }
}
struct Report: Codable {
    let loadMilliseconds: Double
    let rows: [Row]
    let firstRequestMilliseconds: [String:Double]
    let processPeakRSSBytes: UInt64
    let sampledPhysicalFootprintBytes: UInt64
    let modelDirectory: String
    let memoryProfile: [MemoryPoint]
    let evaluateLayers: Bool
    let cacheLimitBytes: Int
    let footprintSamplingIntervalMilliseconds: Int
}
func footprint() -> UInt64 {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let result = withUnsafeMutablePointer(to: &info) { ptr in ptr.withMemoryRebound(to: integer_t.self,capacity: Int(count)) {
        task_info(mach_task_self_,task_flavor_t(TASK_VM_INFO),$0,&count)
    }}
    return result == KERN_SUCCESS ? info.phys_footprint : 0
}
@main struct Benchmark {
    static func main() async throws {
        let args = CommandLine.arguments
        if args.contains("--help") {
            print("gliner-bench --model <export-dir> --cases <cases.json> --output <result.json> [--iterations 5] [--cache-limit-mb 64] [--evaluate-layers true]")
            return
        }
        let options = try BenchmarkOptions.parse(Array(args.dropFirst()))
        let directory = options.model, casesPath = options.cases, output = options.output, iterations = options.iterations
        let cases = try JSONDecoder().decode([Case].self,from: Data(contentsOf: URL(fileURLWithPath: casesPath)))
        if let limit = options.cacheLimitMB { Memory.cacheLimit = limit * 1024 * 1024 }
        Memory.peakMemory = 0
        var memoryProfile = [memoryPoint("before_load")]
        let sampler = FootprintSampler()
        var model: GLiNER? = nil
        let start = ContinuousClock.now
        model = try await GLiNER.load(from: URL(fileURLWithPath: directory),evaluateLayers:options.evaluateLayers)
        func ms(_ start: ContinuousClock.Instant) -> Double { let d = start.duration(to: .now).components; return Double(d.seconds)*1000 + Double(d.attoseconds)/1e15 }
        let load = ms(start)
        memoryProfile.append(memoryPoint("after_load"))
        let labels = ["create_reminder","create_calendar_event","send_message","search_notes","set_timer","other"]
        let descriptions = ["person":"Person name or family member mentioned in the command", "time":"Time of day or duration mentioned in the command"]
        var rows = [Row](), peak = footprint()
        var firstRequests = [String:Double]()
        for task in ["routing","extraction"] {
            guard let first = cases.first(where: { task == "routing" ? $0.action != nil : $0.entities != nil }) else { continue }
            for warmup in 0..<5 {
                let t = ContinuousClock.now
                if task == "routing" { _ = try model!.classify(first.text,labels: labels) }
                else { _ = try model!.extractEntities(first.text,labels: ["person","time"],descriptions: descriptions) }
                if warmup == 0 { firstRequests[task] = ms(t); memoryProfile.append(memoryPoint("first_" + task)) }
            }
            memoryProfile.append(memoryPoint("warm_" + task))
            for item in cases where task == "routing" ? item.action != nil : item.entities != nil {
                var timings = [Double](), choices: [GLiNERChoice]?, entities: [String:[GLiNERSpan]]?
                for _ in 0..<iterations {
                    let t = ContinuousClock.now
                    if task == "routing" { choices = try model!.classify(item.text,labels: labels) }
                    else { entities = try model!.extractEntities(item.text,labels: ["person","time"],descriptions: descriptions) }
                    timings.append(ms(t)); peak = max(peak,footprint())
                }
                let correct: Bool
                if let choices { correct = choices.max(by: { $0.probability < $1.probability })?.label == item.action }
                else { correct = (item.entities ?? [:]).allSatisfy { key,expected in
                    (entities?[key] ?? []).map { $0.text.lowercased() }.sorted() == expected.map { $0.lowercased() }.sorted()
                }}
                rows.append(Row(text:item.text,task:task,milliseconds:timings,choices:choices,entities:entities,correct:correct))
                print("\(task): \(String(format: "%.2f",timings.sorted()[timings.count/2])) ms, match=\(correct)")
            }
        }
        memoryProfile.append(memoryPoint("after_requests"))
        Memory.clearCache()
        memoryProfile.append(memoryPoint("after_cache_clear"))
        model = nil
        Memory.clearCache()
        memoryProfile.append(memoryPoint("after_unload"))
        peak = max(peak,sampler.stop())
        var usage = rusage(); getrusage(RUSAGE_SELF,&usage)
        let report = Report(loadMilliseconds:load,rows:rows,firstRequestMilliseconds:firstRequests,processPeakRSSBytes:UInt64(usage.ru_maxrss),sampledPhysicalFootprintBytes:peak,modelDirectory:directory,memoryProfile:memoryProfile,evaluateLayers:options.evaluateLayers,cacheLimitBytes:Memory.cacheLimit,footprintSamplingIntervalMilliseconds:20)
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted,.sortedKeys]
        try encoder.encode(report).write(to: URL(fileURLWithPath:output),options:.atomic)
    }
}
