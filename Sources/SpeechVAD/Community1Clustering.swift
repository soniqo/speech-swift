#if canImport(CoreML)
import Foundation

/// Exported x-vector and PLDA transforms used by Community-1's VBx stage.
struct Community1PLDA {
    let xvectorMean1: [Float]
    let xvectorMean2: [Float]
    /// Row-major `[256, 128]` matrix.
    let xvectorLDA: [Float]
    let pldaMean: [Float]
    /// Row-major `[128, 128]` matrix.
    let pldaTransform: [Float]
    let phi: [Float]

    static func load(from url: URL) throws -> Community1PLDA {
        let data: Data
        do {
            data = try Data(contentsOf: url, options: .mappedIfSafe)
        } catch {
            throw Community1Error.invalidBundle(
                "could not load \(url.lastPathComponent): \(error.localizedDescription)"
            )
        }

        guard data.count >= 8 else {
            throw Community1Error.invalidBundle("PLDA safetensors header is truncated")
        }
        let headerLength = data.prefix(8).enumerated().reduce(UInt64(0)) { result, pair in
            result | UInt64(pair.element) << UInt64(pair.offset * 8)
        }
        guard headerLength <= UInt64(data.count - 8),
              headerLength <= UInt64(Int.max) else {
            throw Community1Error.invalidBundle("PLDA safetensors header length is invalid")
        }
        let headerEnd = 8 + Int(headerLength)
        let headerObject: [String: Any]
        do {
            guard let object = try JSONSerialization.jsonObject(
                with: data[8..<headerEnd]
            ) as? [String: Any] else {
                throw Community1Error.invalidBundle("PLDA safetensors header is not an object")
            }
            headerObject = object
        } catch let error as Community1Error {
            throw error
        } catch {
            throw Community1Error.invalidBundle(
                "could not decode PLDA safetensors header: \(error.localizedDescription)"
            )
        }

        func tensor(_ key: String, shape: [Int]) throws -> [Float] {
            guard let descriptor = headerObject[key] as? [String: Any] else {
                throw Community1Error.invalidBundle("PLDA weights are missing '\(key)'")
            }
            guard descriptor["dtype"] as? String == "F32",
                  let storedShape = descriptor["shape"] as? [Int],
                  let offsets = descriptor["data_offsets"] as? [Int],
                  offsets.count == 2 else {
                throw Community1Error.invalidBundle("PLDA tensor '\(key)' metadata is invalid")
            }
            guard storedShape == shape else {
                throw Community1Error.invalidBundle(
                    "PLDA tensor '\(key)' has shape \(storedShape), expected \(shape)"
                )
            }
            let elementCount = shape.reduce(1, *)
            let byteCount = elementCount * MemoryLayout<UInt32>.size
            guard offsets[0] >= 0,
                  offsets[1] - offsets[0] == byteCount,
                  headerEnd + offsets[1] <= data.count else {
                throw Community1Error.invalidBundle("PLDA tensor '\(key)' data range is invalid")
            }
            return data.withUnsafeBytes { rawBuffer in
                let bytes = rawBuffer.bindMemory(to: UInt8.self)
                let start = headerEnd + offsets[0]
                return (0..<elementCount).map { index in
                    let position = start + index * 4
                    let bits = UInt32(bytes[position])
                        | UInt32(bytes[position + 1]) << 8
                        | UInt32(bytes[position + 2]) << 16
                        | UInt32(bytes[position + 3]) << 24
                    return Float(bitPattern: bits)
                }
            }
        }

        return try Community1PLDA(
            xvectorMean1: tensor("xvector.mean1", shape: [256]),
            xvectorMean2: tensor("xvector.mean2", shape: [128]),
            xvectorLDA: tensor("xvector.lda", shape: [256, 128]),
            pldaMean: tensor("plda.mean", shape: [128]),
            pldaTransform: tensor("plda.transform", shape: [128, 128]),
            phi: tensor("plda.phi", shape: [128])
        )
    }

    /// Center, normalize, apply LDA, and project into diagonal PLDA space.
    func transform(_ embeddings: [[Float]]) -> [[Double]] {
        let inputDimension = Community1Config.embeddingDimension
        let outputDimension = Community1Config.pldaDimension
        let firstScale = sqrt(Double(inputDimension))
        let secondScale = sqrt(Double(outputDimension))

        return embeddings.map { embedding in
            var centered = [Double](repeating: 0, count: inputDimension)
            for index in 0..<inputDimension {
                centered[index] = Double(embedding[index] - xvectorMean1[index])
            }
            Self.normalize(&centered)

            var xvector = [Double](repeating: 0, count: outputDimension)
            for output in 0..<outputDimension {
                var sum = 0.0
                for input in 0..<inputDimension {
                    sum += firstScale * centered[input]
                        * Double(xvectorLDA[input * outputDimension + output])
                }
                xvector[output] = sum - Double(xvectorMean2[output])
            }
            Self.normalize(&xvector)
            for index in xvector.indices {
                xvector[index] *= secondScale
            }

            var projected = [Double](repeating: 0, count: outputDimension)
            for output in 0..<outputDimension {
                var sum = 0.0
                let row = output * outputDimension
                for input in 0..<outputDimension {
                    sum += (xvector[input] - Double(pldaMean[input]))
                        * Double(pldaTransform[row + input])
                }
                projected[output] = sum
            }
            return projected
        }
    }

    private static func normalize(_ vector: inout [Double]) {
        let norm = sqrt(vector.reduce(0) { $0 + $1 * $1 })
        guard norm > 0 else { return }
        for index in vector.indices {
            vector[index] /= norm
        }
    }
}

struct Community1ClusterResult {
    let hardClusters: [[Int]]
    let centroids: [[Float]]
}

private struct Community1DistanceCandidate {
    let squaredDistance: Double
    let left: Int
    let right: Int

    static func orderedBefore(
        _ lhs: Community1DistanceCandidate,
        _ rhs: Community1DistanceCandidate
    ) -> Bool {
        if lhs.squaredDistance != rhs.squaredDistance {
            return lhs.squaredDistance < rhs.squaredDistance
        }
        if lhs.left != rhs.left { return lhs.left < rhs.left }
        return lhs.right < rhs.right
    }
}

private struct Community1DistanceHeap {
    private var storage = [Community1DistanceCandidate]()

    mutating func reserveCapacity(_ capacity: Int) {
        storage.reserveCapacity(capacity)
    }

    mutating func push(_ candidate: Community1DistanceCandidate) {
        storage.append(candidate)
        var child = storage.count - 1
        while child > 0 {
            let parent = (child - 1) / 2
            guard Community1DistanceCandidate.orderedBefore(
                storage[child], storage[parent]
            ) else { break }
            storage.swapAt(child, parent)
            child = parent
        }
    }

    mutating func pop() -> Community1DistanceCandidate? {
        guard !storage.isEmpty else { return nil }
        if storage.count == 1 { return storage.removeLast() }
        let result = storage[0]
        storage[0] = storage.removeLast()
        var parent = 0
        while true {
            let left = parent * 2 + 1
            guard left < storage.count else { break }
            let right = left + 1
            let child: Int
            if right < storage.count,
               Community1DistanceCandidate.orderedBefore(storage[right], storage[left]) {
                child = right
            } else {
                child = left
            }
            guard Community1DistanceCandidate.orderedBefore(
                storage[child], storage[parent]
            ) else { break }
            storage.swapAt(parent, child)
            parent = child
        }
        return result
    }
}

/// Native host implementation of Community-1's AHC initialization and VBx loop.
enum Community1Clustering {
    static func cluster(
        embeddings: [[[Float]]],
        segmentations: [[[Float]]],
        plda: Community1PLDA,
        config: Community1Config,
        bounds: Community1SpeakerBounds
    ) throws -> Community1ClusterResult {
        guard embeddings.count == segmentations.count else {
            throw Community1Error.inference(
                stage: "clustering", reason: "embedding and segmentation chunk counts differ"
            )
        }
        try validate(bounds)

        var trainEmbeddings = [[Float]]()
        for chunk in embeddings.indices {
            for speaker in 0..<Community1Config.localSpeakers {
                var cleanFrames = 0
                for frame in 0..<Community1Config.framesPerChunk {
                    if segmentations[chunk][frame].reduce(0, +) == 1,
                       segmentations[chunk][frame][speaker] > 0 {
                        cleanFrames += 1
                    }
                }
                let activeEnough = Float(cleanFrames)
                    >= config.minimumActiveRatio * Float(Community1Config.framesPerChunk)
                let valid = embeddings[chunk][speaker].allSatisfy(\.isFinite)
                if activeEnough, valid {
                    trainEmbeddings.append(embeddings[chunk][speaker])
                }
            }
        }

        guard !trainEmbeddings.isEmpty else {
            return Community1ClusterResult(
                hardClusters: Array(
                    repeating: [Int](repeating: -2, count: Community1Config.localSpeakers),
                    count: embeddings.count
                ),
                centroids: []
            )
        }

        let centroids: [[Float]]
        if trainEmbeddings.count < 2 {
            centroids = [trainEmbeddings[0]]
        } else {
            let initial = centroidLinkageLabels(
                trainEmbeddings, threshold: config.clusteringThreshold
            )
            let features = plda.transform(trainEmbeddings)
            let (responsibilities, priors) = vbx(
                initialLabels: initial,
                features: features,
                phi: plda.phi.map(Double.init),
                config: config
            )

            let kept = priors.indices.filter { priors[$0] > Double(config.speakerPriorCutoff) }
            let retained = kept.isEmpty ? [priors.indices.max(by: { priors[$0] < priors[$1] })!] : kept
            centroids = weightedCentroids(
                embeddings: trainEmbeddings,
                responsibilities: responsibilities,
                retained: retained
            )
        }

        let requestedCount = resolvedCount(
            automatic: centroids.count,
            available: trainEmbeddings.count,
            bounds: bounds
        )
        let finalCentroids: [[Float]]
        if requestedCount != centroids.count {
            finalCentroids = kMeansCentroids(trainEmbeddings, count: requestedCount)
        } else {
            finalCentroids = centroids
        }

        let hard = assign(
            embeddings: embeddings,
            segmentations: segmentations,
            centroids: finalCentroids,
            constrained: requestedCount == centroids.count
        )
        return Community1ClusterResult(hardClusters: hard, centroids: finalCentroids)
    }

    private static func validate(_ bounds: Community1SpeakerBounds) throws {
        if let exact = bounds.exact, exact < 1 {
            throw Community1Error.invalidSpeakerBounds("exact must be at least one")
        }
        guard bounds.minimum >= 1 else {
            throw Community1Error.invalidSpeakerBounds("minimum must be at least one")
        }
        if let maximum = bounds.maximum, maximum < bounds.minimum {
            throw Community1Error.invalidSpeakerBounds("maximum must not be below minimum")
        }
    }

    private static func resolvedCount(
        automatic: Int,
        available: Int,
        bounds: Community1SpeakerBounds
    ) -> Int {
        if let exact = bounds.exact {
            return max(1, min(available, exact))
        }
        let maximum = min(available, bounds.maximum ?? available)
        return max(1, min(maximum, max(bounds.minimum, automatic)))
    }

    /// SciPy-compatible centroid linkage cut over L2-normalized embeddings.
    static func centroidLinkageLabels(
        _ embeddings: [[Float]], threshold: Float
    ) -> [Int] {
        guard embeddings.count > 1 else { return [0] }

        let count = embeddings.count
        let nodeCount = 2 * count - 1
        let normalized = embeddings.map { embedding -> [Double] in
            var value = embedding.map(Double.init)
            normalize(&value)
            return value
        }
        var sizes = [Int](repeating: 0, count: nodeCount)
        for index in 0..<count { sizes[index] = 1 }
        var maximumSubtreeDistanceSquared = [Double](repeating: 0, count: nodeCount)
        var representative = Array(repeating: 0, count: nodeCount)
        for index in 0..<count { representative[index] = index }
        var active = [Bool](repeating: false, count: nodeCount)
        for index in 0..<count { active[index] = true }

        // Centroid linkage can be updated exactly from the previous squared
        // distances. Keeping a lazy min-heap avoids the O(n^3) full scan that
        // otherwise dominates long recordings with one embedding per second.
        var distances = [Double](
            repeating: .infinity,
            count: nodeCount * nodeCount
        )
        func distanceIndex(_ left: Int, _ right: Int) -> Int {
            left * nodeCount + right
        }
        func setDistance(_ value: Double, _ left: Int, _ right: Int) {
            distances[distanceIndex(left, right)] = value
            distances[distanceIndex(right, left)] = value
        }

        var candidates = Community1DistanceHeap()
        candidates.reserveCapacity(count * (count - 1) / 2)
        for left in 0..<(count - 1) {
            for right in (left + 1)..<count {
                let distance = squaredDistance(normalized[left], normalized[right])
                setDistance(distance, left, right)
                candidates.push(.init(
                    squaredDistance: distance,
                    left: left,
                    right: right
                ))
            }
        }

        var parent = Array(0..<count)
        func root(_ value: Int, _ parent: inout [Int]) -> Int {
            var current = value
            while parent[current] != current { current = parent[current] }
            var node = value
            while parent[node] != node {
                let next = parent[node]
                parent[node] = current
                node = next
            }
            return current
        }
        func unite(_ left: Int, _ right: Int, _ parent: inout [Int]) {
            let a = root(left, &parent)
            let b = root(right, &parent)
            if a != b { parent[b] = a }
        }

        for mergeIndex in 0..<(count - 1) {
            var best: Community1DistanceCandidate?
            while let candidate = candidates.pop() {
                if active[candidate.left], active[candidate.right] {
                    best = candidate
                    break
                }
            }
            guard let best else { break }
            let bestLeft = best.left
            let bestRight = best.right

            let merged = count + mergeIndex
            let leftSize = Double(sizes[bestLeft])
            let rightSize = Double(sizes[bestRight])
            let total = leftSize + rightSize
            sizes[merged] = Int(total)
            representative[merged] = representative[bestLeft]
            maximumSubtreeDistanceSquared[merged] = max(
                best.squaredDistance,
                max(
                    maximumSubtreeDistanceSquared[bestLeft],
                    maximumSubtreeDistanceSquared[bestRight]
                )
            )
            if maximumSubtreeDistanceSquared[merged]
                <= Double(threshold) * Double(threshold) {
                unite(
                    representative[bestLeft],
                    representative[bestRight],
                    &parent
                )
            }

            for other in 0..<merged where active[other]
                && other != bestLeft && other != bestRight {
                // Lance-Williams update for UPGMC (centroid linkage), applied
                // to squared Euclidean distances.
                let leftDistance = distances[distanceIndex(bestLeft, other)]
                let rightDistance = distances[distanceIndex(bestRight, other)]
                let mergedDistance = max(
                    0,
                    leftSize / total * leftDistance
                        + rightSize / total * rightDistance
                        - leftSize * rightSize / (total * total)
                            * best.squaredDistance
                )
                setDistance(mergedDistance, merged, other)
                candidates.push(.init(
                    squaredDistance: mergedDistance,
                    left: min(merged, other),
                    right: max(merged, other)
                ))
            }
            active[bestLeft] = false
            active[bestRight] = false
            active[merged] = true
        }

        var compact = [Int: Int]()
        var labels = [Int](repeating: 0, count: count)
        for index in 0..<count {
            let representative = root(index, &parent)
            if compact[representative] == nil {
                compact[representative] = compact.count
            }
            labels[index] = compact[representative]!
        }
        return labels
    }

    static func vbx(
        initialLabels: [Int],
        features: [[Double]],
        phi: [Double],
        config: Community1Config
    ) -> (responsibilities: [[Double]], priors: [Double]) {
        let frames = features.count
        let dimension = phi.count
        let speakers = (initialLabels.max() ?? 0) + 1
        var gamma = Array(
            repeating: [Double](repeating: 0, count: speakers), count: frames
        )
        for frame in 0..<frames {
            let logits = (0..<speakers).map {
                $0 == initialLabels[frame] ? Double(config.initialSmoothing) : 0
            }
            gamma[frame] = softmax(logits)
        }
        var priors = [Double](repeating: 1 / Double(speakers), count: speakers)
        let fa = Double(config.fa)
        let fb = Double(config.fb)
        let ratio = fa / fb
        let sqrtPhi = phi.map(sqrt)
        let logConstant = Double(dimension) * log(2 * Double.pi)
        let rho = features.map { feature in
            zip(feature, sqrtPhi).map(*)
        }
        let frameConstant = features.map { feature in
            -0.5 * (feature.reduce(0) { $0 + $1 * $1 } + logConstant)
        }

        var previousELBO: Double?
        for _ in 0..<config.maxIterations {
            var speakerMass = [Double](repeating: 0, count: speakers)
            for frame in 0..<frames {
                for speaker in 0..<speakers {
                    speakerMass[speaker] += gamma[frame][speaker]
                }
            }

            var inverseL = Array(
                repeating: [Double](repeating: 0, count: dimension), count: speakers
            )
            var alpha = inverseL
            for speaker in 0..<speakers {
                for dim in 0..<dimension {
                    inverseL[speaker][dim] = 1 / (1 + ratio * speakerMass[speaker] * phi[dim])
                    var weighted = 0.0
                    for frame in 0..<frames {
                        weighted += gamma[frame][speaker] * rho[frame][dim]
                    }
                    alpha[speaker][dim] = ratio * inverseL[speaker][dim] * weighted
                }
            }

            var logProbability = Array(
                repeating: [Double](repeating: 0, count: speakers), count: frames
            )
            var logMarginal = [Double](repeating: 0, count: frames)
            for frame in 0..<frames {
                for speaker in 0..<speakers {
                    var first = 0.0
                    var second = 0.0
                    for dim in 0..<dimension {
                        first += rho[frame][dim] * alpha[speaker][dim]
                        second += (inverseL[speaker][dim]
                            + alpha[speaker][dim] * alpha[speaker][dim]) * phi[dim]
                    }
                    logProbability[frame][speaker] = fa
                        * (first - 0.5 * second + frameConstant[frame])
                        + log(priors[speaker] + 1e-8)
                }
                logMarginal[frame] = logSumExp(logProbability[frame])
                for speaker in 0..<speakers {
                    gamma[frame][speaker] = exp(
                        logProbability[frame][speaker] - logMarginal[frame]
                    )
                }
            }

            priors = [Double](repeating: 0, count: speakers)
            for frame in 0..<frames {
                for speaker in 0..<speakers {
                    priors[speaker] += gamma[frame][speaker]
                }
            }
            let priorTotal = priors.reduce(0, +)
            for speaker in priors.indices { priors[speaker] /= priorTotal }

            var regularization = 0.0
            for speaker in 0..<speakers {
                for dim in 0..<dimension {
                    regularization += log(inverseL[speaker][dim])
                        - inverseL[speaker][dim]
                        - alpha[speaker][dim] * alpha[speaker][dim] + 1
                }
            }
            let elbo = logMarginal.reduce(0, +) + fb * 0.5 * regularization
            if let previousELBO,
               elbo - previousELBO < Double(config.convergenceEpsilon) {
                break
            }
            previousELBO = elbo
        }
        return (gamma, priors)
    }

    private static func weightedCentroids(
        embeddings: [[Float]],
        responsibilities: [[Double]],
        retained: [Int]
    ) -> [[Float]] {
        retained.map { speaker in
            var centroid = [Double](repeating: 0, count: Community1Config.embeddingDimension)
            var total = 0.0
            for frame in embeddings.indices {
                let weight = responsibilities[frame][speaker]
                total += weight
                for dimension in 0..<Community1Config.embeddingDimension {
                    centroid[dimension] += weight * Double(embeddings[frame][dimension])
                }
            }
            return centroid.map { Float($0 / max(total, 1e-12)) }
        }
    }

    private static func assign(
        embeddings: [[[Float]]],
        segmentations: [[[Float]]],
        centroids: [[Float]],
        constrained: Bool
    ) -> [[Int]] {
        guard !centroids.isEmpty else {
            return Array(
                repeating: [Int](repeating: -2, count: Community1Config.localSpeakers),
                count: embeddings.count
            )
        }

        return embeddings.indices.map { chunk in
            var scores = Array(
                repeating: [Double](repeating: 0, count: centroids.count),
                count: Community1Config.localSpeakers
            )
            for speaker in 0..<Community1Config.localSpeakers {
                for cluster in centroids.indices {
                    scores[speaker][cluster] = 2 - cosineDistance(
                        embeddings[chunk][speaker], centroids[cluster]
                    )
                }
            }
            let finiteMinimum = scores.flatMap { $0 }.filter(\.isFinite).min() ?? -1
            for speaker in 0..<Community1Config.localSpeakers {
                if segmentations[chunk].allSatisfy({ $0[speaker] == 0 }) {
                    scores[speaker] = [Double](
                        repeating: finiteMinimum - 1, count: centroids.count
                    )
                }
                for cluster in centroids.indices where !scores[speaker][cluster].isFinite {
                    scores[speaker][cluster] = finiteMinimum
                }
            }
            if constrained {
                return constrainedArgmax(scores)
            }
            return scores.map { row in
                row.indices.max(by: { row[$0] < row[$1] }) ?? 0
            }
        }
    }

    /// Exact maximum-weight assignment for three local speakers.
    static func constrainedArgmax(_ scores: [[Double]]) -> [Int] {
        let speakers = scores.count
        let clusters = scores.first?.count ?? 0
        guard speakers > 0, clusters > 0 else { return [Int](repeating: -2, count: speakers) }
        let assignmentCount = min(speakers, clusters)
        var bestScore = -Double.infinity
        var best = [Int](repeating: -2, count: speakers)

        func search(
            speaker: Int,
            assigned: Int,
            used: Set<Int>,
            currentScore: Double,
            current: [Int]
        ) {
            if speaker == speakers {
                guard assigned == assignmentCount else { return }
                if currentScore > bestScore {
                    bestScore = currentScore
                    best = current
                }
                return
            }

            let remainingSpeakers = speakers - speaker
            if assigned + remainingSpeakers > assignmentCount {
                search(
                    speaker: speaker + 1,
                    assigned: assigned,
                    used: used,
                    currentScore: currentScore,
                    current: current
                )
            }
            guard assigned < assignmentCount else { return }
            for cluster in 0..<clusters where !used.contains(cluster) {
                var next = current
                next[speaker] = cluster
                var nextUsed = used
                nextUsed.insert(cluster)
                search(
                    speaker: speaker + 1,
                    assigned: assigned + 1,
                    used: nextUsed,
                    currentScore: currentScore + scores[speaker][cluster],
                    current: next
                )
            }
        }
        search(
            speaker: 0,
            assigned: 0,
            used: [],
            currentScore: 0,
            current: [Int](repeating: -2, count: speakers)
        )
        return best
    }

    private static func kMeansCentroids(_ embeddings: [[Float]], count: Int) -> [[Float]] {
        let normalized = embeddings.map { embedding -> [Double] in
            var value = embedding.map(Double.init)
            normalize(&value)
            return value
        }
        var centers = [normalized[0]]
        while centers.count < count {
            let next = normalized.indices.max { left, right in
                nearestSquaredDistance(normalized[left], centers)
                    < nearestSquaredDistance(normalized[right], centers)
            }!
            centers.append(normalized[next])
        }

        var labels = [Int](repeating: 0, count: embeddings.count)
        for _ in 0..<100 {
            let nextLabels = normalized.map { point in
                centers.indices.min(by: {
                    squaredDistance(point, centers[$0]) < squaredDistance(point, centers[$1])
                })!
            }
            if nextLabels == labels { break }
            labels = nextLabels
            for cluster in 0..<count {
                let members = normalized.indices.filter { labels[$0] == cluster }
                guard !members.isEmpty else { continue }
                for dimension in centers[cluster].indices {
                    centers[cluster][dimension] = members.reduce(0) {
                        $0 + normalized[$1][dimension]
                    } / Double(members.count)
                }
            }
        }

        return (0..<count).map { cluster in
            let members = embeddings.indices.filter { labels[$0] == cluster }
            guard !members.isEmpty else { return embeddings[cluster % embeddings.count] }
            return (0..<Community1Config.embeddingDimension).map { dimension in
                members.reduce(Float(0)) { $0 + embeddings[$1][dimension] }
                    / Float(members.count)
            }
        }
    }

    private static func softmax(_ values: [Double]) -> [Double] {
        let maximum = values.max() ?? 0
        let exponentials = values.map { exp($0 - maximum) }
        let total = exponentials.reduce(0, +)
        return exponentials.map { $0 / total }
    }

    private static func logSumExp(_ values: [Double]) -> Double {
        let maximum = values.max() ?? 0
        return maximum + log(values.reduce(0) { $0 + exp($1 - maximum) })
    }

    private static func normalize(_ vector: inout [Double]) {
        let norm = sqrt(vector.reduce(0) { $0 + $1 * $1 })
        guard norm > 0 else { return }
        for index in vector.indices { vector[index] /= norm }
    }

    private static func squaredDistance(_ left: [Double], _ right: [Double]) -> Double {
        zip(left, right).reduce(0) { result, pair in
            let difference = pair.0 - pair.1
            return result + difference * difference
        }
    }

    private static func nearestSquaredDistance(
        _ point: [Double], _ centers: [[Double]]
    ) -> Double {
        centers.map { squaredDistance(point, $0) }.min() ?? .infinity
    }

    private static func cosineDistance(_ left: [Float], _ right: [Float]) -> Double {
        var dot = 0.0
        var leftNorm = 0.0
        var rightNorm = 0.0
        for index in 0..<min(left.count, right.count) {
            let a = Double(left[index])
            let b = Double(right[index])
            dot += a * b
            leftNorm += a * a
            rightNorm += b * b
        }
        let denominator = sqrt(leftNorm * rightNorm)
        guard denominator > 0 else { return .nan }
        return 1 - dot / denominator
    }
}
#endif
