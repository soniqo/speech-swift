import Accelerate

// MARK: - Cosine Affinity Matrix

/// Compute cosine affinity matrix from N embeddings of dimension D.
/// Returns flat N×N matrix with values in [0,1], diagonal = 0.
func cosineAffinityMatrix(embeddings: [[Float]]) -> [Float] {
    let n = embeddings.count
    guard n > 0 else { return [] }
    let d = embeddings[0].count

    // Stack embeddings into flat N×D matrix (row-major)
    var E = [Float](repeating: 0, count: n * d)
    for i in 0..<n {
        E.replaceSubrange(i * d..<(i + 1) * d, with: embeddings[i])
    }

    // L2-normalize each row
    for i in 0..<n {
        var norm: Float = 0
        E.withUnsafeBufferPointer { buf in
            vDSP_dotpr(buf.baseAddress! + i * d, 1, buf.baseAddress! + i * d, 1, &norm, vDSP_Length(d))
        }
        norm = sqrt(norm + 1e-10)
        var invNorm = 1.0 / norm
        E.withUnsafeMutableBufferPointer { buf in
            vDSP_vsmul(buf.baseAddress! + i * d, 1, &invNorm, buf.baseAddress! + i * d, 1, vDSP_Length(d))
        }
    }

    // Transpose E (N×D row-major) → Et (D×N row-major)
    var Et = [Float](repeating: 0, count: d * n)
    vDSP_mtrans(E, 1, &Et, 1, vDSP_Length(n), vDSP_Length(d))

    // E * E^T → N×N dot product matrix
    var A = [Float](repeating: 0, count: n * n)
    vDSP_mmul(E, 1, Et, 1, &A, 1, vDSP_Length(n), vDSP_Length(n), vDSP_Length(d))

    // Map from [-1,1] to [0,1]: affinity = (1 + dot) / 2
    var half: Float = 0.5
    var one: Float = 1.0
    // A = (A + 1) * 0.5
    vDSP_vsadd(A, 1, &one, &A, 1, vDSP_Length(n * n))
    vDSP_vsmul(A, 1, &half, &A, 1, vDSP_Length(n * n))

    // Zero diagonal
    for i in 0..<n {
        A[i * n + i] = 0
    }

    return A
}

// MARK: - Normalized Laplacian

/// Compute symmetric normalized Laplacian L_sym = I - D^{-1/2} A D^{-1/2}.
/// Input: flat N×N affinity matrix. Returns flat N×N Laplacian.
func normalizedLaplacian(affinity: [Float], n: Int) -> [Float] {
    // Degree vector: d_i = sum of row i
    var degrees = [Float](repeating: 0, count: n)
    affinity.withUnsafeBufferPointer { buf in
        for i in 0..<n {
            var sum: Float = 0
            vDSP_sve(buf.baseAddress! + i * n, 1, &sum, vDSP_Length(n))
            degrees[i] = sum
        }
    }

    // D^{-1/2}
    var dInvSqrt = [Float](repeating: 0, count: n)
    for i in 0..<n {
        dInvSqrt[i] = degrees[i] > 1e-10 ? 1.0 / sqrt(degrees[i]) : 0
    }

    // L = I - D^{-1/2} A D^{-1/2}
    var L = [Float](repeating: 0, count: n * n)
    for i in 0..<n {
        for j in 0..<n {
            if i == j {
                L[i * n + j] = 1.0 - dInvSqrt[i] * affinity[i * n + j] * dInvSqrt[j]
            } else {
                L[i * n + j] = -dInvSqrt[i] * affinity[i * n + j] * dInvSqrt[j]
            }
        }
    }

    return L
}

// MARK: - Eigendecomposition

/// Eigendecompose a symmetric N×N matrix using LAPACK ssyev_.
/// Returns eigenvalues (ascending) and eigenvectors (columns of N×N matrix, column-major).
func eigendecompose(matrix: inout [Float], n: Int) -> (eigenvalues: [Float], eigenvectors: [Float]) {
    var eigenvalues = [Float](repeating: 0, count: n)

    // Use LAPACK ssyev via Accelerate
    // For symmetric matrices, row-major == column-major, so no transpose needed.
    var N = __CLPK_integer(n)
    var lda = __CLPK_integer(n)
    var lwork = __CLPK_integer(max(3 * n - 1, 1))
    var work = [Float](repeating: 0, count: Int(lwork))
    var info: __CLPK_integer = 0

    // jobz='V' → compute eigenvectors, uplo='U' → upper triangle stored
    var jobz = CChar(Character("V").asciiValue!)
    var uplo = CChar(Character("U").asciiValue!)
    ssyev_(&jobz, &uplo, &N, &matrix, &lda, &eigenvalues, &work, &lwork, &info)

    // matrix now contains eigenvectors as columns (column-major)
    return (eigenvalues, matrix)
}

// MARK: - K-Means++

/// K-means clustering with k-means++ initialization.
/// Points: flat N×dim array (row-major). Returns cluster assignments [0..<k].
func kMeans(points: [Float], n: Int, k: Int, dim: Int, maxIter: Int = 100, seed: UInt64 = 42) -> [Int] {
    guard n > 0, k > 0 else { return [] }
    if k >= n {
        return Array(0..<n)
    }

    var rng = SeededRNG(seed: seed)

    // K-means++ initialization
    var centroids = [Float](repeating: 0, count: k * dim)

    // Pick first centroid randomly
    let first = Int.random(in: 0..<n, using: &rng)
    centroids.replaceSubrange(0..<dim, with: points[first * dim..<(first + 1) * dim])

    // Pick remaining centroids proportional to squared distance
    for c in 1..<k {
        var distances = [Float](repeating: Float.infinity, count: n)
        for i in 0..<n {
            for prev in 0..<c {
                var dist: Float = 0
                for d in 0..<dim {
                    let diff = points[i * dim + d] - centroids[prev * dim + d]
                    dist += diff * diff
                }
                distances[i] = min(distances[i], dist)
            }
        }

        // Weighted random selection
        var totalDist: Float = 0
        vDSP_sve(distances, 1, &totalDist, vDSP_Length(n))
        let threshold = Float.random(in: 0..<totalDist, using: &rng)
        var cumulative: Float = 0
        var chosen = n - 1
        for i in 0..<n {
            cumulative += distances[i]
            if cumulative >= threshold {
                chosen = i
                break
            }
        }
        centroids.replaceSubrange(c * dim..<(c + 1) * dim, with: points[chosen * dim..<(chosen + 1) * dim])
    }

    // Lloyd's algorithm
    var assignments = [Int](repeating: 0, count: n)

    for _ in 0..<maxIter {
        // Assign each point to nearest centroid
        var changed = false
        for i in 0..<n {
            var bestCluster = 0
            var bestDist: Float = Float.infinity
            for c in 0..<k {
                var dist: Float = 0
                for d in 0..<dim {
                    let diff = points[i * dim + d] - centroids[c * dim + d]
                    dist += diff * diff
                }
                if dist < bestDist {
                    bestDist = dist
                    bestCluster = c
                }
            }
            if assignments[i] != bestCluster {
                assignments[i] = bestCluster
                changed = true
            }
        }

        if !changed { break }

        // Update centroids
        var counts = [Int](repeating: 0, count: k)
        var newCentroids = [Float](repeating: 0, count: k * dim)
        for i in 0..<n {
            let c = assignments[i]
            counts[c] += 1
            for d in 0..<dim {
                newCentroids[c * dim + d] += points[i * dim + d]
            }
        }
        for c in 0..<k {
            if counts[c] > 0 {
                var invCount = 1.0 / Float(counts[c])
                newCentroids.withUnsafeMutableBufferPointer { buf in
                    vDSP_vsmul(buf.baseAddress! + c * dim, 1, &invCount,
                               buf.baseAddress! + c * dim, 1, vDSP_Length(dim))
                }
            }
        }
        centroids = newCentroids
    }

    return assignments
}

// MARK: - GMM-BIC

/// Fit diagonal-covariance GMM via EM, return BIC score and assignments.
/// Data: flat N×D array (row-major).
func gmmBIC(data: [Float], n: Int, d: Int, k: Int, maxIter: Int = 50, seed: UInt64 = 42) -> (bic: Float, assignments: [Int]) {
    guard n > k, k > 0 else {
        return (Float.infinity, Array(repeating: 0, count: n))
    }

    let varianceFloor: Float = 1e-3

    // Initialize with k-means
    let initAssign = kMeans(points: data, n: n, k: k, dim: d, seed: seed)

    // Initialize GMM parameters from k-means
    var weights = [Float](repeating: 0, count: k)
    var means = [Float](repeating: 0, count: k * d)
    var variances = [Float](repeating: varianceFloor, count: k * d)

    for c in 0..<k {
        var count = 0
        for i in 0..<n {
            if initAssign[i] == c {
                count += 1
                for j in 0..<d {
                    means[c * d + j] += data[i * d + j]
                }
            }
        }
        if count > 0 {
            weights[c] = Float(count) / Float(n)
            for j in 0..<d { means[c * d + j] /= Float(count) }
            // Compute variance
            for i in 0..<n {
                if initAssign[i] == c {
                    for j in 0..<d {
                        let diff = data[i * d + j] - means[c * d + j]
                        variances[c * d + j] += diff * diff
                    }
                }
            }
            for j in 0..<d {
                variances[c * d + j] = max(variances[c * d + j] / Float(count), varianceFloor)
            }
        } else {
            weights[c] = 1.0 / Float(k)
        }
    }

    // EM iterations
    var responsibilities = [Float](repeating: 0, count: n * k)

    for _ in 0..<maxIter {
        // E-step: compute log responsibilities
        for i in 0..<n {
            var maxLogR: Float = -Float.infinity
            for c in 0..<k {
                var logProb: Float = 0
                for j in 0..<d {
                    let diff = data[i * d + j] - means[c * d + j]
                    logProb -= 0.5 * (diff * diff / variances[c * d + j] + log(variances[c * d + j]))
                }
                logProb -= 0.5 * Float(d) * log(2 * Float.pi)
                logProb += log(max(weights[c], 1e-30))
                responsibilities[i * k + c] = logProb
                maxLogR = max(maxLogR, logProb)
            }
            // Log-sum-exp normalization
            var sumExp: Float = 0
            for c in 0..<k {
                responsibilities[i * k + c] = exp(responsibilities[i * k + c] - maxLogR)
                sumExp += responsibilities[i * k + c]
            }
            let invSum = 1.0 / max(sumExp, 1e-30)
            for c in 0..<k {
                responsibilities[i * k + c] *= invSum
            }
        }

        // M-step
        var newWeights = [Float](repeating: 0, count: k)
        var newMeans = [Float](repeating: 0, count: k * d)
        var newVariances = [Float](repeating: varianceFloor, count: k * d)

        for c in 0..<k {
            var nk: Float = 0
            for i in 0..<n {
                let r = responsibilities[i * k + c]
                nk += r
                for j in 0..<d {
                    newMeans[c * d + j] += r * data[i * d + j]
                }
            }
            if nk > 1e-10 {
                newWeights[c] = nk / Float(n)
                for j in 0..<d { newMeans[c * d + j] /= nk }
                for i in 0..<n {
                    let r = responsibilities[i * k + c]
                    for j in 0..<d {
                        let diff = data[i * d + j] - newMeans[c * d + j]
                        newVariances[c * d + j] += r * diff * diff
                    }
                }
                for j in 0..<d {
                    newVariances[c * d + j] = max(newVariances[c * d + j] / nk, varianceFloor)
                }
            } else {
                newWeights[c] = 1.0 / Float(k)
            }
        }

        weights = newWeights
        means = newMeans
        variances = newVariances
    }

    // Compute log-likelihood
    var logL: Float = 0
    var assignments = [Int](repeating: 0, count: n)
    for i in 0..<n {
        var maxLogR: Float = -Float.infinity
        var logProbs = [Float](repeating: 0, count: k)
        var bestC = 0
        var bestLogP: Float = -Float.infinity
        for c in 0..<k {
            var logProb: Float = 0
            for j in 0..<d {
                let diff = data[i * d + j] - means[c * d + j]
                logProb -= 0.5 * (diff * diff / variances[c * d + j] + log(variances[c * d + j]))
            }
            logProb -= 0.5 * Float(d) * log(2 * Float.pi)
            logProb += log(max(weights[c], 1e-30))
            logProbs[c] = logProb
            maxLogR = max(maxLogR, logProb)
            if logProb > bestLogP {
                bestLogP = logProb
                bestC = c
            }
        }
        assignments[i] = bestC

        // log(sum(exp(logProbs)))
        var sumExp: Float = 0
        for c in 0..<k { sumExp += exp(logProbs[c] - maxLogR) }
        logL += maxLogR + log(sumExp)
    }

    // BIC = -2*logL + p*ln(N)
    // p = k*(2d + 1) - 1  (k means + k diagonal covariances + k-1 mixture weights)
    let p = Float(k * (2 * d + 1) - 1)
    let bic = -2 * logL + p * log(Float(n))

    return (bic, assignments)
}

// MARK: - Spectral Clustering (Top-level)

/// Spectral clustering with GMM-BIC speaker count estimation.
///
/// - Parameters:
///   - embeddings: N speaker embeddings, each D-dimensional
///   - minClusters: minimum number of clusters (0 = automatic, minimum 1)
///   - maxClusters: maximum number of clusters (0 = automatic)
/// - Returns: cluster IDs [0..<k] and L2-normalized centroid for each cluster
func spectralClustering(
    embeddings: [[Float]],
    minClusters: Int = 0,
    maxClusters: Int = 0
) -> (clusterIds: [Int], centroids: [[Float]]) {
    let n = embeddings.count
    guard n > 0 else { return ([], []) }

    let effectiveMin = max(minClusters, 1)
    let effectiveMax = maxClusters > 0 ? min(maxClusters, n) : n

    if n == 1 {
        return ([0], [embeddings[0]])
    }

    // If min == max, skip model selection
    if effectiveMin == effectiveMax {
        let k = effectiveMin
        if k == 1 {
            return singleClusterResult(embeddings: embeddings)
        }
        return clusterWithK(embeddings: embeddings, k: k)
    }

    // Build affinity matrix and Laplacian
    let affinity = cosineAffinityMatrix(embeddings: embeddings)

    // Quick check: if minimum off-diagonal affinity is very high, all embeddings are similar → 1 cluster
    if effectiveMin <= 1 {
        var minAff: Float = 1.0
        for i in 0..<n {
            for j in 0..<n where i != j {
                minAff = min(minAff, affinity[i * n + j])
            }
        }
        if minAff > 0.95 {
            return singleClusterResult(embeddings: embeddings)
        }
    }

    var laplacian = normalizedLaplacian(affinity: affinity, n: n)

    // Eigendecompose
    let (eigenvalues, eigenvectors) = eigendecompose(matrix: &laplacian, n: n)

    // Select k via GMM-BIC on eigenvector representations
    let searchMax = min(effectiveMax, max(n / 2, 2), 10)  // cap search space
    let searchMin = effectiveMin

    if n < 4 || searchMax <= 1 {
        // Too few points for GMM — use eigengap heuristic
        let k = eigengapK(eigenvalues: eigenvalues, minK: searchMin, maxK: searchMax)
        if k == 1 {
            return singleClusterResult(embeddings: embeddings)
        }
        return clusterFromEigenvectors(
            eigenvectors: eigenvectors, eigenvalues: eigenvalues,
            n: n, k: k, embeddings: embeddings
        )
    }

    // Extract top-searchMax eigenvectors as fixed-dimensional features for GMM-BIC
    let featureDim = searchMax
    var spectralPoints = [Float](repeating: 0, count: n * featureDim)
    for i in 0..<n {
        for j in 0..<featureDim {
            spectralPoints[i * featureDim + j] = eigenvectors[j * n + i]
        }
    }

    // Row-normalize spectral points
    for i in 0..<n {
        var norm: Float = 0
        for j in 0..<featureDim {
            norm += spectralPoints[i * featureDim + j] * spectralPoints[i * featureDim + j]
        }
        norm = sqrt(norm + 1e-10)
        for j in 0..<featureDim {
            spectralPoints[i * featureDim + j] /= norm
        }
    }

    // Vary k (number of clusters) while keeping feature dimension fixed
    var bestK = searchMin
    var bestBIC: Float = Float.infinity

    for candidateK in searchMin...searchMax {
        let (bic, _) = gmmBIC(data: spectralPoints, n: n, d: featureDim, k: candidateK)
        if bic < bestBIC {
            bestBIC = bic
            bestK = candidateK
        }
    }

    if bestK == 1 {
        return singleClusterResult(embeddings: embeddings)
    }

    return clusterFromEigenvectors(
        eigenvectors: eigenvectors, eigenvalues: eigenvalues,
        n: n, k: bestK, embeddings: embeddings
    )
}

// MARK: - Helpers

private func singleClusterResult(embeddings: [[Float]]) -> (clusterIds: [Int], centroids: [[Float]]) {
    let ids = [Int](repeating: 0, count: embeddings.count)
    let centroid = computeNormalizedCentroid(embeddings: embeddings, indices: Array(0..<embeddings.count))
    return (ids, [centroid])
}

private func clusterWithK(embeddings: [[Float]], k: Int) -> (clusterIds: [Int], centroids: [[Float]]) {
    let n = embeddings.count
    let affinity = cosineAffinityMatrix(embeddings: embeddings)
    var laplacian = normalizedLaplacian(affinity: affinity, n: n)
    let (eigenvalues, eigenvectors) = eigendecompose(matrix: &laplacian, n: n)
    return clusterFromEigenvectors(
        eigenvectors: eigenvectors, eigenvalues: eigenvalues,
        n: n, k: k, embeddings: embeddings
    )
}

private func clusterFromEigenvectors(
    eigenvectors: [Float], eigenvalues: [Float],
    n: Int, k: Int, embeddings: [[Float]]
) -> (clusterIds: [Int], centroids: [[Float]]) {
    let d = embeddings[0].count

    // Extract top-k eigenvectors (smallest eigenvalues)
    var spectralPoints = [Float](repeating: 0, count: n * k)
    for i in 0..<n {
        for j in 0..<k {
            spectralPoints[i * k + j] = eigenvectors[j * n + i]
        }
    }

    // Row-normalize
    for i in 0..<n {
        var norm: Float = 0
        for j in 0..<k {
            norm += spectralPoints[i * k + j] * spectralPoints[i * k + j]
        }
        norm = sqrt(norm + 1e-10)
        for j in 0..<k {
            spectralPoints[i * k + j] /= norm
        }
    }

    // K-means on spectral representation
    let clusterIds = kMeans(points: spectralPoints, n: n, k: k, dim: k)

    // Compute centroids from original embeddings
    var centroids = [[Float]]()
    for c in 0..<k {
        let indices = clusterIds.enumerated().filter { $0.element == c }.map(\.offset)
        if indices.isEmpty {
            centroids.append([Float](repeating: 0, count: d))
        } else {
            centroids.append(computeNormalizedCentroid(embeddings: embeddings, indices: indices))
        }
    }

    return (clusterIds, centroids)
}

private func computeNormalizedCentroid(embeddings: [[Float]], indices: [Int]) -> [Float] {
    let d = embeddings[0].count
    var centroid = [Float](repeating: 0, count: d)
    for idx in indices {
        for j in 0..<d {
            centroid[j] += embeddings[idx][j]
        }
    }
    let invCount = 1.0 / Float(indices.count)
    for j in 0..<d { centroid[j] *= invCount }

    // L2 normalize
    var norm: Float = 0
    for j in 0..<d { norm += centroid[j] * centroid[j] }
    norm = sqrt(norm + 1e-10)
    for j in 0..<d { centroid[j] /= norm }

    return centroid
}

private func eigengapK(eigenvalues: [Float], minK: Int, maxK: Int) -> Int {
    let n = eigenvalues.count
    guard n > 1 else { return 1 }

    var bestK = minK
    var maxGap: Float = -1

    let limit = min(maxK, n - 1)
    for i in max(minK, 1)..<limit {
        let gap = eigenvalues[i] - eigenvalues[i - 1]
        if gap > maxGap {
            maxGap = gap
            bestK = i
        }
    }

    return max(bestK, minK)
}

// MARK: - Deterministic RNG

private struct SeededRNG: RandomNumberGenerator {
    var state: UInt64

    init(seed: UInt64) {
        state = seed
    }

    mutating func next() -> UInt64 {
        // SplitMix64
        state &+= 0x9e3779b97f4a7c15
        var z = state
        z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9
        z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
        return z ^ (z >> 31)
    }
}
