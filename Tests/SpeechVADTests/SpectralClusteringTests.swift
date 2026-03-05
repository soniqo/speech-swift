import XCTest
@testable import SpeechVAD

final class SpectralClusteringTests: XCTestCase {

    // MARK: - Affinity Matrix

    func testAffinityMatrixIdentical() {
        let emb: [Float] = [Float](repeating: 0.5, count: 4)
        let embeddings = [[Float]](repeating: emb, count: 3)
        let A = cosineAffinityMatrix(embeddings: embeddings)

        XCTAssertEqual(A.count, 9)
        // Diagonal should be 0
        XCTAssertEqual(A[0], 0, accuracy: 1e-5)
        XCTAssertEqual(A[4], 0, accuracy: 1e-5)
        XCTAssertEqual(A[8], 0, accuracy: 1e-5)
        // Off-diagonal: identical vectors → cosine=1 → affinity=(1+1)/2=1.0
        XCTAssertEqual(A[1], 1.0, accuracy: 1e-4)
        XCTAssertEqual(A[3], 1.0, accuracy: 1e-4)
        XCTAssertEqual(A[5], 1.0, accuracy: 1e-4)
    }

    func testAffinityMatrixOrthogonal() {
        let e1: [Float] = [1, 0, 0, 0]
        let e2: [Float] = [0, 1, 0, 0]
        let A = cosineAffinityMatrix(embeddings: [e1, e2])

        XCTAssertEqual(A.count, 4)
        // Orthogonal → cosine=0 → affinity=(1+0)/2=0.5
        XCTAssertEqual(A[1], 0.5, accuracy: 1e-4)
        XCTAssertEqual(A[2], 0.5, accuracy: 1e-4)
        // Diagonal = 0
        XCTAssertEqual(A[0], 0, accuracy: 1e-5)
        XCTAssertEqual(A[3], 0, accuracy: 1e-5)
    }

    // MARK: - Laplacian

    func testLaplacianSymmetric() {
        let e1: [Float] = [1, 0, 0]
        let e2: [Float] = [0, 1, 0]
        let e3: [Float] = [0.7, 0.7, 0]
        let A = cosineAffinityMatrix(embeddings: [e1, e2, e3])
        let L = normalizedLaplacian(affinity: A, n: 3)

        // L_sym should be symmetric
        for i in 0..<3 {
            for j in 0..<3 {
                XCTAssertEqual(L[i * 3 + j], L[j * 3 + i], accuracy: 1e-4,
                               "L[\(i),\(j)] != L[\(j),\(i)]")
            }
        }
    }

    // MARK: - Eigendecomposition

    func testEigendecompose3x3() {
        // Diagonal matrix with known eigenvalues
        var M: [Float] = [
            3, 0, 0,
            0, 1, 0,
            0, 0, 2
        ]
        let (eigenvalues, _) = eigendecompose(matrix: &M, n: 3)

        // Eigenvalues should be sorted ascending: 1, 2, 3
        XCTAssertEqual(eigenvalues[0], 1.0, accuracy: 1e-4)
        XCTAssertEqual(eigenvalues[1], 2.0, accuracy: 1e-4)
        XCTAssertEqual(eigenvalues[2], 3.0, accuracy: 1e-4)
    }

    // MARK: - K-Means

    func testKMeans2Clusters() {
        // Two well-separated 2D clusters
        var points = [Float]()
        // Cluster A: around (0, 0)
        for _ in 0..<20 {
            points.append(Float.random(in: -0.1...0.1))
            points.append(Float.random(in: -0.1...0.1))
        }
        // Cluster B: around (10, 10)
        for _ in 0..<20 {
            points.append(10.0 + Float.random(in: -0.1...0.1))
            points.append(10.0 + Float.random(in: -0.1...0.1))
        }

        let assignments = kMeans(points: points, n: 40, k: 2, dim: 2)

        // All first 20 should be in same cluster, all last 20 in other cluster
        let cluster0 = assignments[0]
        for i in 0..<20 {
            XCTAssertEqual(assignments[i], cluster0, "Point \(i) should be in cluster \(cluster0)")
        }
        let cluster1 = assignments[20]
        XCTAssertNotEqual(cluster0, cluster1, "Two clusters should have different IDs")
        for i in 20..<40 {
            XCTAssertEqual(assignments[i], cluster1, "Point \(i) should be in cluster \(cluster1)")
        }
    }

    // MARK: - GMM-BIC

    func testGMMBICSelectsCorrectK() {
        // Generate two well-separated Gaussian clusters in 4D
        var data = [Float]()
        let n = 40
        // Cluster A: around origin
        for _ in 0..<(n / 2) {
            for _ in 0..<4 {
                data.append(Float.random(in: -0.5...0.5))
            }
        }
        // Cluster B: around (10,10,10,10)
        for _ in 0..<(n / 2) {
            for _ in 0..<4 {
                data.append(10.0 + Float.random(in: -0.5...0.5))
            }
        }

        let (bic1, _) = gmmBIC(data: data, n: n, d: 4, k: 1)
        let (bic2, _) = gmmBIC(data: data, n: n, d: 4, k: 2)
        let (bic3, _) = gmmBIC(data: data, n: n, d: 4, k: 3)

        // k=2 should have lowest BIC for 2-cluster data
        XCTAssertLessThan(bic2, bic1, "BIC(k=2) should be lower than BIC(k=1)")
        XCTAssertLessThan(bic2, bic3, "BIC(k=2) should be lower than BIC(k=3)")
    }

    // MARK: - Spectral Clustering

    func testSpectralClustering2Groups() {
        // Two groups of similar 256-dim embeddings
        var group1Base = [Float](repeating: 0, count: 256)
        group1Base[0] = 1.0  // unit vector along dim 0
        var group2Base = [Float](repeating: 0, count: 256)
        group2Base[1] = 1.0  // unit vector along dim 1

        var embeddings = [[Float]]()
        // Add small perturbations
        for _ in 0..<10 {
            var e = group1Base
            for j in 0..<256 { e[j] += Float.random(in: -0.01...0.01) }
            embeddings.append(e)
        }
        for _ in 0..<10 {
            var e = group2Base
            for j in 0..<256 { e[j] += Float.random(in: -0.01...0.01) }
            embeddings.append(e)
        }

        let (clusterIds, centroids) = spectralClustering(embeddings: embeddings)

        XCTAssertEqual(clusterIds.count, 20)
        XCTAssertEqual(centroids.count, 2)

        // First 10 should all be same cluster, last 10 should be different
        let c0 = clusterIds[0]
        for i in 0..<10 {
            XCTAssertEqual(clusterIds[i], c0, "Group 1 point \(i) should be in cluster \(c0)")
        }
        let c1 = clusterIds[10]
        XCTAssertNotEqual(c0, c1)
        for i in 10..<20 {
            XCTAssertEqual(clusterIds[i], c1, "Group 2 point \(i) should be in cluster \(c1)")
        }
    }

    func testSpectralClustering1Group() {
        // All similar embeddings → should detect 1 cluster
        let base = [Float](repeating: 0.5, count: 32)
        var embeddings = [[Float]]()
        for _ in 0..<8 {
            var e = base
            for j in 0..<32 { e[j] += Float.random(in: -0.01...0.01) }
            embeddings.append(e)
        }

        let (clusterIds, centroids) = spectralClustering(embeddings: embeddings)

        XCTAssertEqual(clusterIds.count, 8)
        XCTAssertEqual(centroids.count, 1, "All-similar embeddings should produce 1 cluster")
        // All same cluster
        for id in clusterIds {
            XCTAssertEqual(id, 0)
        }
    }

    // MARK: - Min/Max Constraints

    func testMinMaxConstraints() {
        // All similar embeddings but force minClusters=2
        let base = [Float](repeating: 0.5, count: 32)
        var embeddings = [[Float]]()
        for _ in 0..<8 {
            var e = base
            for j in 0..<32 { e[j] += Float.random(in: -0.01...0.01) }
            embeddings.append(e)
        }

        let (ids2, centroids2) = spectralClustering(embeddings: embeddings, minClusters: 2)
        XCTAssertGreaterThanOrEqual(Set(ids2).count, 2, "minClusters=2 should produce at least 2 clusters")
        XCTAssertGreaterThanOrEqual(centroids2.count, 2)

        // Two distinct groups but maxClusters=1
        var twoGroupEmbs = [[Float]]()
        for _ in 0..<5 {
            var e = [Float](repeating: 0, count: 32)
            e[0] = 1.0
            for j in 0..<32 { e[j] += Float.random(in: -0.01...0.01) }
            twoGroupEmbs.append(e)
        }
        for _ in 0..<5 {
            var e = [Float](repeating: 0, count: 32)
            e[1] = 1.0
            for j in 0..<32 { e[j] += Float.random(in: -0.01...0.01) }
            twoGroupEmbs.append(e)
        }

        let (ids1, centroids1) = spectralClustering(embeddings: twoGroupEmbs, maxClusters: 1)
        XCTAssertEqual(Set(ids1).count, 1, "maxClusters=1 should force 1 cluster")
        XCTAssertEqual(centroids1.count, 1)
    }

    // MARK: - Edge Cases

    func testEdgeCasesN0N1N2() {
        // N=0
        let (ids0, c0) = spectralClustering(embeddings: [])
        XCTAssertTrue(ids0.isEmpty)
        XCTAssertTrue(c0.isEmpty)

        // N=1
        let single: [[Float]] = [[1, 0, 0, 0]]
        let (ids1, c1) = spectralClustering(embeddings: single)
        XCTAssertEqual(ids1, [0])
        XCTAssertEqual(c1.count, 1)

        // N=2 identical
        let pair: [[Float]] = [[1, 0, 0], [1, 0, 0]]
        let (ids2, c2) = spectralClustering(embeddings: pair)
        XCTAssertEqual(ids2.count, 2)
        XCTAssertGreaterThanOrEqual(c2.count, 1)
    }

    // MARK: - Config

    func testConfigMinSpeakers() {
        let config = DiarizationConfig.default
        XCTAssertEqual(config.minSpeakers, 0)
        XCTAssertEqual(config.maxSpeakers, 0)

        let custom = DiarizationConfig(minSpeakers: 2, maxSpeakers: 5)
        XCTAssertEqual(custom.minSpeakers, 2)
        XCTAssertEqual(custom.maxSpeakers, 5)
    }
}
