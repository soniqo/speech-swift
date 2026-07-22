// LocalVQE v1.4 adaptive echo-cancellation front end.
//
// Derived from localai-org/LocalVQE's ggml/daf_frontend.cpp (Apache-2.0).
// The Core ML residual model was trained behind this exact controller and
// partitioned-block Kalman filter; replacing it with a generic subtractor
// materially changes the neural model's inputs.

#include "LocalVQEAECFrontend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <new>
#include <vector>

namespace {

constexpr int kBlock = 128;
constexpr int kPartitions = 128;
constexpr int kFFT = 256;
constexpr int kBins = 129;
constexpr int kIterations = 2;
constexpr float kDecay = 0.999f;
constexpr float kInitialCovariance = 1000.0f;

constexpr int kGCCWindow = 16384;
constexpr int kGCCHop = 8000;
constexpr int kGCCFFT = 32768;
constexpr int kGCCMaxLag = 16384;
constexpr float kGCCGateDB = 26.0f;
constexpr float kGCCConfidenceThreshold = 8.0f;
constexpr int64_t kLockAfter = 56000;
constexpr size_t kControllerWeightCount = 2742;
constexpr double kPi = 3.14159265358979323846264338327950288;

void fftInPlace(std::vector<float>& real, std::vector<float>& imag, bool inverse) {
    const size_t count = real.size();
    for (size_t index = 1, reversed = 0; index < count; ++index) {
        size_t bit = count >> 1;
        for (; reversed & bit; bit >>= 1) {
            reversed ^= bit;
        }
        reversed ^= bit;
        if (index < reversed) {
            std::swap(real[index], real[reversed]);
            std::swap(imag[index], imag[reversed]);
        }
    }

    for (size_t length = 2; length <= count; length <<= 1) {
        const double angle = (inverse ? 2.0 : -2.0) * kPi
            / static_cast<double>(length);
        const double rootReal = std::cos(angle);
        const double rootImag = std::sin(angle);
        for (size_t offset = 0; offset < count; offset += length) {
            double currentReal = 1.0;
            double currentImag = 0.0;
            for (size_t index = 0; index < length / 2; ++index) {
                const float upperReal = real[offset + index];
                const float upperImag = imag[offset + index];
                const size_t lower = offset + index + length / 2;
                const float lowerReal = static_cast<float>(
                    real[lower] * currentReal - imag[lower] * currentImag);
                const float lowerImag = static_cast<float>(
                    real[lower] * currentImag + imag[lower] * currentReal);
                real[offset + index] = upperReal + lowerReal;
                imag[offset + index] = upperImag + lowerImag;
                real[lower] = upperReal - lowerReal;
                imag[lower] = upperImag - lowerImag;

                const double nextReal = currentReal * rootReal
                    - currentImag * rootImag;
                currentImag = currentReal * rootImag + currentImag * rootReal;
                currentReal = nextReal;
            }
        }
    }

    if (inverse) {
        const float scale = 1.0f / static_cast<float>(count);
        for (size_t index = 0; index < count; ++index) {
            real[index] *= scale;
            imag[index] *= scale;
        }
    }
}

void realFFT(const float *input, int count, float *real, float *imag) {
    // These transforms run several times per 8 ms adaptive-filter block.
    // Reuse thread-local capacity instead of allocating on the audio path.
    static thread_local std::vector<float> workReal;
    static thread_local std::vector<float> workImag;
    workReal.assign(input, input + count);
    workImag.assign(static_cast<size_t>(count), 0.0f);
    fftInPlace(workReal, workImag, false);
    for (int bin = 0; bin <= count / 2; ++bin) {
        real[bin] = workReal[bin];
        imag[bin] = workImag[bin];
    }
}

void inverseRealFFT(
    const float *binReal,
    const float *binImag,
    int count,
    float *output
) {
    static thread_local std::vector<float> real;
    static thread_local std::vector<float> imag;
    real.assign(static_cast<size_t>(count), 0.0f);
    imag.assign(static_cast<size_t>(count), 0.0f);
    for (int bin = 0; bin <= count / 2; ++bin) {
        real[bin] = binReal[bin];
        imag[bin] = binImag[bin];
    }
    for (int bin = 1; bin < count / 2; ++bin) {
        real[count - bin] = binReal[bin];
        imag[count - bin] = -binImag[bin];
    }
    fftInPlace(real, imag, true);
    std::copy(real.begin(), real.end(), output);
}

void gruCell(
    const float *input,
    int inputSize,
    float *hidden,
    int hiddenSize,
    const std::vector<float>& inputWeights,
    const std::vector<float>& hiddenWeights,
    const std::vector<float>& inputBias,
    const std::vector<float>& hiddenBias,
    std::vector<float>& inputGates,
    std::vector<float>& hiddenGates
) {
    for (int gate = 0; gate < 3 * hiddenSize; ++gate) {
        float inputSum = inputBias[gate];
        const float *inputRow = inputWeights.data()
            + static_cast<size_t>(gate) * inputSize;
        for (int index = 0; index < inputSize; ++index) {
            inputSum += inputRow[index] * input[index];
        }
        inputGates[gate] = inputSum;

        float hiddenSum = hiddenBias[gate];
        const float *hiddenRow = hiddenWeights.data()
            + static_cast<size_t>(gate) * hiddenSize;
        for (int index = 0; index < hiddenSize; ++index) {
            hiddenSum += hiddenRow[index] * hidden[index];
        }
        hiddenGates[gate] = hiddenSum;
    }

    for (int index = 0; index < hiddenSize; ++index) {
        const float reset = 1.0f / (1.0f + std::exp(
            -(inputGates[index] + hiddenGates[index])));
        const float update = 1.0f / (1.0f + std::exp(
            -(inputGates[hiddenSize + index]
              + hiddenGates[hiddenSize + index])));
        const float candidate = std::tanh(
            inputGates[2 * hiddenSize + index]
            + reset * hiddenGates[2 * hiddenSize + index]);
        hidden[index] = (1.0f - update) * candidate + update * hidden[index];
    }
}

void normalizeFeatures(
    const float *input,
    float *output,
    const std::vector<float>& weight,
    const std::vector<float>& bias,
    int count
) {
    float mean = 0.0f;
    for (int index = 0; index < 6; ++index) {
        mean += input[index];
    }
    mean /= 6.0f;
    float variance = 0.0f;
    for (int index = 0; index < 6; ++index) {
        const float difference = input[index] - mean;
        variance += difference * difference;
    }
    variance /= 6.0f;
    const float inverse = 1.0f / std::sqrt(variance + 1e-5f);
    for (int index = 0; index < 6; ++index) {
        output[index] = (input[index] - mean) * inverse * weight[index]
            + bias[index];
    }
    for (int index = 6; index < count; ++index) {
        output[index] = input[index];
    }
}

inline float safeLog10(float value) {
    return std::log10(value + 1e-10f);
}

} // namespace

struct localvqe_aec_daf {
    bool prealignmentEnabled = true;

    std::vector<float> globalNormWeight, globalNormBias;
    std::vector<float> globalInputWeight, globalHiddenWeight;
    std::vector<float> globalInputBias, globalHiddenBias;
    std::vector<float> binNormWeight, binNormBias;
    std::vector<float> binInputWeight, binHiddenWeight;
    std::vector<float> binInputBias, binHiddenBias;
    std::vector<float> partitionNormWeight, partitionNormBias;
    std::vector<float> partitionInputWeight, partitionHiddenWeight;
    std::vector<float> partitionInputBias, partitionHiddenBias;
    std::vector<float> binHeadWeight, binHeadBias;
    std::vector<float> partitionHeadWeight, partitionHeadBias;

    std::vector<float> filterReal, filterImag;
    std::vector<float> referenceReal, referenceImag;
    std::vector<float> covariance;
    std::vector<float> previousReference;
    int constrainedPartition = 0;

    std::vector<float> globalHidden;
    std::vector<float> binHidden;
    std::vector<float> partitionHidden;

    std::vector<float> gccReal, gccImag;
    std::vector<float> microphoneRing, referenceRing;
    int64_t samplesSeen = 0;
    float maximumReferenceRMS = 1e-12f;
    float delayConfidence = 0.0f;
    int currentDelay = 0;
    bool delayLocked = false;
    std::vector<float> referenceDelayLine;
    size_t referenceDelayPosition = 0;

    bool loadWeights(const float *weights, size_t count) {
        if (weights == nullptr || count != kControllerWeightCount) {
            return false;
        }
        size_t cursor = 0;
        auto take = [&](std::vector<float>& destination, size_t size) {
            destination.assign(weights + cursor, weights + cursor + size);
            cursor += size;
        };

        take(globalNormWeight, 6); take(globalNormBias, 6);
        take(globalInputWeight, 24 * 10); take(globalHiddenWeight, 24 * 8);
        take(globalInputBias, 24); take(globalHiddenBias, 24);
        take(binNormWeight, 6); take(binNormBias, 6);
        take(binInputWeight, 48 * 18); take(binHiddenWeight, 48 * 16);
        take(binInputBias, 48); take(binHiddenBias, 48);
        take(partitionNormWeight, 2); take(partitionNormBias, 2);
        take(partitionInputWeight, 24 * 10);
        take(partitionHiddenWeight, 24 * 8);
        take(partitionInputBias, 24); take(partitionHiddenBias, 24);
        take(binHeadWeight, 16); take(binHeadBias, 1);
        take(partitionHeadWeight, 8); take(partitionHeadBias, 1);
        return cursor == count;
    }

    void reset() {
        auto resetValues = [](
            std::vector<float>& values,
            size_t count,
            float initialValue
        ) {
            if (values.size() != count) {
                values.assign(count, initialValue);
            } else {
                std::fill(values.begin(), values.end(), initialValue);
            }
        };
        const size_t filterCount = static_cast<size_t>(kPartitions) * kBins + 16;
        resetValues(filterReal, filterCount, 0.0f);
        resetValues(filterImag, filterCount, 0.0f);
        resetValues(referenceReal, filterCount, 0.0f);
        resetValues(referenceImag, filterCount, 0.0f);
        resetValues(covariance, filterCount, kInitialCovariance);
        resetValues(previousReference, kBlock, 0.0f);
        constrainedPartition = 0;
        resetValues(globalHidden, 8, 0.0f);
        resetValues(binHidden, static_cast<size_t>(kBins) * 16, 0.0f);
        resetValues(
            partitionHidden, static_cast<size_t>(kPartitions) * 8, 0.0f);
        resetValues(gccReal, kGCCFFT / 2 + 1, 0.0f);
        resetValues(gccImag, kGCCFFT / 2 + 1, 0.0f);
        resetValues(microphoneRing, kGCCWindow, 0.0f);
        resetValues(referenceRing, kGCCWindow, 0.0f);
        samplesSeen = 0;
        maximumReferenceRMS = 1e-12f;
        delayConfidence = 0.0f;
        currentDelay = 0;
        delayLocked = false;
        resetValues(referenceDelayLine, kGCCMaxLag + kBlock, 0.0f);
        referenceDelayPosition = 0;
    }

    void updateDelay() {
        if (delayLocked) {
            return;
        }
        static thread_local std::vector<float> referenceFFTReal(
            kGCCFFT / 2 + 1);
        static thread_local std::vector<float> referenceFFTImag(
            kGCCFFT / 2 + 1);
        static thread_local std::vector<float> microphoneFFTReal(
            kGCCFFT / 2 + 1);
        static thread_local std::vector<float> microphoneFFTImag(
            kGCCFFT / 2 + 1);
        static thread_local std::vector<float> padded(kGCCFFT);
        static thread_local std::vector<float> correlation(kGCCFFT);

        float rms = 0.0f;
        for (float sample : referenceRing) {
            rms += sample * sample;
        }
        rms = std::sqrt(rms / static_cast<float>(kGCCWindow));
        maximumReferenceRMS = std::max(maximumReferenceRMS, rms);
        const bool retain = rms > maximumReferenceRMS
            * std::pow(10.0f, -kGCCGateDB / 20.0f);
        if (retain) {
            std::copy(referenceRing.begin(), referenceRing.end(), padded.begin());
            std::fill(padded.begin() + kGCCWindow, padded.end(), 0.0f);
            realFFT(
                padded.data(), kGCCFFT,
                referenceFFTReal.data(), referenceFFTImag.data());
            std::copy(microphoneRing.begin(), microphoneRing.end(), padded.begin());
            std::fill(padded.begin() + kGCCWindow, padded.end(), 0.0f);
            realFFT(
                padded.data(), kGCCFFT,
                microphoneFFTReal.data(), microphoneFFTImag.data());
            for (int bin = 0; bin <= kGCCFFT / 2; ++bin) {
                const float crossReal = microphoneFFTReal[bin]
                        * referenceFFTReal[bin]
                    + microphoneFFTImag[bin] * referenceFFTImag[bin];
                const float crossImag = microphoneFFTImag[bin]
                        * referenceFFTReal[bin]
                    - microphoneFFTReal[bin] * referenceFFTImag[bin];
                const float magnitude = std::sqrt(
                    crossReal * crossReal + crossImag * crossImag) + 1e-9f;
                gccReal[bin] += crossReal / magnitude;
                gccImag[bin] += crossImag / magnitude;
            }
        }

        inverseRealFFT(gccReal.data(), gccImag.data(), kGCCFFT, correlation.data());
        int bestLag = 0;
        float peak = correlation[0];
        float absoluteSum = 0.0f;
        for (int lag = 0; lag < kGCCMaxLag; ++lag) {
            if (correlation[lag] > peak) {
                peak = correlation[lag];
                bestLag = lag;
            }
            absoluteSum += std::abs(correlation[lag]);
        }
        delayConfidence = peak
            / (absoluteSum / static_cast<float>(kGCCMaxLag) + 1e-12f);
        currentDelay = delayConfidence > kGCCConfidenceThreshold
            ? std::max(0, bestLag - kBlock) / kBlock * kBlock
            : 0;
        if (delayConfidence > kGCCConfidenceThreshold
            && samplesSeen >= kLockAfter) {
            delayLocked = true;
        }
    }

    void processBlock(
        const float *microphone,
        const float *reference,
        float *residual
    ) {
        // One block is only 8 ms. Keep its scratch storage off the allocator
        // after the first call, matching the reference implementation while
        // preserving independent state in each frontend instance.
        static thread_local std::vector<float> fftInput(kFFT);
        static thread_local std::vector<float> currentReferenceReal(kBins);
        static thread_local std::vector<float> currentReferenceImag(kBins);
        static thread_local std::vector<float> echoReal(kBins);
        static thread_local std::vector<float> echoImag(kBins);
        static thread_local std::vector<float> timeBuffer(kFFT);
        static thread_local std::vector<float> errorReal(kBins);
        static thread_local std::vector<float> errorImag(kBins);
        static thread_local std::vector<float> microphoneReal(kBins);
        static thread_local std::vector<float> microphoneImag(kBins);
        static thread_local std::vector<float> referencePower(kBins);
        static thread_local std::vector<float> normalizedFeatures(18);
        static thread_local std::vector<float> globalFeatures(10);
        static thread_local std::vector<float> inputGates(48);
        static thread_local std::vector<float> hiddenGates(48);
        static thread_local std::vector<float> binGain(kBins);
        static thread_local std::vector<float> partitionGain(kPartitions);
        static thread_local std::vector<float> stepSize(
            static_cast<size_t>(kPartitions) * kBins);
        static thread_local std::vector<float> meanCovariance(kBins);
        static thread_local std::vector<float> features(
            static_cast<size_t>(kBins) * 10);

        std::fill(echoReal.begin(), echoReal.end(), 0.0f);
        std::fill(echoImag.begin(), echoImag.end(), 0.0f);
        std::fill(referencePower.begin(), referencePower.end(), 0.0f);
        std::fill(meanCovariance.begin(), meanCovariance.end(), 0.0f);
        std::fill(globalFeatures.begin(), globalFeatures.end(), 0.0f);

        std::copy(previousReference.begin(), previousReference.end(), fftInput.begin());
        std::copy(reference, reference + kBlock, fftInput.begin() + kBlock);
        std::copy(reference, reference + kBlock, previousReference.begin());
        realFFT(
            fftInput.data(), kFFT,
            currentReferenceReal.data(), currentReferenceImag.data());

        std::memmove(
            referenceReal.data() + kBins,
            referenceReal.data(),
            static_cast<size_t>(kPartitions - 1) * kBins * sizeof(float));
        std::memmove(
            referenceImag.data() + kBins,
            referenceImag.data(),
            static_cast<size_t>(kPartitions - 1) * kBins * sizeof(float));
        std::copy(
            currentReferenceReal.begin(), currentReferenceReal.end(),
            referenceReal.begin());
        std::copy(
            currentReferenceImag.begin(), currentReferenceImag.end(),
            referenceImag.begin());

        for (int partition = 0; partition < kPartitions; ++partition) {
            const size_t base = static_cast<size_t>(partition) * kBins;
            for (int bin = 0; bin < kBins; ++bin) {
                const size_t index = base + bin;
                echoReal[bin] += filterReal[index] * referenceReal[index]
                    - filterImag[index] * referenceImag[index];
                echoImag[bin] += filterReal[index] * referenceImag[index]
                    + filterImag[index] * referenceReal[index];
            }
        }
        inverseRealFFT(echoReal.data(), echoImag.data(), kFFT, timeBuffer.data());
        for (int index = 0; index < kBlock; ++index) {
            residual[index] = microphone[index] - timeBuffer[kBlock + index];
        }

        std::fill(fftInput.begin(), fftInput.begin() + kBlock, 0.0f);
        std::copy(residual, residual + kBlock, fftInput.begin() + kBlock);
        realFFT(fftInput.data(), kFFT, errorReal.data(), errorImag.data());
        std::copy(microphone, microphone + kBlock, fftInput.begin() + kBlock);
        realFFT(
            fftInput.data(), kFFT,
            microphoneReal.data(), microphoneImag.data());

        for (int partition = 0; partition < kPartitions; ++partition) {
            const size_t base = static_cast<size_t>(partition) * kBins;
            for (int bin = 0; bin < kBins; ++bin) {
                const float real = referenceReal[base + bin];
                const float imag = referenceImag[base + bin];
                referencePower[bin] += real * real + imag * imag;
                meanCovariance[bin] += covariance[base + bin];
            }
        }
        for (int bin = 0; bin < kBins; ++bin) {
            meanCovariance[bin] /= static_cast<float>(kPartitions);
            const float referenceMagnitude = std::sqrt(
                currentReferenceReal[bin] * currentReferenceReal[bin]
                + currentReferenceImag[bin] * currentReferenceImag[bin]);
            const float errorMagnitude = std::sqrt(
                errorReal[bin] * errorReal[bin]
                + errorImag[bin] * errorImag[bin]);
            const float echoMagnitude = std::sqrt(
                echoReal[bin] * echoReal[bin]
                + echoImag[bin] * echoImag[bin]);
            const float microphoneMagnitude = std::sqrt(
                microphoneReal[bin] * microphoneReal[bin]
                + microphoneImag[bin] * microphoneImag[bin]);
            float *feature = features.data() + static_cast<size_t>(bin) * 10;
            feature[0] = safeLog10(referenceMagnitude * referenceMagnitude);
            feature[1] = safeLog10(referencePower[bin]);
            feature[2] = safeLog10(microphoneMagnitude * microphoneMagnitude);
            feature[3] = safeLog10(errorMagnitude * errorMagnitude);
            feature[4] = safeLog10(echoMagnitude * echoMagnitude);
            feature[5] = safeLog10(meanCovariance[bin]);
            constexpr float epsilon = 1e-9f;
            feature[6] = (errorReal[bin] * microphoneReal[bin]
                          + errorImag[bin] * microphoneImag[bin])
                / (errorMagnitude * microphoneMagnitude + epsilon);
            feature[7] = (microphoneReal[bin] * echoReal[bin]
                          + microphoneImag[bin] * echoImag[bin])
                / (microphoneMagnitude * echoMagnitude + epsilon);
            feature[8] = (errorReal[bin] * echoReal[bin]
                          + errorImag[bin] * echoImag[bin])
                / (errorMagnitude * echoMagnitude + epsilon);
            feature[9] = (currentReferenceReal[bin] * microphoneReal[bin]
                          + currentReferenceImag[bin] * microphoneImag[bin])
                / (referenceMagnitude * microphoneMagnitude + epsilon);
            for (int index = 0; index < 10; ++index) {
                globalFeatures[index] += feature[index];
            }
        }

        for (float& value : globalFeatures) {
            value /= static_cast<float>(kBins);
        }
        normalizeFeatures(
            globalFeatures.data(), normalizedFeatures.data(),
            globalNormWeight, globalNormBias, 10);
        gruCell(
            normalizedFeatures.data(), 10, globalHidden.data(), 8,
            globalInputWeight, globalHiddenWeight,
            globalInputBias, globalHiddenBias,
            inputGates, hiddenGates);

        for (int bin = 0; bin < kBins; ++bin) {
            normalizeFeatures(
                features.data() + static_cast<size_t>(bin) * 10,
                normalizedFeatures.data(), binNormWeight, binNormBias, 10);
            std::copy(
                globalHidden.begin(), globalHidden.end(),
                normalizedFeatures.begin() + 10);
            gruCell(
                normalizedFeatures.data(), 18,
                binHidden.data() + static_cast<size_t>(bin) * 16, 16,
                binInputWeight, binHiddenWeight, binInputBias, binHiddenBias,
                inputGates, hiddenGates);
            float logit = binHeadBias[0];
            for (int index = 0; index < 16; ++index) {
                logit += binHeadWeight[index]
                    * binHidden[static_cast<size_t>(bin) * 16 + index];
            }
            binGain[bin] = 2.0f / (1.0f + std::exp(-logit));
        }

        for (int partition = 0; partition < kPartitions; ++partition) {
            const size_t base = static_cast<size_t>(partition) * kBins;
            float filterPower = 0.0f;
            float covarianceMean = 0.0f;
            for (int bin = 0; bin < kBins; ++bin) {
                filterPower += filterReal[base + bin] * filterReal[base + bin]
                    + filterImag[base + bin] * filterImag[base + bin];
                covarianceMean += covariance[base + bin];
            }
            float pair[2] = {
                safeLog10(filterPower / kBins),
                safeLog10(covarianceMean / kBins),
            };
            const float mean = 0.5f * (pair[0] + pair[1]);
            const float variance = 0.5f
                * ((pair[0] - mean) * (pair[0] - mean)
                   + (pair[1] - mean) * (pair[1] - mean));
            const float inverse = 1.0f / std::sqrt(variance + 1e-5f);
            float partitionInput[10];
            partitionInput[0] = (pair[0] - mean) * inverse
                    * partitionNormWeight[0]
                + partitionNormBias[0];
            partitionInput[1] = (pair[1] - mean) * inverse
                    * partitionNormWeight[1]
                + partitionNormBias[1];
            std::copy(
                globalHidden.begin(), globalHidden.end(), partitionInput + 2);
            gruCell(
                partitionInput, 10,
                partitionHidden.data() + static_cast<size_t>(partition) * 8, 8,
                partitionInputWeight, partitionHiddenWeight,
                partitionInputBias, partitionHiddenBias,
                inputGates, hiddenGates);
            float logit = partitionHeadBias[0];
            for (int index = 0; index < 8; ++index) {
                logit += partitionHeadWeight[index]
                    * partitionHidden[static_cast<size_t>(partition) * 8 + index];
            }
            partitionGain[partition] = 2.0f / (1.0f + std::exp(-logit));
        }

        const float squaredDecay = kDecay * kDecay;
        for (int partition = 0; partition < kPartitions; ++partition) {
            const size_t base = static_cast<size_t>(partition) * kBins;
            for (int bin = 0; bin < kBins; ++bin) {
                const size_t index = base + bin;
                const float errorPower = (
                    errorReal[bin] * errorReal[bin]
                    + errorImag[bin] * errorImag[bin]) / kPartitions;
                const float posterior = 0.5f * covariance[index]
                        * referencePower[bin]
                    + errorPower;
                float update = covariance[index] / (posterior + 1e-10f);
                update *= binGain[bin] * partitionGain[partition];
                const float normalizedUpdate = update * referencePower[bin];
                if (normalizedUpdate > 2.0f) {
                    update *= 2.0f / normalizedUpdate;
                }
                const float factor = std::max(
                    1.0f - 0.5f * update * referencePower[bin], 0.0f);
                covariance[index] = std::max(
                    squaredDecay * factor * covariance[index]
                    + (1.0f - squaredDecay)
                        * (filterReal[index] * filterReal[index]
                           + filterImag[index] * filterImag[index]),
                    1e-12f);
                stepSize[index] = update;
            }
        }

        static thread_local std::vector<float> refreshedErrorReal(kBins);
        static thread_local std::vector<float> refreshedErrorImag(kBins);
        for (int iteration = 0; iteration < kIterations; ++iteration) {
            const float *activeErrorReal = errorReal.data();
            const float *activeErrorImag = errorImag.data();
            if (iteration > 0) {
                std::fill(echoReal.begin(), echoReal.end(), 0.0f);
                std::fill(echoImag.begin(), echoImag.end(), 0.0f);
                for (int partition = 0; partition < kPartitions; ++partition) {
                    const size_t base = static_cast<size_t>(partition) * kBins;
                    for (int bin = 0; bin < kBins; ++bin) {
                        const size_t index = base + bin;
                        echoReal[bin] += filterReal[index] * referenceReal[index]
                            - filterImag[index] * referenceImag[index];
                        echoImag[bin] += filterReal[index] * referenceImag[index]
                            + filterImag[index] * referenceReal[index];
                    }
                }
                inverseRealFFT(
                    echoReal.data(), echoImag.data(), kFFT, timeBuffer.data());
                std::fill(fftInput.begin(), fftInput.begin() + kBlock, 0.0f);
                for (int index = 0; index < kBlock; ++index) {
                    fftInput[kBlock + index] = microphone[index]
                        - timeBuffer[kBlock + index];
                }
                realFFT(
                    fftInput.data(), kFFT,
                    refreshedErrorReal.data(), refreshedErrorImag.data());
                activeErrorReal = refreshedErrorReal.data();
                activeErrorImag = refreshedErrorImag.data();
            }
            for (int partition = 0; partition < kPartitions; ++partition) {
                const size_t base = static_cast<size_t>(partition) * kBins;
                for (int bin = 0; bin < kBins; ++bin) {
                    const size_t index = base + bin;
                    filterReal[index] += stepSize[index]
                        * (activeErrorReal[bin] * referenceReal[index]
                           + activeErrorImag[bin] * referenceImag[index]);
                    filterImag[index] += stepSize[index]
                        * (activeErrorImag[bin] * referenceReal[index]
                           - activeErrorReal[bin] * referenceImag[index]);
                }
            }
        }

        const size_t constrainedBase = static_cast<size_t>(constrainedPartition)
            * kBins;
        inverseRealFFT(
            filterReal.data() + constrainedBase,
            filterImag.data() + constrainedBase,
            kFFT,
            timeBuffer.data());
        std::fill(timeBuffer.begin() + kBlock, timeBuffer.end(), 0.0f);
        realFFT(
            timeBuffer.data(), kFFT,
            filterReal.data() + constrainedBase,
            filterImag.data() + constrainedBase);
        constrainedPartition = (constrainedPartition + 1) % kPartitions;
    }

    void process(
        const float *microphone,
        const float *reference,
        size_t sampleCount,
        float *residual,
        float *echoEstimate
    ) {
        const int delayLineCount = static_cast<int>(referenceDelayLine.size());
        for (size_t offset = 0; offset < sampleCount; offset += kBlock) {
            std::memmove(
                microphoneRing.data(), microphoneRing.data() + kBlock,
                (kGCCWindow - kBlock) * sizeof(float));
            std::memcpy(
                microphoneRing.data() + kGCCWindow - kBlock,
                microphone + offset,
                kBlock * sizeof(float));
            std::memmove(
                referenceRing.data(), referenceRing.data() + kBlock,
                (kGCCWindow - kBlock) * sizeof(float));
            std::memcpy(
                referenceRing.data() + kGCCWindow - kBlock,
                reference + offset,
                kBlock * sizeof(float));
            for (int index = 0; index < kBlock; ++index) {
                referenceDelayLine[(referenceDelayPosition + index)
                    % delayLineCount] = reference[offset + index];
            }
            samplesSeen += kBlock;
            if (prealignmentEnabled && samplesSeen >= kGCCWindow
                && ((samplesSeen - kGCCWindow) % kGCCHop) < kBlock) {
                updateDelay();
            }

            float shiftedReference[kBlock];
            for (int index = 0; index < kBlock; ++index) {
                const int64_t delayedIndex = static_cast<int64_t>(
                    referenceDelayPosition) + index - currentDelay;
                const int64_t sourcePosition = samplesSeen - kBlock + index;
                const int wrapped = static_cast<int>(
                    (delayedIndex % delayLineCount + delayLineCount)
                    % delayLineCount);
                shiftedReference[index] = sourcePosition >= currentDelay
                    ? referenceDelayLine[wrapped]
                    : 0.0f;
            }
            referenceDelayPosition = (referenceDelayPosition + kBlock)
                % delayLineCount;
            processBlock(
                microphone + offset, shiftedReference, residual + offset);
        }
        for (size_t index = 0; index < sampleCount; ++index) {
            echoEstimate[index] = microphone[index] - residual[index];
        }
    }

    void primeDelay(
        const float *microphone,
        const float *reference,
        size_t sampleCount
    ) {
        reset();
        int bestDelay = 0;
        float bestConfidence = 0.0f;
        for (size_t offset = 0; offset + kBlock <= sampleCount;
             offset += kBlock) {
            std::memmove(
                microphoneRing.data(), microphoneRing.data() + kBlock,
                (kGCCWindow - kBlock) * sizeof(float));
            std::memcpy(
                microphoneRing.data() + kGCCWindow - kBlock,
                microphone + offset,
                kBlock * sizeof(float));
            std::memmove(
                referenceRing.data(), referenceRing.data() + kBlock,
                (kGCCWindow - kBlock) * sizeof(float));
            std::memcpy(
                referenceRing.data() + kGCCWindow - kBlock,
                reference + offset,
                kBlock * sizeof(float));
            samplesSeen += kBlock;
            if (samplesSeen >= kGCCWindow
                && ((samplesSeen - kGCCWindow) % kGCCHop) < kBlock) {
                updateDelay();
                delayLocked = false;
                if (delayConfidence > bestConfidence) {
                    bestConfidence = delayConfidence;
                    bestDelay = currentDelay;
                }
            }
        }
        reset();
        if (bestConfidence > kGCCConfidenceThreshold) {
            currentDelay = bestDelay;
            delayConfidence = bestConfidence;
            delayLocked = true;
        }
    }
};

size_t localvqe_aec_daf_weight_count(void) {
    return kControllerWeightCount;
}

localvqe_aec_daf *localvqe_aec_daf_create(
    const float *weights,
    size_t count
) {
    auto *frontend = new (std::nothrow) localvqe_aec_daf();
    if (frontend == nullptr) {
        return nullptr;
    }
    try {
        if (!frontend->loadWeights(weights, count)) {
            delete frontend;
            return nullptr;
        }
        frontend->reset();
        return frontend;
    } catch (...) {
        delete frontend;
        return nullptr;
    }
}

void localvqe_aec_daf_destroy(localvqe_aec_daf *frontend) {
    delete frontend;
}

void localvqe_aec_daf_reset(localvqe_aec_daf *frontend) {
    if (frontend != nullptr) {
        try {
            frontend->reset();
        } catch (...) {
            // Reset normally only fills storage allocated at creation. Never
            // allow an allocation failure to cross the C ABI into Swift.
        }
    }
}

void localvqe_aec_daf_set_prealignment(
    localvqe_aec_daf *frontend,
    bool enabled
) {
    if (frontend != nullptr) {
        frontend->prealignmentEnabled = enabled;
    }
}

bool localvqe_aec_daf_prime_delay(
    localvqe_aec_daf *frontend,
    const float *microphone,
    const float *reference,
    size_t sample_count
) {
    if (frontend == nullptr || microphone == nullptr || reference == nullptr) {
        return false;
    }
    try {
        frontend->primeDelay(microphone, reference, sample_count);
        return true;
    } catch (...) {
        return false;
    }
}

bool localvqe_aec_daf_process(
    localvqe_aec_daf *frontend,
    const float *microphone,
    const float *reference,
    size_t sample_count,
    float *residual,
    float *echo_estimate
) {
    if (frontend == nullptr || microphone == nullptr || reference == nullptr
        || residual == nullptr || echo_estimate == nullptr
        || sample_count == 0 || sample_count % kBlock != 0) {
        return false;
    }
    try {
        frontend->process(
            microphone, reference, sample_count, residual, echo_estimate);
        return true;
    } catch (...) {
        return false;
    }
}

int32_t localvqe_aec_daf_current_delay_samples(
    const localvqe_aec_daf *frontend
) {
    return frontend == nullptr ? 0 : frontend->currentDelay;
}

float localvqe_aec_daf_delay_confidence(
    const localvqe_aec_daf *frontend
) {
    return frontend == nullptr ? 0.0f : frontend->delayConfidence;
}
