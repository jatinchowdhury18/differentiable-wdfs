#pragma once

#include "../CircuitModel.h"

class DiodeClipper : public CircuitModel
{
public:
    DiodeClipper (const String& prefix, AudioProcessorValueTreeState& vts);

    static void addParameters (chowdsp::Parameters& params, const String& prefix);

    void prepare (double sampleRate, int samplesPerBlock) override;

    void process (AudioBuffer<float>& buffer) override;

private:
    std::atomic<float>* gainDBParam = nullptr;
    std::atomic<float>* cutoffHzParam = nullptr;

    dsp::Gain<float> inputGain;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (DiodeClipper)
};
