#pragma once

#include "../CircuitModel.h"
#include "DiodeClipperWDF.h"

class MultiDiodeClipper : public CircuitModel
{
public:
    MultiDiodeClipper (const String& prefix, AudioProcessorValueTreeState& vts);

    static void addParameters (chowdsp::Parameters& params, const String& prefix);

    void prepare (double sampleRate, int samplesPerBlock) override;

    void process (AudioBuffer<float>& buffer) override;

private:
    std::atomic<float>* gainDBParam = nullptr;
    std::atomic<float>* cutoffHzParam = nullptr;
    std::atomic<float>* modelChoiceParam = nullptr;

    dsp::Gain<float> inputGain;

    DiodeClipperWDF wdf;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MultiDiodeClipper)
};
