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

    // WDF...
    static constexpr float capVal = 2.2e-9f;

    wdft::ResistiveVoltageSourceT<float> Vs { 47000.0f };
    wdft::CapacitorT<float> C { capVal };
    wdft::WDFParallelT<float, decltype (Vs), decltype (C)> P1 { Vs, C };

    wdft::DiodePairT<float, decltype (P1)> dp { P1, 4.352e-9f, 25.85e-3f, 1.906f }; // 1N4148

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (DiodeClipper)
};
