#pragma once

#include "../CircuitModel.h"
#include "DiodePairNeuralModel.h"
#include "Toms917DiodePair.h"

class HPFDiodeClipper : public CircuitModel
{
public:
    HPFDiodeClipper (const String& prefix, AudioProcessorValueTreeState& vts);

    static void addParameters (chowdsp::Parameters& params, const String& prefix);

    void prepare (double sampleRate, int samplesPerBlock) override;

    void process (AudioBuffer<float>& buffer) override;

private:
    std::atomic<float>* gainDBParam = nullptr;
    std::atomic<float>* cutoffHzParam = nullptr;
    std::atomic<float>* modelChoiceParam = nullptr;

    dsp::Gain<float> inputGain;

    // WDF
    static constexpr float capVal = 2.2e-9f;

    wdft::ResistiveVoltageSourceT<float> Vs { 1.0f };
    wdft::CapacitorT<float> C { capVal };
    wdft::WDFSeriesT<float, decltype (Vs), decltype (C)> S1 { Vs, C };
    wdft::ResistorT<float> R { 47000.0f };
    wdft::WDFParallelT<float, decltype (R), decltype (S1)> P1 { R, S1 };

    Toms917DiodePairT<float, decltype (P1)> dpToms { P1, 4.352e-9f, 25.85e-3f, 1.906f };
    wdft::DiodePairT<float, decltype (P1)> dpApprox { P1, 4.352e-9f, 25.85e-3f, 1.906f }; // 1N4148
    DiodePairNeuralModel<decltype (P1), 2, 16> dpExtrModel { P1, "_1N4148_1U1D_2x16_training_2000_json" };
    DiodePairNeuralModel<decltype (P1), 2, 16> dpTrainModel { P1, "_1N4148_1U1D_2x16_training_1_hpf_json" };

    int modelChoice = 0;
    int prevModelChoice = 0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (HPFDiodeClipper)
};
