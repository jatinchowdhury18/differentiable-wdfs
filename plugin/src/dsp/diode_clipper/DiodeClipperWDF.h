#pragma once

#include "DiodePairNeuralModel.h"
#include "Toms917DiodePair.h"

class DiodeClipperWDF
{
public:
    DiodeClipperWDF() = default;

    void prepare (double sampleRate);

    void setParameters (float cutoffFreqHz, int modelIndex);

    void process (AudioBuffer<float>& buffer);

private:
    static constexpr float capVal = 2.2e-9f;

    wdft::ResistiveVoltageSourceT<float> Vs { 47000.0f };
    wdft::CapacitorT<float> C { capVal };
    wdft::WDFParallelT<float, decltype (Vs), decltype (C)> P1 { Vs, C };

    Toms917DiodePairT<float, decltype (P1)> dpToms { P1, 4.352e-9f, 25.85e-3f, 1.906f };
    wdft::DiodePairT<float, decltype (P1)> dpApprox { P1, 4.352e-9f, 25.85e-3f, 1.906f }; // 1N4148
    DiodePairNeuralModel<decltype (P1), 4, 8> dp4x8Model { P1, "_1N4148_4x8_training_1_json" };

    int modelChoice = 0;
    int prevModelChoice = 0;
};
