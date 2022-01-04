#pragma once

#include "dsp/diode_clipper/DiodeClipper.h"

namespace DiffWDFParams
{
const String diodeClipperPrefix = "diode_clipper_";
}

class DifferentiableWDFPlugin : public chowdsp::PluginBase<DifferentiableWDFPlugin>
{
public:
    DifferentiableWDFPlugin();

    static void addParameters (Parameters& params);

    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    void processAudioBlock (AudioBuffer<float>& buffer) override;

    AudioProcessorEditor* createEditor() override;

    auto& getVTS() { return vts; }
    auto& getDiodeClipper() { return diodeClipper; }

private:
    DiodeClipper diodeClipper;

    AudioBuffer<float> monoBuffer;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (DifferentiableWDFPlugin)
};
