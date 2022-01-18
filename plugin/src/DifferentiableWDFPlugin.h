#pragma once

#include "dsp/diode_clipper/DiodeClipper.h"
#include "dsp/tube_screamer/TubeScreamer.h"

namespace DiffWDFParams
{
const String diodeClipperPrefix = "diode_clipper_";
const String tubeScreamerPrefix = "tube_screamer_";

const String circuitChoiceTag = "circuit_choice";
const StringArray circuitChoices { "Diode Clipper", "Tube Screamer" };
} // namespace DiffWDFParams

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
    auto& getLoadMeasurer() { return loadMeasurer; }

    auto& getDiodeClipper() { return diodeClipper; }
    auto& getTubeScreamer() { return tubeScreamer; }

private:
    std::atomic<float>* modelChoiceParam = nullptr;

    DiodeClipper diodeClipper;
    TubeScreamer tubeScreamer;

    AudioBuffer<float> monoBuffer;

    AudioProcessLoadMeasurer loadMeasurer;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (DifferentiableWDFPlugin)
};
