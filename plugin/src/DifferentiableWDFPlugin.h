#pragma once

#include "dsp/diode_clipper/DiodeClipper.h"
#include "dsp/diode_clipper/HPFDiodeClipper.h"
#include "dsp/diode_clipper/MultiDiodeClipper.h"
#include "dsp/tube_screamer/TubeScreamer.h"

namespace DiffWDFParams
{
const String diodeClipperPrefix = "diode_clipper_";
const String multiDiodeClipperPrefix = "multi_diode_clipper_";
const String tubeScreamerPrefix = "tube_screamer_";

const String circuitChoiceTag = "circuit_choice";
const StringArray circuitChoices { "Diode Clipper", "Multi Diode Clipper", "Tube Screamer Clipping Stage" };
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
    auto& getMultiDiodeClipper() { return multiDiodeClipper; }
    auto& getTubeScreamer() { return tubeScreamer; }

private:
    std::atomic<float>* modelChoiceParam = nullptr;

    DiodeClipper diodeClipper;
    MultiDiodeClipper multiDiodeClipper;
    TubeScreamer tubeScreamer;

    AudioBuffer<float> monoBuffer;

    chowdsp::FirstOrderHPF<float> dcBlocker;

    AudioProcessLoadMeasurer loadMeasurer;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (DifferentiableWDFPlugin)
};
