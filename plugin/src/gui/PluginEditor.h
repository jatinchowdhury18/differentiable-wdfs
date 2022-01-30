#pragma once

#include "CPUMeter.h"
#include "CircuitModelGUI.h"

class PluginEditor : public AudioProcessorEditor,
                     private AudioProcessorValueTreeState::Listener
{
public:
    explicit PluginEditor (DifferentiableWDFPlugin& plugin);
    ~PluginEditor() override;

    void paint (Graphics& g) override;
    void resized() override;

    void parameterChanged (const String& paramID, float newValue) final;

private:
    AudioProcessorValueTreeState& vts;

    ComboBox circuitModelSelector;
    std::unique_ptr<ComboBoxParameterAttachment> circuitModelSelectorAttach;

    static constexpr int numCircuits = 4;
    std::unique_ptr<CircuitModelGUI> modelGui[numCircuits];
    CPUMeter cpuMeter;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (PluginEditor)
};
