#pragma once

#include "DifferentiableWDFPlugin.h"
#include "CircuitModelGUI.h"

class PluginEditor : public AudioProcessorEditor
{
public:
    explicit PluginEditor (DifferentiableWDFPlugin& plugin);

    void paint (Graphics& g) override;
    void resized() override;

private:
    DifferentiableWDFPlugin& plugin;

    ComboBox circuitModelSelector;
    CircuitModelGUI modelGui;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (PluginEditor)
};
