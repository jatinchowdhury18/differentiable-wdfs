#pragma once

#include "DifferentiableWDFPlugin.h"

class PluginEditor : public AudioProcessorEditor
{
public:
    PluginEditor (DifferentiableWDFPlugin& plugin);

    void paint (Graphics& g) override;

private:
    DifferentiableWDFPlugin& plugin;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (PluginEditor)
};
