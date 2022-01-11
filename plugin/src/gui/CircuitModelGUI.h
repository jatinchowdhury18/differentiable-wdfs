#pragma once

#include "dsp/CircuitModel.h"

class CircuitModelGUI : public Component
{
public:
    CircuitModelGUI (const CircuitModel& model, AudioProcessorValueTreeState& vts);

    void paint (Graphics& g) override;
    void resized() override;

private:
    OwnedArray<Slider> sliders;
    OwnedArray<ComboBox> boxes;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (CircuitModelGUI)
};
