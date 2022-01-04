#pragma once

#include "dsp/CircuitModel.h"

class CircuitModelGUI : public Component
{
public:
    CircuitModelGUI (const CircuitModel& model, AudioProcessorValueTreeState& vts);

    void paint (Graphics& g) override;
    void resized() override;

private:
    using SliderAttachment = AudioProcessorValueTreeState::SliderAttachment;
    using ComboBoxAttachment = AudioProcessorValueTreeState::ComboBoxAttachment;
    using ButtonAttachment = AudioProcessorValueTreeState::ButtonAttachment;
    
    struct SliderWithAttachment : public Slider
    {
        std::unique_ptr<SliderAttachment> attachment;
    };

    struct BoxWithAttachment : public ComboBox
    {
        std::unique_ptr<ComboBoxAttachment> attachment;
    };

    struct ButtonWithAttachment : public TextButton
    {
        std::unique_ptr<ButtonAttachment> attachment;
    };

    OwnedArray<Slider> sliders;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (CircuitModelGUI)
};
