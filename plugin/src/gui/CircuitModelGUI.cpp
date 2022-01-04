#include "CircuitModelGUI.h"

CircuitModelGUI::CircuitModelGUI (const CircuitModel& model, AudioProcessorValueTreeState& vts)
{
    auto addSlider = [=, &vts] (AudioParameterFloat* param)
    {
        auto newSlide = std::make_unique<SliderWithAttachment>();
        addAndMakeVisible (newSlide.get());
        newSlide->attachment = std::make_unique<SliderAttachment> (vts, param->paramID, *newSlide);

        newSlide->setSliderStyle (Slider::LinearVertical);
        newSlide->setName (param->name);
        newSlide->setTextBoxStyle (Slider::TextBoxBelow, false, 80, 20);

        sliders.add (std::move (newSlide));
    };

    for (auto& paramTag : model.getParamTags())
    {
        auto* vtsParam = vts.getParameter (paramTag);

        if (auto* floatParam = dynamic_cast<AudioParameterFloat*> (vtsParam))
            addSlider (floatParam);
    }
}

void CircuitModelGUI::paint (Graphics& g)
{
}

void CircuitModelGUI::resized()
{
    const auto numElements = sliders.size();
    const auto width = getWidth();
    const auto elWidth = proportionOfWidth (1.0f / (float) numElements);

    auto bounds = getLocalBounds();
    for (auto* s : sliders)
    {
        auto sBounds = bounds.removeFromLeft (elWidth).reduced (20);
        s->setBounds (sBounds);
    }
}
