#include "CircuitModelGUI.h"

namespace
{
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

constexpr int nameHeight = 25;
} // namespace

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

    auto addBox = [=, &vts] (AudioParameterChoice* param)
    {
        auto newBox = std::make_unique<BoxWithAttachment>();
        addAndMakeVisible (newBox.get());
        newBox->setName (param->name);
        newBox->addItemList (param->choices, 1);
        newBox->setSelectedItemIndex (0);

        newBox->attachment = std::make_unique<ComboBoxAttachment> (vts, param->paramID, *newBox);

        boxes.add (std::move (newBox));
    };

    for (auto& paramTag : model.getParamTags())
    {
        auto* vtsParam = vts.getParameter (paramTag);

        if (auto* floatParam = dynamic_cast<AudioParameterFloat*> (vtsParam))
            addSlider (floatParam);

        else if (auto* paramChoice = dynamic_cast<AudioParameterChoice*> (vtsParam))
            addBox (paramChoice);
    }
}

void CircuitModelGUI::paint (Graphics& g)
{
    g.setColour (Colours::white);

    auto makeName = [&g] (Component& comp, const String& name)
    {
        const auto textWidth = comp.proportionOfWidth (0.8f);
        if (textWidth == 0)
            return;

        auto font = Font ((float) nameHeight - 2.0f).boldened();
        while (font.getStringWidth (name) > textWidth)
            font = Font (font.getHeight() - 0.5f).boldened();

        g.setFont (font);
        Rectangle<int> nameBox (comp.getX(), comp.getY() - nameHeight, comp.getWidth(), nameHeight);
        g.drawFittedText (name, nameBox, Justification::centred, 1);
    };

    for (auto* s : sliders)
        makeName (*s, s->getName());

    for (auto* b : boxes)
        makeName (*b, b->getName());
}

void CircuitModelGUI::resized()
{
    const auto numElements = sliders.size() + boxes.size();
    const auto elWidth = proportionOfWidth (1.0f / (float) numElements);

    auto bounds = getLocalBounds();
    bounds.removeFromTop (nameHeight);
    for (auto* s : sliders)
    {
        auto sBounds = bounds.removeFromLeft (elWidth).reduced (20);
        s->setBounds (sBounds);
    }

    for (auto* b : boxes)
    {
        auto bBounds = bounds.removeFromLeft (elWidth).reduced (20, (getHeight() - 75) / 2);
        b->setBounds (bBounds);
    }
}
