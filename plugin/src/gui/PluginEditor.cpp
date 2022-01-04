#include "PluginEditor.h"

namespace
{
constexpr float topBarHeight = 0.15f;
}

PluginEditor::PluginEditor (DifferentiableWDFPlugin& p) : AudioProcessorEditor (p),
                                                          plugin (p),
                                                          modelGui (p.getDiodeClipper(), p.getVTS())
{
    setSize (400, 400);

    addAndMakeVisible (circuitModelSelector);
    addAndMakeVisible (modelGui);
}

void PluginEditor::paint (Graphics& g)
{
    g.fillAll (Colours::black);

    auto bounds = getLocalBounds();
    auto topBarBounds = bounds.removeFromTop (proportionOfHeight (topBarHeight));

    g.setColour (Colours::white);
    g.setFont (float (topBarBounds.getHeight()) * 0.45f);
    g.drawLine (Line { topBarBounds.getBottomLeft(), topBarBounds.getBottomRight() }.toFloat(), 2.0f);

    auto nameBounds = topBarBounds.removeFromLeft (proportionOfWidth (0.5f)).reduced (10);
    g.drawFittedText ("Differentiable WDFs", nameBounds, Justification::centred, 1);
}

void PluginEditor::resized()
{
    auto bounds = getLocalBounds();
    auto topBarBounds = bounds.removeFromTop (proportionOfHeight (topBarHeight));

    auto selectorBounds = topBarBounds.removeFromRight (proportionOfWidth (0.5f)).reduced (10);
    circuitModelSelector.setBounds (selectorBounds);
    modelGui.setBounds (bounds);
}
