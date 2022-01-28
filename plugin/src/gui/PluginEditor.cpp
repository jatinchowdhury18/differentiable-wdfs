#include "PluginEditor.h"

namespace
{
constexpr float topBarHeight = 0.15f;
}

PluginEditor::PluginEditor (DifferentiableWDFPlugin& p) : AudioProcessorEditor (p),
                                                          vts (p.getVTS()),
                                                          cpuMeter (p)
{
    modelGui[0] = std::make_unique<CircuitModelGUI> (p.getDiodeClipper(), vts);
    modelGui[1] = std::make_unique<CircuitModelGUI> (p.getMultiDiodeClipper(), vts);
    modelGui[2] = std::make_unique<CircuitModelGUI> (p.getTubeScreamer(), vts);

    addAndMakeVisible (cpuMeter);
    for (auto& gui : modelGui)
        addAndMakeVisible (gui.get());

    addAndMakeVisible (circuitModelSelector);
    circuitModelSelector.addItemList (DiffWDFParams::circuitChoices, 1);
    circuitModelSelectorAttach = std::make_unique<ComboBoxParameterAttachment> (*vts.getParameter (DiffWDFParams::circuitChoiceTag), circuitModelSelector, vts.undoManager);

    vts.addParameterListener (DiffWDFParams::circuitChoiceTag, this);
    parameterChanged (DiffWDFParams::circuitChoiceTag, *vts.getRawParameterValue (DiffWDFParams::circuitChoiceTag));

    setSize (600, 400);
}

PluginEditor::~PluginEditor()
{
    vts.removeParameterListener (DiffWDFParams::circuitChoiceTag, this);
}

void PluginEditor::parameterChanged (const String& paramID, float newValue)
{
    if (paramID != DiffWDFParams::circuitChoiceTag)
        return;

    const auto circuitChoice = (int) newValue;
    for (int i = 0; i < numCircuits; ++i)
        modelGui[i]->setVisible (i == circuitChoice);
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

    topBarBounds.removeFromLeft (proportionOfWidth (0.5f));
    circuitModelSelector.setBounds (topBarBounds.removeFromLeft (proportionOfWidth (0.32f)).reduced (5, 10));
    cpuMeter.setBounds (topBarBounds.reduced (5, 10));

    for (auto& gui : modelGui)
        gui->setBounds (bounds);
}
