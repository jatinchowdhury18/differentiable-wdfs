#pragma once

#include <pch.h>

template <int NLayers, int HiddenSize>
struct DiodeModelType;

template <int HiddenSize>
struct DiodeModelType<2, HiddenSize>
{
    using ModelType = RTNeural::ModelT<float,
                                       2,
                                       1,
                                       RTNeural::DenseT<float, 2, HiddenSize>,          // Input layer
                                       RTNeural::TanhActivationT<float, HiddenSize>,
                                       RTNeural::DenseT<float, HiddenSize, HiddenSize>, // Hidden layer 1
                                       RTNeural::TanhActivationT<float, HiddenSize>,
                                       RTNeural::DenseT<float, HiddenSize, HiddenSize>, // Hidden layer 2
                                       RTNeural::TanhActivationT<float, HiddenSize>,
                                       RTNeural::DenseT<float, HiddenSize, 1>>;         // Output layer
};

template <int HiddenSize>
struct DiodeModelType<4, HiddenSize>
{
    using ModelType = RTNeural::ModelT<float,
                                       2,
                                       1,
                                       RTNeural::DenseT<float, 2, HiddenSize>,          // Input layer
                                       RTNeural::TanhActivationT<float, HiddenSize>,
                                       RTNeural::DenseT<float, HiddenSize, HiddenSize>, // Hidden layer 1
                                       RTNeural::TanhActivationT<float, HiddenSize>,
                                       RTNeural::DenseT<float, HiddenSize, HiddenSize>, // Hidden layer 2
                                       RTNeural::TanhActivationT<float, HiddenSize>,
                                       RTNeural::DenseT<float, HiddenSize, HiddenSize>, // Hidden layer 3
                                       RTNeural::TanhActivationT<float, HiddenSize>,
                                       RTNeural::DenseT<float, HiddenSize, HiddenSize>, // Hidden layer 4
                                       RTNeural::TanhActivationT<float, HiddenSize>,
                                       RTNeural::DenseT<float, HiddenSize, 1>>;         // Output layer
};

template <typename Next, int NLayers, int HiddenSize>
class DiodePairNeuralModel : public wdft::RootWDF
{
public:
    explicit DiodePairNeuralModel (Next& n, const String& modelName) : next (n)
    {
        DBG ("Initalizing diode model: " + modelName);
#if JUCE_DEBUG
        constexpr bool debug = true;
#else
        constexpr bool debug = false;
#endif

        int modelDataSize;
        auto modelData = ModelData::getNamedResource (modelName.getCharPointer(), modelDataSize);
        jassert (modelData != nullptr);

        MemoryInputStream modelInputStream { modelData, (size_t) modelDataSize, false };
        auto jsonInput = nlohmann::json::parse (modelInputStream.readEntireStreamAsString().toStdString());
        model.parseJson (jsonInput, debug);
    }

    void calcImpedance() override { logR = std::log (next.wdf.R); }

    inline void incident (float x) { wdf.a = x; }

    inline float reflected()
    {
        const float inData alignas(16)[] = { wdf.a, logR };
        wdf.b = -model.template forward (inData);
        return wdf.b;
    }

    wdft::WDFMembers<float> wdf;

private:
    const Next& next;
    float logR = 1.0f;

    typename DiodeModelType<NLayers, HiddenSize>::ModelType model;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (DiodePairNeuralModel)
};
