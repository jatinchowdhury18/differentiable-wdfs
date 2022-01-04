# juce_set_vst2_sdk_path(C:/SDKs/VST_SDK/VST2_SDK)

# set default plugin formats to build
set(JUCE_FORMATS AU VST3 Standalone)

# Build VST2 is SDK set
if(TARGET juce_vst2_sdk)
    message(STATUS "Building VST2 plugin format")
    list(APPEND JUCE_FORMATS VST)
endif()

juce_add_plugin(DifferentiableWDFs
    COMPANY_NAME "${COMPANY_NAME}"
    PLUGIN_MANUFACTURER_CODE Chow
    PLUGIN_CODE Dwdf
    FORMATS ${JUCE_FORMATS}
    PRODUCT_NAME "${PRODUCT_NAME}"

    MICROPHONE_PERMISSION_ENABLED TRUE
)

juce_generate_juce_header(DifferentiableWDFs)
add_subdirectory(src)
include_directories(src)
add_subdirectory(res)

target_compile_definitions(DifferentiableWDFs PUBLIC
    JUCE_VST3_CAN_REPLACE_VST2=0
)

target_link_libraries(DifferentiableWDFs PRIVATE juce_plugin_modules)
