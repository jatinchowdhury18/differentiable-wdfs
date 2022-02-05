add_subdirectory(JUCE)
include(cmake/SubprojectVersion.cmake)
subproject_version(JUCE juce_version)
message(STATUS "VERSION for JUCE: ${juce_version}")

# add custom JUCE modules
add_subdirectory(chowdsp_utils)

# add RTNeural
set(RTNEURAL_XSIMD ON CACHE BOOL "Use RTNeural with XSIMD backend" FORCE)
add_subdirectory(RTNeural)

add_subdirectory(toms917)

include(cmake/WarningFlags.cmake)
add_library(juce_plugin_modules STATIC)
target_link_libraries(juce_plugin_modules
    PRIVATE
        juce::juce_audio_utils
        juce::juce_audio_plugin_client
        juce::juce_dsp
        chowdsp_dsp
        chowdsp_gui
        chowdsp_plugin_utils
        RTNeural
        toms917
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        warning_flags
)

target_compile_definitions(juce_plugin_modules
    PUBLIC
        JUCE_DISPLAY_SPLASH_SCREEN=0
        JUCE_REPORT_APP_USAGE=0
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_JACK=1
        JUCE_ALSA=1
        CHOWDSP_USE_FOLEYS_CLASSES=0
        JucePlugin_Manufacturer="${COMPANY_NAME}"
        JucePlugin_VersionString="${CMAKE_PROJECT_VERSION}"
        JucePlugin_Name="${CMAKE_PROJECT_NAME}"
        JUCE_MODAL_LOOPS_PERMITTED=1
    INTERFACE
        $<TARGET_PROPERTY:juce_plugin_modules,COMPILE_DEFINITIONS>
)

target_include_directories(juce_plugin_modules
    INTERFACE
        $<TARGET_PROPERTY:juce_plugin_modules,INCLUDE_DIRECTORIES>
    PUBLIC
        ${CMAKE_SOURCE_DIR}/modules/RTNeural
        ${CMAKE_SOURCE_DIR}/modules/RTNeural/modules/Eigen
        ${CMAKE_SOURCE_DIR}/modules/RTNeural/modules/xsimd/include
        ${CMAKE_SOURCE_DIR}/modules/toms917
)

set_target_properties(juce_plugin_modules PROPERTIES
    POSITION_INDEPENDENT_CODE TRUE
    VISIBILITY_INLINES_HIDDEN TRUE
    C_VISBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
)

if(ASIOSDK_DIR)
    message(STATUS "Using ASIO SDK from ${ASIOSDK_DIR}")
    target_include_directories(juce_plugin_modules PUBLIC ${ASIOSDK_DIR}/common)
    target_compile_definitions(juce_plugin_modules PUBLIC JUCE_ASIO=1)
endif()
