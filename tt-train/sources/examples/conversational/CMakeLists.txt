project(conversational)

add_executable(conversational main.cpp)
target_link_libraries(conversational PRIVATE ttml nlohmann_json::nlohmann_json)

target_compile_definitions(
    conversational
    PRIVATE
        DATA_FOLDER="${CMAKE_SOURCE_DIR}/data"
        CONFIGS_FOLDER="${CMAKE_SOURCE_DIR}/configs"
)
