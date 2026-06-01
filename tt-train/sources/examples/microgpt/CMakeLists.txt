project(microgpt)

add_executable(microgpt main.cpp)
target_link_libraries(microgpt PRIVATE ttml)

target_compile_definitions(
    microgpt
    PRIVATE
        DATA_FOLDER="${CMAKE_SOURCE_DIR}/data"
        CONFIGS_FOLDER="${CMAKE_SOURCE_DIR}/configs"
)

# Download the makemore `names` dataset (pinned commit), if not already present.
set(NAMES_URL "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt")
set(NAMES_FILE "${CMAKE_SOURCE_DIR}/data/names.txt")

if(NOT EXISTS "${NAMES_FILE}")
    message(STATUS "Downloading names dataset to ${NAMES_FILE}")
    file(
        DOWNLOAD
            ${NAMES_URL}
            ${NAMES_FILE}
        SHOW_PROGRESS
    )
else()
    message(STATUS "Names dataset already exists at ${NAMES_FILE}, skipping download.")
endif()
