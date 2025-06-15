# Smoke tests (fast, not necessarily thorough)
# Already present
add_library(unit_tests_stl_smoke OBJECT)
add_library(TT::Metalium::Test::STL::Smoke ALIAS unit_tests_stl_smoke)

target_sources(
    unit_tests_stl_smoke
    PRIVATE
        test_any_range.cpp
        test_indestructible.cpp
        test_slotmap.cpp
        test_span.cpp
        test_strong_type.cpp
)

target_include_directories(
    unit_tests_stl_smoke
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${boost_SOURCE_DIR}/libs/core/include
)

target_link_libraries(
    unit_tests_stl_smoke
    PRIVATE
        gmock
        gtest
        gtest_main
        metal_common_libs
)

add_executable(unit_tests_stl)
target_link_libraries(unit_tests_stl PRIVATE TT::Metalium::Test::STL::Smoke)
