file(GLOB ANALYTICAL_SOURCES CONFIGURE_DEPENDS "*.cc")
file(GLOB ANALYTICAL_HEADERS CONFIGURE_DEPENDS "*.hh" "*.h")

add_executable(SimAI_analytical ${ANALYTICAL_SOURCES} ${ANALYTICAL_HEADERS})
target_include_directories(SimAI_analytical PRIVATE "${ASTRA_SIM_DIR}")
target_link_libraries(SimAI_analytical PRIVATE AstraSim)
