file( GLOB APP_SOURCES "*.cpp")
foreach( sourcefile ${APP_SOURCES} )
    # executable name cut off .cpp
    string( REGEX REPLACE ".+/" "" executablename1 ${sourcefile} )
    string( REPLACE ".cpp" "" executablename ${executablename1} )

    message ("source files: " ${sourcefile} " executable: " ${executablename})
    add_executable( ${executablename} ${sourcefile} )
    target_link_libraries( ${executablename} -Wl,--allow-multiple-definition -Wl,--whole-archive tensorflow -Wl,--no-whole-archive )
endforeach( sourcefile ${APP_SOURCES} )