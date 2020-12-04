#include "facegenlib.h"
#include "facegen.h"

IMPORT_EXPORT Graph* _cdecl create_graph() {
    return new GraphLib;
}