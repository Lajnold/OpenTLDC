#include <Eigen/Core>

#include "structs.h"

void tldReinitBB(TldStruct& tld, Eigen::Vector4d& bb) {
    tld.reinitBB = bb;
    tld.isReinitBB = true;
}
