/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2021 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/MorseBondForceImpl.h"
#include "kernels.h"
#include <sstream>

using namespace OpenMM;

MorseBondForceImpl::MorseBondForceImpl(const MorseBondForce& owner) : owner(owner) {
}

MorseBondForceImpl::~MorseBondForceImpl() {
}

void MorseBondForceImpl::initialize(ContextImpl& context) {
    const System& system = context.getSystem();
    for (int i = 0; i < owner.getNumBonds(); i++) {
        int particle[2];
        double length, k, d;
        owner.getBondParameters(i, particle[0], particle[1], length, k, d);
        for (int j = 0; j < 2; j++) {
            if (particle[j] < 0 || particle[j] >= system.getNumParticles()) {
                std::stringstream msg;
                msg << "MorseBondForce: Illegal particle index for a bond: ";
                msg << particle[j];
                throw OpenMMException(msg.str());
            }
        }
        if (length < 0) {
            throw OpenMMException("MorseBondForce: bond length cannot be negative");
        }
        // FIXME: do we check well-depth here too?
    }
    kernel = context.getPlatform().createKernel(CalcMorseBondForceKernel::Name(), context);
    kernel.getAs<CalcMorseBondForceKernel>().initialize(context.getSystem(), owner);
}

double MorseBondForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces,
                                               bool includeEnergy, int groups) {
    if ((groups&(1<<owner.getForceGroup())) != 0) {
        return kernel.getAs<CalcMorseBondForceKernel>().execute(context, includeForces, includeEnergy);
    }
    return 0.0;
}

std::vector<std::string> MorseBondForceImpl::getKernelNames() {
    std::vector<std::string> names;
    names.push_back(CalcMorseBondForceKernel::Name());
    return names;
}

std::vector<std::pair<int, int>> MorseBondForceImpl::getBondedParticles() const {
    int numBonds = owner.getNumBonds();
    std::vector<std::pair<int, int>> bonds(numBonds);
    for (int i = 0; i < numBonds; i++) {
        double length, k, d;
        owner.getBondParameters(i, bonds[i].first, bonds[i].second, length, k, d);
    }
    return bonds;
}

void MorseBondForceImpl::updateParametersInContext(ContextImpl& context) {
    kernel.getAs<CalcMorseBondForceKernel>().copyParametersToContext(context, owner);
    context.systemChanged();
}