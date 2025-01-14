/* Portions copyright (c) 2006-2016 Stanford University and Simbios.
 * Contributors: Pande Group
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "ReferenceMorseBondIxn.h"
#include "ReferenceForce.h"
#include <cmath>

using std::vector;
using namespace OpenMM;

/**---------------------------------------------------------------------------------------

   ReferenceMorseBondIxn constructor

   --------------------------------------------------------------------------------------- */

ReferenceMorseBondIxn::ReferenceMorseBondIxn() : usePeriodic(false) {
}

/**---------------------------------------------------------------------------------------

   ReferenceMorseBondIxn destructor

   --------------------------------------------------------------------------------------- */

ReferenceMorseBondIxn::~ReferenceMorseBondIxn() {
}

void ReferenceMorseBondIxn::setPeriodic(OpenMM::Vec3* vectors) {
    usePeriodic = true;
    boxVectors[0] = vectors[0];
    boxVectors[1] = vectors[1];
    boxVectors[2] = vectors[2];
}

/**---------------------------------------------------------------------------------------

   Calculate Morse Bond Ixn

   @param atomIndices      atom indices of atom participating in bond
   @param atomCoordinates  atom coordinates
   @param parameters       parameters: parameters[0] = ideal bond length
                                       parameters[1] = bond k
                                       parameters[2] = bond d
   @param forces           force array (forces added to input values)
   @param totalEnergy      if not null, the energy will be added to this

   --------------------------------------------------------------------------------------- */

void ReferenceMorseBondIxn::calculateBondIxn(vector<int>& atomIndices,
                                             vector<Vec3>& atomCoordinates,
                                             vector<double>& parameters,
                                             vector<Vec3>& forces,
                                             double* totalEnergy, double* energyParamDerivs) {
    double deltaR[ReferenceForce::LastDeltaRIndex];
    
    int atomAIndex = atomIndices[0];
    int atomBIndex = atomIndices[1];
    if (usePeriodic)
       ReferenceForce::getDeltaRPeriodic(atomCoordinates[atomAIndex], atomCoordinates[atomBIndex], boxVectors, deltaR);  
    else
       ReferenceForce::getDeltaR(atomCoordinates[atomAIndex], atomCoordinates[atomBIndex], deltaR);  

    // deltaIdeal r - r0
    double deltaIdeal = deltaR[ReferenceForce::RIndex] - parameters[0];

    // beta
    double beta = sqrt(parameters[1]/(2.0*parameters[2]));

    // Exponential term
    double expTerm = exp(-beta*deltaIdeal);

    // Chain rule
    double dEdR = 2.0*parameters[2]*beta*(1-expTerm)*expTerm;
    dEdR = deltaR[ReferenceForce::RIndex] > 0.0 ? (dEdR/deltaR[ReferenceForce::RIndex]) : 0.0;

    // atomAIndex
    forces[atomAIndex][0]     += dEdR*deltaR[ReferenceForce::XIndex];
    forces[atomAIndex][1]     += dEdR*deltaR[ReferenceForce::YIndex];
    forces[atomAIndex][2]     += dEdR*deltaR[ReferenceForce::ZIndex];

    // atomBIndex
    forces[atomBIndex][0]     -= dEdR*deltaR[ReferenceForce::XIndex];
    forces[atomBIndex][1]     -= dEdR*deltaR[ReferenceForce::YIndex];
    forces[atomBIndex][2]     -= dEdR*deltaR[ReferenceForce::ZIndex];

    // Total energy
    if (totalEnergy != NULL)
        *totalEnergy += parameters[2]*(1-expTerm)*(1-expTerm);

}