//======================================================================================================================
//! \file 01_BasicLBM_CollideStreamSplit.cpp
//! \brief Fork of tutorials/lbm/01_BasicLBM with CLI/prm switch:
//!   sweepMode = stock  → fused lbm::makeCellwiseSweep (stream-pull + collide)
//!   sweepMode = split  → StreamPull then contiguous SoA collide (auto-vec friendly)
//!
//! Same DomainSetup / Boundaries / VTK as the tutorial. Prints WALL, MLUPS, density checksum.
//======================================================================================================================

#include "blockforest/all.h"
#include "core/all.h"
#include "domain_decomposition/all.h"
#include "field/all.h"
#include "geometry/all.h"
#include "lbm/all.h"
#include "timeloop/all.h"

#include "SoaCollideKernels.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

namespace walberla {

using LatticeModel_T = lbm::D2Q9< lbm::collision_model::SRT >;
using Stencil_T = LatticeModel_T::Stencil;
using CommunicationStencil_T = LatticeModel_T::CommunicationStencil;
using PdfField_T = lbm::PdfField< LatticeModel_T >;
using flag_t = walberla::uint8_t;
using FlagField_T = FlagField< flag_t >;

namespace {

std::string normalizeMode(std::string m)
{
   for (char& c : m) {
      if (c >= 'A' && c <= 'Z')
         c = static_cast<char>(c - 'A' + 'a');
   }
   if (m == "cellwise" || m == "fused" || m == "stock")
      return "stock";
   if (m == "split" || m == "soa" || m == "collide-stream" || m == "collide_stream")
      return "split";
   return m;
}

/// Contiguous-fzyx SoA collide sweep (in-place), fluid cells only via flag blend in kernel.
class SoaCollideSweep
{
 public:
   SoaCollideSweep(const BlockDataID& pdfFieldId, const ConstBlockDataID& flagFieldId, const FlagUID& fluidFlag)
      : pdfFieldId_(pdfFieldId), flagFieldId_(flagFieldId), fluidFlag_(fluidFlag)
   {}

   void operator()(IBlock* const block)
   {
      auto* pdf = block->getData< PdfField_T >(pdfFieldId_);
      auto* flags = block->getData< FlagField_T >(flagFieldId_);
      WALBERLA_ASSERT_NOT_NULLPTR(pdf);
      WALBERLA_ASSERT_NOT_NULLPTR(flags);
      WALBERLA_ASSERT_EQUAL(pdf->layout(), field::fzyx);
      WALBERLA_ASSERT_EQUAL(pdf->xStride(), 1);

      const flag_t fluid = flags->getFlag(fluidFlag_);
      const real_t omega = pdf->latticeModel().collisionModel().omega();
      const cell_idx_t xSize = cell_idx_c(pdf->xSize());
      const cell_idx_t ySize = cell_idx_c(pdf->ySize());
      const cell_idx_t zSize = cell_idx_c(pdf->zSize());

      using namespace stencil;

      for (cell_idx_t z = 0; z < zSize; ++z) {
         for (cell_idx_t y = 0; y < ySize; ++y) {
            // fzyx: each PDF component is contiguous along x for fixed (y,z)
            double* fC  = &pdf->get(0, y, z, Stencil_T::idx[C]);
            double* fN  = &pdf->get(0, y, z, Stencil_T::idx[N]);
            double* fS  = &pdf->get(0, y, z, Stencil_T::idx[S]);
            double* fW  = &pdf->get(0, y, z, Stencil_T::idx[W]);
            double* fE  = &pdf->get(0, y, z, Stencil_T::idx[E]);
            double* fNW = &pdf->get(0, y, z, Stencil_T::idx[NW]);
            double* fNE = &pdf->get(0, y, z, Stencil_T::idx[NE]);
            double* fSW = &pdf->get(0, y, z, Stencil_T::idx[SW]);
            double* fSE = &pdf->get(0, y, z, Stencil_T::idx[SE]);
            const flag_t* rowFlags = &flags->get(0, y, z);

            cssplit::collideSoARowD2Q9IncompSRT(
               fC, fN, fS, fW, fE, fNW, fNE, fSW, fSE,
               reinterpret_cast< const std::uint8_t* >(rowFlags),
               static_cast< std::uint8_t >(fluid),
               static_cast< std::size_t >(xSize),
               double(omega));
         }
      }
   }

 private:
   BlockDataID pdfFieldId_;
   ConstBlockDataID flagFieldId_;
   FlagUID fluidFlag_;
};

double densityChecksum(const shared_ptr< StructuredBlockForest >& blocks,
                       BlockDataID pdfFieldId, BlockDataID flagFieldId, const FlagUID& fluidFlag)
{
   double localSum = 0.0;
   uint_t localCells = 0;
   for (auto block = blocks->begin(); block != blocks->end(); ++block) {
      auto* pdf = block->getData< PdfField_T >(pdfFieldId);
      auto* flags = block->getData< FlagField_T >(flagFieldId);
      const flag_t fluid = flags->getFlag(fluidFlag);
      const cell_idx_t xSize = cell_idx_c(pdf->xSize());
      const cell_idx_t ySize = cell_idx_c(pdf->ySize());
      const cell_idx_t zSize = cell_idx_c(pdf->zSize());
      for (cell_idx_t z = 0; z < zSize; ++z)
         for (cell_idx_t y = 0; y < ySize; ++y)
            for (cell_idx_t x = 0; x < xSize; ++x) {
               if (!flags->isFlagSet(x, y, z, fluid))
                  continue;
               // Incompressible: getDensity returns ρ_phys = ρ_dev + 1
               localSum += double(pdf->getDensity(x, y, z));
               ++localCells;
            }
   }
   double globalSum = localSum;
   uint_t globalCells = localCells;
   mpi::allReduceInplace(globalSum, mpi::SUM);
   mpi::allReduceInplace(globalCells, mpi::SUM);
   WALBERLA_LOG_RESULT_ON_ROOT("checksum_density_sum=" << globalSum
                                                       << " fluid_cells=" << globalCells
                                                       << " mean_density=" << (globalCells ? globalSum / double(globalCells) : 0.0));
   return globalSum;
}

} // namespace

int main(int argc, char** argv)
{
   walberla::Environment walberlaEnv(argc, argv);

   auto blocks = blockforest::createUniformBlockGridFromConfig(walberlaEnv.config());

   auto parameters = walberlaEnv.config()->getOneBlock("Parameters");

   const real_t omega = parameters.getParameter< real_t >("omega", real_c(1.4));
   const Vector3< real_t > initialVelocity =
      parameters.getParameter< Vector3< real_t > >("initialVelocity", Vector3< real_t >());
   const uint_t timesteps = parameters.getParameter< uint_t >("timesteps", uint_c(10));
   const real_t remainingTimeLoggerFrequency =
      parameters.getParameter< real_t >("remainingTimeLoggerFrequency", real_c(3.0));

   std::string sweepMode = normalizeMode(parameters.getParameter< std::string >("sweepMode", "stock"));
   // Optional CLI override: --sweepMode=split|stock
   for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i] ? argv[i] : "";
      const std::string key = "--sweepMode=";
      if (a.rfind(key, 0) == 0)
         sweepMode = normalizeMode(a.substr(key.size()));
      else if (a == "--sweepMode" && i + 1 < argc)
         sweepMode = normalizeMode(argv[++i]);
   }
   if (sweepMode != "stock" && sweepMode != "split") {
      WALBERLA_ABORT("Unknown sweepMode '" << sweepMode << "' (expected stock|split)");
   }
   WALBERLA_LOG_RESULT_ON_ROOT("sweepMode=" << sweepMode);

   LatticeModel_T latticeModel = LatticeModel_T(lbm::collision_model::SRT(omega));
   BlockDataID pdfFieldId = lbm::addPdfFieldToStorage(blocks, "pdf field", latticeModel, initialVelocity, real_t(1));
   BlockDataID flagFieldId = field::addFlagFieldToStorage< FlagField_T >(blocks, "flag field");

   const FlagUID fluidFlagUID("Fluid");

   auto boundariesConfig = walberlaEnv.config()->getOneBlock("Boundaries");

   using BHFactory = lbm::DefaultBoundaryHandlingFactory< LatticeModel_T, FlagField_T >;

   BlockDataID boundaryHandlingId = BHFactory::addBoundaryHandlingToStorage(
      blocks, "boundary handling", flagFieldId, pdfFieldId, fluidFlagUID,
      boundariesConfig.getParameter< Vector3< real_t > >("velocity0", Vector3< real_t >()),
      boundariesConfig.getParameter< Vector3< real_t > >("velocity1", Vector3< real_t >()),
      boundariesConfig.getParameter< real_t >("pressure0", real_c(1.0)),
      boundariesConfig.getParameter< real_t >("pressure1", real_c(1.0)));

   geometry::initBoundaryHandling< BHFactory::BoundaryHandling >(*blocks, boundaryHandlingId, boundariesConfig);
   geometry::setNonBoundaryCellsToDomain< BHFactory::BoundaryHandling >(*blocks, boundaryHandlingId);

   SweepTimeloop timeloop(blocks->getBlockStorage(), timesteps);

   blockforest::communication::UniformBufferedScheme< CommunicationStencil_T > communication(blocks);
   communication.addPackInfo(make_shared< lbm::PdfFieldPackInfo< LatticeModel_T > >(pdfFieldId));

   timeloop.add() << BeforeFunction(communication, "communication")
                  << Sweep(BHFactory::BoundaryHandling::getBlockSweep(boundaryHandlingId), "boundary handling");

   auto cellwise =
      lbm::makeCellwiseSweep< LatticeModel_T, FlagField_T >(pdfFieldId, flagFieldId, fluidFlagUID);

   if (sweepMode == "stock") {
      timeloop.add() << Sweep(makeSharedSweep(cellwise), "LB stream & collide");
   } else {
      // Order matches CellwiseSweep: stream-pull first, then collide in-place.
      timeloop.add() << Sweep([cellwise](IBlock* const b) { cellwise->stream(b); }, "LB stream");
      timeloop.add() << Sweep(SoaCollideSweep(pdfFieldId, flagFieldId, fluidFlagUID), "LB SoA collide");
   }

   auto checkFunction = [](PdfField_T::value_type value) { return math::finite(value); };
   timeloop.addFuncAfterTimeStep(
      makeSharedFunctor(field::makeStabilityChecker< PdfField_T, FlagField_T >(
         walberlaEnv.config(), blocks, pdfFieldId, flagFieldId, fluidFlagUID, checkFunction)),
      "LBM stability check");

   timeloop.addFuncAfterTimeStep(
      timing::RemainingTimeLogger(timeloop.getNrOfTimeSteps(), remainingTimeLoggerFrequency),
      "remaining time logger");

   lbm::VTKOutput< LatticeModel_T, FlagField_T >::addToTimeloop(
      timeloop, blocks, walberlaEnv.config(), pdfFieldId, flagFieldId, fluidFlagUID);

   const auto t0 = std::chrono::steady_clock::now();
   timeloop.run();
   const auto t1 = std::chrono::steady_clock::now();
   const double wall =
      std::chrono::duration_cast< std::chrono::duration< double > >(t1 - t0).count();

   const uint_t cellsPerBlock =
      blocks->getNumberOfXCellsPerBlock() * blocks->getNumberOfYCellsPerBlock() * blocks->getNumberOfZCellsPerBlock();
   const uint_t nBlocks = blocks->getNumberOfBlocks();
   uint_t globalBlocks = nBlocks;
   mpi::allReduceInplace(globalBlocks, mpi::SUM);
   // Approx domain cells (includes boundary cells in blocks — OK for MLUPS report)
   const double totalCells = double(cellsPerBlock) * double(globalBlocks);
   const double mlups = (wall > 0.0) ? (totalCells * double(timesteps) * 1e-6 / wall) : 0.0;

   WALBERLA_LOG_RESULT_ON_ROOT("WALL " << wall);
   WALBERLA_LOG_RESULT_ON_ROOT("MLUPS " << mlups << " (cells≈" << totalCells << " steps=" << timesteps << ")");
   WALBERLA_LOG_RESULT_ON_ROOT("sweepMode=" << sweepMode);

   densityChecksum(blocks, pdfFieldId, flagFieldId, fluidFlagUID);

   return EXIT_SUCCESS;
}

} // namespace walberla

int main(int argc, char** argv)
{
   return walberla::main(argc, argv);
}
