//======================================================================================================================
//
//! \file 02_HeatEquation.cpp
//! Bench-oriented fork of waLBerla tutorial PDE 02 (stock Jacobi, no VTK).
//! Adds Config/prm sizing, MPI block layout, and a final checksum for A/B runs.
//
//======================================================================================================================

#include "blockforest/Initialization.h"
#include "blockforest/communication/UniformBufferedScheme.h"

#include "core/Environment.h"
#include "core/math/Constants.h"
#include "core/mpi/Reduce.h"

#include "field/AddToStorage.h"
#include "field/Field.h"
#include "field/communication/PackInfo.h"

#include "stencil/D2Q5.h"

#include "timeloop/SweepTimeloop.h"

#include <cmath>
#include <iomanip>
#include <iostream>


namespace walberla {


using ScalarField = GhostLayerField< real_t, 1 >;
using Stencil_T  = stencil::D2Q5;


void initU( const shared_ptr< StructuredBlockStorage > & blocks, const BlockDataID & uID )
{
   for( auto block = blocks->begin(); block != blocks->end(); ++block )
   {
      auto u = block->getData< ScalarField >( uID );

      CellInterval xyz = u->xyzSize();

      for( auto cell : xyz )
      {
         const Vector3< real_t > p = blocks->getBlockLocalCellCenter( *block, cell );
         u->get( cell ) = std::sin( math::pi * p[0] ) * std::sin( math::pi * p[1] );
      }
   }
}


class JacobiIteration
{
public:
   JacobiIteration( const BlockDataID & srcID, const BlockDataID & dstID, const BlockDataID & rhsID,
                    const std::vector< real_t > & weights, const shared_ptr< StructuredBlockStorage > & blocks,
                    blockforest::communication::UniformBufferedScheme< Stencil_T > & myCommScheme,
                    const uint_t & maxIterations )
      : srcID_( srcID ), dstID_( dstID ), rhsID_( rhsID ), weights_( weights ), blocks_( blocks ),
        myCommScheme_( myCommScheme ), maxIterations_( maxIterations )
   {}

   void operator()();

private:
   const BlockDataID srcID_;
   const BlockDataID dstID_;
   const BlockDataID rhsID_;

   std::vector< real_t > weights_;
   const shared_ptr< StructuredBlockStorage > blocks_;
   blockforest::communication::UniformBufferedScheme< Stencil_T > myCommScheme_;

   const uint_t maxIterations_;
};

void JacobiIteration::operator()()
{
   for( uint_t i = 0; i < maxIterations_; ++i )
   {
      myCommScheme_();

      for( auto block = blocks_->begin(); block != blocks_->end(); ++block )
      {
         auto src = block->getData< ScalarField >( srcID_ );
         auto dst = block->getData< ScalarField >( dstID_ );
         auto rhs = block->getData< ScalarField >( rhsID_ );

         WALBERLA_FOR_ALL_CELLS_XYZ( src,

            dst->get( x, y, z ) = rhs->get( x, y, z );

            for( auto dir = Stencil_T::beginNoCenter(); dir != Stencil_T::end(); ++dir )
               dst->get( x, y, z ) -= weights_[ dir.toIdx() ] * src->getNeighbor( x, y, z, *dir );

            dst->get( x, y, z ) /= weights_[ Stencil_T::idx[ stencil::C ] ];
         )

         src->swapDataPointers( dst );
      }
   }
}


class Reinitialize
{
public:
   Reinitialize( const BlockDataID & srcID, const BlockDataID & rhsID )
      : srcID_( srcID ), rhsID_( rhsID )
   {}

   void operator()( IBlock * block );

private:
   const BlockDataID srcID_;
   const BlockDataID rhsID_;
};

void Reinitialize::operator()( IBlock * block )
{
   auto src = block->getData< ScalarField >( srcID_ );
   auto rhs = block->getData< ScalarField >( rhsID_ );
   src->swapDataPointers( rhs );
}


real_t fieldChecksum( const shared_ptr< StructuredBlockStorage > & blocks, const BlockDataID & uID )
{
   real_t sum( real_c( 0 ) );
   for( auto block = blocks->begin(); block != blocks->end(); ++block )
   {
      auto u = block->getData< ScalarField >( uID );
      for( auto cell : u->xyzSize() )
         sum += u->get( cell );
   }
   mpi::allReduceInplace( sum, mpi::SUM );
   return sum;
}


int main( int argc, char ** argv )
{
   walberla::Environment env( argc, argv );

   // Defaults match stock tutorial; prm overrides for stable A/B WALL.
   uint_t xBlocks = uint_c( 1 );
   uint_t yBlocks = uint_c( 1 );
   uint_t zBlocks = uint_c( 1 );
   uint_t xCells  = uint_c( 25 );
   uint_t yCells  = uint_c( 25 );
   uint_t zCells  = uint_c( 1 );
   uint_t timesteps         = uint_c( 20 );
   uint_t jacobiIterations  = uint_c( 10000 );
   real_t dt    = real_c( 0.01 );
   real_t kappa = real_c( 1 );

   if( env.config() )
   {
      auto parameters = env.config()->getOneBlock( "Parameters" );
      dt               = parameters.getParameter< real_t >( "dt", dt );
      kappa            = parameters.getParameter< real_t >( "kappa", kappa );
      timesteps        = parameters.getParameter< uint_t >( "timesteps", timesteps );
      jacobiIterations = parameters.getParameter< uint_t >( "jacobiIterations", jacobiIterations );

      if( env.config()->getNumBlocks( "DomainSetup" ) >= 1 )
      {
         auto domain = env.config()->getOneBlock( "DomainSetup" );
         const Vector3< uint_t > blocksVec =
            domain.getParameter< Vector3< uint_t > >( "blocks", Vector3< uint_t >( xBlocks, yBlocks, zBlocks ) );
         const Vector3< uint_t > cellsVec =
            domain.getParameter< Vector3< uint_t > >( "cellsPerBlock", Vector3< uint_t >( xCells, yCells, zCells ) );
         xBlocks = blocksVec[0];
         yBlocks = blocksVec[1];
         zBlocks = blocksVec[2];
         xCells  = cellsVec[0];
         yCells  = cellsVec[1];
         zCells  = cellsVec[2];
      }
   }

   const uint_t processes = uint_c( MPIManager::instance()->numProcesses() );
   const uint_t nBlocks   = xBlocks * yBlocks * zBlocks;
   if( processes != nBlocks )
      WALBERLA_ABORT( "Number of processes (" << processes << ") must equal total blocks (" << nBlocks << ")" );

   const real_t xMin = real_c( 0 );
   const real_t xMax = real_c( 1 );
   const real_t yMin = real_c( 0 );
   const real_t yMax = real_c( 1 );

   const real_t dx = ( xMax - xMin ) / real_c( xBlocks * xCells + uint_c( 1 ) );
   const real_t dy = ( yMax - yMin ) / real_c( yBlocks * yCells + uint_c( 1 ) );

   auto aabb = math::AABB( xMin + real_c( 0.5 ) * dx, yMin + real_c( 0.5 ) * dy, real_c( 0 ),
                           xMax - real_c( 0.5 ) * dx, yMax - real_c( 0.5 ) * dy, dx );

   // one block per process (true) so np matches DomainSetup.blocks
   shared_ptr< StructuredBlockForest > blocks = blockforest::createUniformBlockGrid(
      aabb, xBlocks, yBlocks, zBlocks, xCells, yCells, zCells, true, false, false, false );

   BlockDataID srcID = field::addToStorage< ScalarField >( blocks, "src", real_c( 0 ), field::fzyx, uint_c( 1 ) );
   BlockDataID dstID = field::addToStorage< ScalarField >( blocks, "dst", real_c( 0 ), field::fzyx, uint_c( 1 ) );
   BlockDataID rhsID = field::addToStorage< ScalarField >( blocks, "rhs", real_c( 0 ), field::fzyx, uint_c( 1 ) );

   initU( blocks, srcID );

   blockforest::communication::UniformBufferedScheme< Stencil_T > myCommScheme( blocks );
   myCommScheme.addPackInfo( make_shared< field::communication::PackInfo< ScalarField > >( srcID ) );

   std::vector< real_t > weights( Stencil_T::Size );
   weights[ Stencil_T::idx[ stencil::C ] ] = real_c( 2 ) * dt * kappa / ( dx * dx ) + real_c( 2 ) * dt * kappa / ( dy * dy ) + real_c( 1 );
   weights[ Stencil_T::idx[ stencil::E ] ] = -dt * kappa / ( dx * dx );
   weights[ Stencil_T::idx[ stencil::W ] ] = -dt * kappa / ( dx * dx );
   weights[ Stencil_T::idx[ stencil::N ] ] = -dt * kappa / ( dy * dy );
   weights[ Stencil_T::idx[ stencil::S ] ] = -dt * kappa / ( dy * dy );

   WALBERLA_ROOT_SECTION()
   {
      std::cout << "HeatEquation bench: blocks=<" << xBlocks << "," << yBlocks << "," << zBlocks << "> "
                << "cellsPerBlock=<" << xCells << "," << yCells << "," << zCells << "> "
                << "timesteps=" << timesteps << " jacobiIterations=" << jacobiIterations
                << " dt=" << dt << " kappa=" << kappa << " (VTK off, stock Jacobi)\n";
   }

   SweepTimeloop timeloop( blocks, timesteps );

   timeloop.add() << Sweep( Reinitialize( srcID, rhsID ), "Reinitialize" );
   timeloop.addFuncAfterTimeStep(
      JacobiIteration( srcID, dstID, rhsID, weights, blocks, myCommScheme, jacobiIterations ), "JacobiIteration" );

   timeloop.run();

   const real_t checksum = fieldChecksum( blocks, srcID );
   WALBERLA_ROOT_SECTION()
   {
      std::cout << std::setprecision( 17 ) << "checksum_u_sum=" << checksum << std::endl;
   }

   return 0;
}
} // namespace walberla


int main( int argc, char ** argv )
{
   return walberla::main( argc, argv );
}
