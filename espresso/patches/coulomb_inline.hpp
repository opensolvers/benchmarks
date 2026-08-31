/*
 * Copyright (C) 2010-2022 The ESPResSo project
 *
 * This file is part of ESPResSo.
 *
 * Opensolvers patch: replace std::function short-range Coulomb force kernels
 * with a trivial function+context ref to cut type-erasure overhead in
 * add_non_bonded_pair_force (hot on P3M workloads).
 */

#ifndef ESPRESSO_SRC_CORE_ELECTROSTATICS_COULOMB_INLINE_HPP
#define ESPRESSO_SRC_CORE_ELECTROSTATICS_COULOMB_INLINE_HPP

#include "config.hpp"

#include "electrostatics/coulomb.hpp"

#include "Particle.hpp"

#include <utils/Vector.hpp>
#include <utils/demangle.hpp>
#include <utils/math/tensor_product.hpp>
#include <utils/matrix.hpp>

#include <functional>
#include <memory>
#include <tuple>
#include <type_traits>

namespace Coulomb {

/** Trivial callable: direct call, no std::function type erasure. */
struct ForceKernelRef {
  using Fn = Utils::Vector3d (*)(void const *, double, Utils::Vector3d const &,
                                 double);
  Fn fn = nullptr;
  void const *ctx = nullptr;

  Utils::Vector3d operator()(double q1q2, Utils::Vector3d const &d,
                             double dist) const {
    return fn(ctx, q1q2, d, dist);
  }

  explicit operator bool() const { return fn != nullptr; }
};

struct CorrectionKernelRef {
  using Fn = void (*)(void const *, Particle &, Particle &, double);
  Fn fn = nullptr;
  void const *ctx = nullptr;

  void operator()(Particle &p1, Particle &p2, double q1q2) const {
    fn(ctx, p1, p2, q1q2);
  }

  explicit operator bool() const { return fn != nullptr; }
};

struct ShortRangeForceKernel
    : public boost::static_visitor<boost::optional<ForceKernelRef>> {

  using kernel_type = result_type::value_type;

#ifdef ELECTROSTATICS
  template <typename T>
  result_type operator()(std::shared_ptr<T> const &ptr) const {
    return kernel_type{
        [](void const *ctx, double q1q2, Utils::Vector3d const &d,
           double dist) {
          return static_cast<T const *>(ctx)->pair_force(q1q2, d, dist);
        },
        static_cast<void const *>(ptr.get())};
  }

#ifdef P3M
  auto
  operator()(std::shared_ptr<ElectrostaticLayerCorrection> const &ptr) const {
    return boost::apply_visitor(*this, ptr->base_solver);
  }
#endif // P3M

#ifdef MMM1D_GPU
  result_type operator()(std::shared_ptr<CoulombMMM1DGpu> const &) const {
    return {};
  }
#endif // MMM1D_GPU
#endif // ELECTROSTATICS
};

struct ShortRangeForceCorrectionsKernel
    : public boost::static_visitor<boost::optional<CorrectionKernelRef>> {

  using kernel_type = result_type::value_type;

  template <typename T>
  result_type operator()(std::shared_ptr<T> const &) const {
    return {};
  }

#ifdef P3M
  result_type
  operator()(std::shared_ptr<ElectrostaticLayerCorrection> const &ptr) const {
    return kernel_type{
        [](void const *ctx, Particle &p1, Particle &p2, double q1q2) {
          static_cast<ElectrostaticLayerCorrection const *>(ctx)
              ->add_pair_force_corrections(p1, p2, q1q2);
        },
        static_cast<void const *>(ptr.get())};
  }
#endif // P3M
};

inline ShortRangeForceKernel::result_type pair_force_kernel() {
#ifdef ELECTROSTATICS
  if (electrostatics_actor) {
    auto const visitor = ShortRangeForceKernel();
    return boost::apply_visitor(visitor, *electrostatics_actor);
  }
#endif // ELECTROSTATICS
  return {};
}

struct ShortRangePressureKernel
    : public boost::static_visitor<boost::optional<std::function<Utils::Matrix<
          double, 3, 3>(double, Utils::Vector3d const &, double)>>> {

  using kernel_type = result_type::value_type;

#ifdef ELECTROSTATICS
  template <typename T,
            std::enable_if_t<traits::has_pressure<T>::value> * = nullptr>
  result_type operator()(std::shared_ptr<T> const &ptr) const {
    result_type pressure_kernel = {};
    if (auto const force_kernel_opt = pair_force_kernel()) {
      pressure_kernel =
          kernel_type{[force_kernel = *force_kernel_opt](
                          double q1q2, Utils::Vector3d const &d, double dist) {
            auto const force = force_kernel(q1q2, d, dist);
            return Utils::tensor_product(force, d);
          }};
    }
    return pressure_kernel;
  }

  template <typename T,
            std::enable_if_t<!traits::has_pressure<T>::value> * = nullptr>
  result_type operator()(std::shared_ptr<T> const &) const {
    return {};
  }
#endif // ELECTROSTATICS
};

struct ShortRangeEnergyKernel
    : public boost::static_visitor<boost::optional<
          std::function<double(Particle const &, Particle const &, double,
                               Utils::Vector3d const &, double)>>> {

  using kernel_type = result_type::value_type;

#ifdef ELECTROSTATICS
  template <typename T>
  result_type operator()(std::shared_ptr<T> const &ptr) const {
    auto const &actor = *ptr;
    return kernel_type{[&actor](Particle const &, Particle const &, double q1q2,
                                Utils::Vector3d const &, double dist) {
      return actor.pair_energy(q1q2, dist);
    }};
  }
#ifdef P3M
  result_type
  operator()(std::shared_ptr<ElectrostaticLayerCorrection> const &ptr) const {
    auto const &actor = *ptr;
    auto const energy_kernel = boost::apply_visitor(*this, actor.base_solver);
    return kernel_type{[&actor, energy_kernel](
                           Particle const &p1, Particle const &p2, double q1q2,
                           Utils::Vector3d const &d, double dist) {
      auto energy = 0.;
      if (energy_kernel) {
        energy = (*energy_kernel)(p1, p2, q1q2, d, dist);
      }
      return energy + actor.pair_energy_correction(q1q2, p1, p2);
    }};
  }
#endif // P3M
#ifdef MMM1D_GPU
  result_type operator()(std::shared_ptr<CoulombMMM1DGpu> const &) const {
    return {};
  }
#endif // MMM1D_GPU
  result_type operator()(std::shared_ptr<CoulombMMM1D> const &actor) const {
    return kernel_type{[&actor](Particle const &, Particle const &, double q1q2,
                                Utils::Vector3d const &d, double dist) {
      return actor->pair_energy(q1q2, d, dist);
    }};
  }
#endif // ELECTROSTATICS
};

inline ShortRangeForceCorrectionsKernel::result_type pair_force_elc_kernel() {
#ifdef ELECTROSTATICS
  if (electrostatics_actor) {
    auto const visitor = ShortRangeForceCorrectionsKernel();
    return boost::apply_visitor(visitor, *electrostatics_actor);
  }
#endif // ELECTROSTATICS
  return {};
}

inline ShortRangePressureKernel::result_type pair_pressure_kernel() {
#ifdef ELECTROSTATICS
  if (electrostatics_actor) {
    auto const visitor = ShortRangePressureKernel();
    return boost::apply_visitor(visitor, *electrostatics_actor);
  }
#endif // ELECTROSTATICS
  return {};
}

inline ShortRangeEnergyKernel::result_type pair_energy_kernel() {
#ifdef ELECTROSTATICS
  if (electrostatics_actor) {
    auto const visitor = ShortRangeEnergyKernel();
    return boost::apply_visitor(visitor, *electrostatics_actor);
  }
#endif // ELECTROSTATICS
  return {};
}

} // namespace Coulomb

#endif
