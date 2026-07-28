#include "lammpsplugin.h"

#include "version.h"

#include "pair_lj_cut_rvv.h"

#include <cstring>

using namespace LAMMPS_NS;

static Pair *ljcutrvvcreator(LAMMPS *lmp) { return new PairLJCutRVV(lmp); }

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
  lammpsplugin_t plugin;
  lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc) regfunc;

  plugin.version = LAMMPS_VERSION;
  plugin.style = "pair";
  plugin.name = "lj/cut/rvv";
  plugin.info = "LJ/cut with hand RVV SoA tiles (X60) v1.0";
  plugin.author = "OpenSolvers RVV path";
  plugin.creator.v1 = (lammpsplugin_factory1 *) &ljcutrvvcreator;
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);
}
