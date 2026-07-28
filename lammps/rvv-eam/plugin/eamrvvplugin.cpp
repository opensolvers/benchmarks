#include "lammpsplugin.h"

#include "version.h"

#include "pair_eam_rvv.h"

using namespace LAMMPS_NS;

static Pair *eamrvvcreator(LAMMPS *lmp) { return new PairEAMRVV(lmp); }

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
  lammpsplugin_t plugin;
  lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc) regfunc;

  plugin.version = LAMMPS_VERSION;
  plugin.style = "pair";
  plugin.name = "eam/rvv";
  plugin.info = "EAM with hand RVV spline tiles (X60) v1.0";
  plugin.author = "OpenSolvers RVV path";
  plugin.creator.v1 = (lammpsplugin_factory1 *) &eamrvvcreator;
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);
}
