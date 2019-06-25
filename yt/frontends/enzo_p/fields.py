"""
Fields specific to Enzo-P



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
import numpy as np
from yt.fields.field_info_container import \
    FieldInfoContainer
from yt.fields.particle_fields import \
    add_union_field
from yt.frontends.enzo_p.misc import \
    nested_dict_get

rho_units = "code_mass / code_length**3"
vel_units = "code_velocity"
acc_units = "code_velocity / code_time"
energy_units = "code_velocity**2"
b_units = "code_magnetic"

known_species_names = {
}

NODAL_FLAGS = {
    'bfieldi_x': [1, 0, 0],
    'bfieldi_y': [0, 1, 0],
    'bfieldi_z': [0, 0, 1],
}

class EnzoPFieldInfo(FieldInfoContainer):
    known_other_fields = (
        ("velocity_x", (vel_units, ["velocity_x"], None)),
        ("velocity_y", (vel_units, ["velocity_y"], None)),
        ("velocity_z", (vel_units, ["velocity_z"], None)),
        ("acceleration_x", (acc_units, ["acceleration_x"], None)),
        ("acceleration_y", (acc_units, ["acceleration_y"], None)),
        ("acceleration_z", (acc_units, ["acceleration_z"], None)),
        ("density", (rho_units, ["density"], None)),
        ("density_total", (rho_units, ["total_density"], None)),
        ("total_energy", (energy_units, ["total_energy"], None)),
        ("internal_energy", (energy_units, ["internal_energy"], None)),
        ("bfield_x", (b_units, [], None)),
        ("bfield_y", (b_units, [], None)),
        ("bfield_z", (b_units, [], None)),
        ("bfieldi_x", (b_units, [], None)),
        ("bfieldi_y", (b_units, [], None)),
        ("bfieldi_z", (b_units, [], None)),
    )

    known_particle_fields = (
        ("x", ("code_length", ["particle_position_x"], None)),
        ("y", ("code_length", ["particle_position_y"], None)),
        ("z", ("code_length", ["particle_position_z"], None)),
        ("vx", (vel_units, ["particle_velocity_x"], None)),
        ("vy", (vel_units, ["particle_velocity_y"], None)),
        ("vz", (vel_units, ["particle_velocity_z"], None)),
        ("ax", (acc_units, ["particle_acceleration_x"], None)),
        ("ay", (acc_units, ["particle_acceleration_y"], None)),
        ("az", (acc_units, ["particle_acceleration_z"], None)),
        ("mass", ("code_mass", ["particle_mass"], None)),
    )

    def __init__(self, ds, field_list, slice_info = None):
        super(EnzoPFieldInfo, self).__init__(
            ds, field_list, slice_info=slice_info)

        # setup nodal flag information
        for field in NODAL_FLAGS:
            if ('enzop', field) in self:
                finfo = self['enzop', field]
                finfo.nodal_flag = np.array(NODAL_FLAGS[field])

    def setup_fluid_fields(self):
        from yt.fields.magnetic_field import \
            setup_magnetic_field_aliases
        self.setup_energy_field()
        setup_magnetic_field_aliases(self, "enzop",
                                     ["bfield_%s" % ax for ax in "xyz"])

    def setup_energy_field(self):
        unit_system = self.ds.unit_system
        # check if we need to include magnetic energy
        has_magnetic = ('bfield_x' in self.ds.parameters['Field']['list'])

        if not has_magnetic:
            # thermal energy = total energy - kinetic energy
            def _tot_minus_kin(field, data):
                ret = data["total_energy"] - 0.5*data["velocity_x"]**2.0
                if data.ds.dimensionality > 1:
                    ret -= 0.5*data["velocity_y"]**2.0
                if data.ds.dimensionality > 2:
                    ret -= 0.5*data["velocity_z"]**2.0
                return ret
            self.add_field(("gas", "thermal_energy"), sampling_type="cell",
                           function = _tot_minus_kin,
                           units = unit_system["specific_energy"])
        else:
            # thermal energy = total energy - kinetic energy - magnetic energy
            def _sub_b(field, data):
                ret = data["total_energy"] - 0.5*data["velocity_x"]**2.0
                if data.ds.dimensionality > 1:
                    ret -= 0.5*data["velocity_y"]**2.0
                if data.ds.dimensionality > 2:
                    ret -= 0.5*data["velocity_z"]**2.0
                ret -= data["magnetic_energy"]/data["density"]
                return ret
            self.add_field(("gas", "thermal_energy"), sampling_type="cell",
                           function=_sub_b,
                           units = unit_system["specific_energy"])
        

    def setup_particle_fields(self, ptype, ftype='gas', num_neighbors=64):
        super(EnzoPFieldInfo, self).setup_particle_fields(
            ptype, ftype=ftype, num_neighbors=num_neighbors)
        self.setup_particle_mass_field(ptype)

    def setup_particle_mass_field(self, ptype):
        name = "particle_mass"
        if ptype in self.ds.particle_unions:
            add_union_field(self, ptype, name, "code_mass")
            return

        constants = nested_dict_get(
            self.ds.parameters, ("Particle", ptype, "constants"),
            default=[])
        if not constants:
            names = []
        else:
            if not isinstance(constants[0], tuple):
                constants = (constants,)
            names = [c[0] for c in constants]

        if "mass" in names:
            val = constants[names.index("mass")][2]
            val = self.ds.quan(val, self.ds.mass_unit)
            if self.ds.cosmological_simulation:
                val /= self.ds.domain_dimensions.prod()

            def _pmass(field, data):
                return val * data[ptype, "particle_ones"]
            self.add_field((ptype, name),
                            function=_pmass, units="code_mass",
                            sampling_type="particle")
