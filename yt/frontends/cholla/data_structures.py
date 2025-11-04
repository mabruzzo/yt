import os
import weakref

import numpy as np

from yt._typing import ParticleType
from yt.data_objects.index_subobjects.grid_patch import AMRGridPatch
from yt.data_objects.static_output import Dataset
from yt.funcs import setdefaultattr
from yt.geometry.api import Geometry
from yt.geometry.grid_geometry_handler import GridIndex
from yt.utilities.logger import ytLogger as mylog
from yt.utilities.on_demand_imports import _h5py

from .fields import ChollaFieldInfo
from .misc import _detect_particle_fields, _determine_data_layout


class ChollaGrid(AMRGridPatch):
    _id_offset = 0

    def __init__(self, id, index, level, dims, filename, particle_filename=None):
        super().__init__(id, filename=filename, index=index)
        self.Parent = None
        self.Children = []
        self.Level = level
        self.ActiveDimensions = dims
        self.particle_filename = particle_filename


class ChollaHierarchy(GridIndex):
    grid = ChollaGrid
    _grid_chunksize = 1

    def __init__(self, ds, dataset_type="cholla"):
        self.dataset_type = dataset_type
        self.dataset = weakref.proxy(ds)
        # for now, the index file is the dataset!
        self.index_filename = self.dataset.parameter_filename
        self.directory = os.path.dirname(self.index_filename)
        # float type for the simulation edges and must be float64 now
        self.float_type = np.float64
        super().__init__(ds, dataset_type)

    def _detect_output_fields(self):
        # importantly, this is called after ``_count_grids`` & ``_parse_index``

        # Do this only on the root processor to save disk work (this is what the Enzo-E
        # frontend does)
        if self.comm.rank in (0, None):
            print(self.index_filename)
            with _h5py.File(self.index_filename, mode="r") as h5f:
                grp = h5f.get("field", h5f)
                _field_list = [("cholla", k) for k in grp.keys()]
            _field_list.extend(_detect_particle_fields(self._dataset_mapping))
        else:
            _field_list = None
        self.field_list = list(self.comm.mpi_bcast(_field_list))

        # we are following the convention of the Enzo-E frontend and setting particle
        # types right here. If we want to do it sooner, (before fully initializing the
        # ChollaHierarchy instance), that will involve some refactoring
        self.dataset.particle_types = self._dataset_mapping.particle_types
        self.dataset.particle_types_raw = self._dataset_mapping.particle_types

    def _count_grids(self):
        with _h5py.File(self.index_filename, "r") as f:
            self._blockid_location_arr, self._dataset_mapping = _determine_data_layout(
                f
            )
        self.num_grids = self._blockid_location_arr.size

    def _parse_index(self):
        # fill in self.grid_left_edge, self.grid_right_edge, self.grid_particle_count,
        # self.grid_dimensions and self.grid_levels

        _dset_mapping = self._dataset_mapping
        _p_mapping = self._dataset_mapping.particle_mapping

        # first, handle everything other than self.grid_particle_count
        if _p_mapping is not None:
            _get_particle_fname = _p_mapping.fname_template.format
            _concatenated_particles = (
                _get_particle_fname(blockid=0) == _p_mapping.fname_template
            )
        else:
            _concatenated_particles = False

            def _get_particle_fname(blockid):
                return None

        _get_field_fname = _dset_mapping.field_mapping.fname_template.format

        self.grids = np.empty(self.num_grids, dtype="object")

        shape_arr = np.array(self._blockid_location_arr.shape)
        dims_local = (self.ds.domain_dimensions[:] / shape_arr).astype("=i8")

        for idx3D, blockid in np.ndenumerate(self._blockid_location_arr):
            idx3D_arr = np.array(idx3D)
            left_frac = idx3D_arr / shape_arr
            right_frac = (1 + idx3D_arr) / shape_arr

            level = 0

            self.grids[blockid] = self.grid(
                id=blockid,
                index=self,
                level=level,
                dims=dims_local,
                filename=_get_field_fname(blockid=blockid),
                particle_filename=_get_particle_fname(blockid=blockid),
            )

            self.grid_left_edge[blockid, :] = left_frac
            self.grid_right_edge[blockid, :] = right_frac
            self.grid_dimensions[blockid, :] = dims_local
            self.grid_levels[blockid, 0] = level

        slope = self.ds.domain_width / self.ds.arr(np.ones(3), "code_length")
        self.grid_left_edge = self.grid_left_edge * slope + self.ds.domain_left_edge
        self.grid_right_edge = self.grid_right_edge * slope + self.ds.domain_left_edge

        self.max_level = 0

        # now, deal with initializing self.grid_particle_count
        if len(_dset_mapping.particle_types) == 0:
            self.grid_particle_count[()] = 0

        elif _concatenated_particles:
            for g in self.grids:
                idx = _p_mapping.idx_map[g.id]
                assert len(idx) == 1  # sanity check!
                slc = idx[0]
                assert (
                    (slc.start >= 0)
                    and (slc.stop >= 0)
                    and (slc.step is None or slc.step == 1)
                )  # another sanity check!
                self.grid_particle_count[g.id, 0] = slc.stop - slc.start

        else:
            # It's unfortunate that we need to go through and count up all of the
            # particles. To try to mitigate the cost, lets only do it on the root
            # processor to save disk work
            # -> in the future, we might be able to adjust Cholla's file format to
            #    reduce this cost
            # -> the Enzo-E frontend appears to entirely skip initializing the
            #    self.grid_particle_count arrays (and I think it loads the data as it
            #    becomes needed)

            if self.comm.rank in (0, None):
                for g in self.grids:
                    with _h5py.File(g.particle_filename, "r") as f:
                        n_particles = f.attrs["n_particles_local"][0]
                    self.grid_particle_count[g.id, 0] = n_particles
            else:
                pass
            self.grid_particle_count = self.comm.mpi_bcast(self.grid_particle_count)

    def _populate_grid_objects(self):
        for i in range(self.num_grids):
            g = self.grids[i]
            g._prepare_grid()
            g._setup_dx()


class ChollaDataset(Dataset):
    """
    Cholla-specific output, set at a fixed time.
    """

    # set class variable values:
    _load_requirements = ["h5py"]
    _index_class = ChollaHierarchy
    _field_info_class = ChollaFieldInfo
    # set default instance variable values:
    particle_types: tuple[ParticleType, ...] = ()
    particle_types_raw: tuple[ParticleType, ...] | None = None

    def __init__(
        self,
        filename,
        dataset_type="cholla",
        storage_filename=None,
        units_override=None,
        unit_system="cgs",
    ):
        self.fluid_types += ("cholla",)
        super().__init__(filename, dataset_type, units_override=units_override)
        self.storage_filename = storage_filename

    def _set_code_unit_attributes(self):
        # This is where quantities are created that represent the various
        # on-disk units.  These are the defaults, but if they are listed
        # in the HDF5 attributes for a file, which is loaded first, then those are
        # used instead.
        #
        if not self.length_unit:
            self.length_unit = self.quan(1.0, "pc")
        if not self.mass_unit:
            self.mass_unit = self.quan(1.0, "Msun")
        if not self.time_unit:
            self.time_unit = self.quan(1000, "yr")
        if not self.velocity_unit:
            self.velocity_unit = self.quan(1.0, "cm/s")
        if not self.magnetic_unit:
            self.magnetic_unit = self.quan(1.0, "gauss")

        for key, unit in self.__class__.default_units.items():
            setdefaultattr(self, key, self.quan(1, unit))

    def _parse_parameter_file(self):
        with _h5py.File(self.parameter_filename, mode="r") as h5f:
            attrs = h5f.attrs
            self.parameters = dict(attrs.items())
            self.domain_left_edge = attrs["bounds"][:].astype("=f8")
            self.domain_right_edge = self.domain_left_edge + attrs["domain"][:].astype(
                "=f8"
            )
            self.dimensionality = len(attrs["dims"][:])
            self.domain_dimensions = attrs["dims"][:].astype("=i8")
            self.current_time = attrs["t"][:]
            self._periodicity = tuple(attrs.get("periodicity", (False, False, False)))
            self.gamma = attrs.get("gamma", 5.0 / 3.0)
            if (self.default_species_fields is not None) and "mu" in attrs:
                raise ValueError(
                    'default_species_fields must be None when "mu" is an hdf5 attribute'
                )
            elif "mu" in attrs:
                self.mu = attrs["mu"]
            elif self.default_species_fields is None:
                # other yt-machinery can't handle ds.mu == None, so we simply
                # avoid defining the mu attribute if we don't know its value
                mylog.info(
                    'add the "mu" hdf5 attribute OR use the default_species_fields kwarg '
                    "to compute temperature"
                )
            self.refine_by = 1

            # If header specifies code units, default to those (in CGS)
            length_unit = attrs.get("length_unit", None)
            mass_unit = attrs.get("mass_unit", None)
            time_unit = attrs.get("time_unit", None)
            velocity_unit = attrs.get("velocity_unit", None)
            magnetic_unit = attrs.get("magnetic_unit", None)
            if length_unit:
                self.length_unit = self.quan(length_unit[0], "cm")
            if mass_unit:
                self.mass_unit = self.quan(mass_unit[0], "g")
            if time_unit:
                self.time_unit = self.quan(time_unit[0], "s")
            if velocity_unit:
                self.velocity_unit = self.quan(velocity_unit[0], "cm/s")
            if magnetic_unit:
                self.magnetic_unit = self.quan(magnetic_unit[0], "gauss")

            # this minimalistic implementation fills the requirements for
            # this frontend to run, change it to make it run _correctly_ !
            for key, unit in self.__class__.default_units.items():
                setdefaultattr(self, key, self.quan(1, unit))

        # CHOLLA cannot yet be run as a cosmological simulation
        self.cosmological_simulation = 0
        self.current_redshift = 0.0
        self.omega_lambda = 0.0
        self.omega_matter = 0.0
        self.hubble_constant = 0.0

        # CHOLLA datasets are always unigrid cartesian
        self.geometry = Geometry.CARTESIAN

    @classmethod
    def _is_valid(cls, filename: str, *args, **kwargs) -> bool:
        # This accepts a filename or a set of arguments and returns True or
        # False depending on if the file is of the type requested.
        if cls._missing_load_requirements():
            return False

        try:
            fileh = _h5py.File(filename, mode="r")
        except OSError:
            return False

        try:
            attrs = fileh.attrs
        except AttributeError:
            return False
        else:
            return (
                "bounds" in attrs
                and "domain" in attrs
                and attrs.get("data_type") != "yt_light_ray"
            )
        finally:
            fileh.close()
