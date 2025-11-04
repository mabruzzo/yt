from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, TypeAlias

import numpy as np

from yt._typing import FieldKey, ParticleCoordinateTuple, ParticleType
from yt.utilities.io_handler import BaseIOHandler

from .misc import _CachedH5Openner

ChunkT: TypeAlias = Any  # <- maybe yt._typing could provide a better definition?
SelectorT: TypeAlias = Any  # <- maybe yt._typing could provide a better definition?


class ChollaIOHandler(BaseIOHandler):
    _particle_reader = False
    _dataset_type = "cholla"

    def _read_particle_coords(
        self, chunks: Iterable[ChunkT], ptf: Mapping[ParticleType, Sequence[str]]
    ) -> Iterator[ParticleCoordinateTuple]:
        # An iterator that yields particle coordinates for each chunk by particle type.
        #
        # The output tuple has the form (particle type, xyz, hsml) by chunk. It is not
        # obvious to me what hsml means. The super-class has a note that if the
        # frontend does not have a smoothing length, yield (particle type, xyz, 0.0).
        # Since the Enzo-E frontend does that, we'll also do that (but, I'm not
        # entirely sure what a smoothing length is in this context)
        yield from (
            (ptype, xyz, 0.0)
            for ptype, xyz in self._read_particle_fields(chunks, ptf, None)
        )

    def _read_particle_fields(
        self,
        chunks: Iterable[ChunkT],
        ptf: Mapping[ParticleType, Sequence[str]],
        selector: SelectorT | None,
    ) -> (
        Iterator[tuple[tuple[ParticleType, str], np.ndarray]]
        | Iterator[tuple[ParticleType, tuple[np.ndarray, np.ndarray, np.ndarray]]]
    ):
        mapper = self.ds.index._dataset_mapping.particle_mapping
        with _CachedH5Openner(mode="r") as h5_context_manager:
            for chunk in chunks:  # These should be organized by grid filename
                for obj in chunk.objs:
                    # NOTE: the Enzo-E frontend doesn't pre-fill the grid_particle_count
                    # attribute of the index.
                    # - Instead, it tracks the particle count as part of the Grid class
                    # - Moreover, the particle counts are lazily initialized right here
                    #   & now.
                    # - we may want to move to adopting a similar model so we don't
                    #   need to manually pre-fetch all particle counts
                    tot_particles_in_obj = obj.index.grid_particle_count[obj.id]
                    if tot_particles_in_obj == 0:
                        continue

                    assert obj.particle_filename is not None  # sanity check!

                    fh = h5_context_manager.open_fh(obj.particle_filename)

                    assert len(self.ds.index._dataset_mapping.particle_types) <= 1

                    # get the indices in a generic dataset that correspond to obj.id
                    # (in the future, the indices probably need to be specific to both
                    # the obj.id and the particle-type)
                    idx = mapper.idx_map[obj.id]

                    for ptype, field_list in sorted(ptf.items()):
                        # access HDF5 group containing the datasets of ptype properties
                        grp = fh[mapper.h5_group_map[ptype]]

                        # retrieve the particle positions
                        x, y, z = [grp[f"pos_{ax}"][idx].astype("=f8") for ax in "xyz"]

                        if selector is None:
                            # This only ever happens if the call is made from
                            # _read_particle_coords.
                            yield ptype, (x, y, z)

                        mask = selector.select_points(x, y, z, 0.0)
                        if mask is None:
                            continue
                        for field in field_list:
                            data = np.asarray(grp[field][idx], "=f8")
                            yield (ptype, field), data[mask]

    def io_iter(
        self, chunks: Iterable[ChunkT], fields: Sequence[FieldKey]
    ) -> Iterator[tuple[FieldKey, Any, np.ndarray]]:
        # this is loosely inspired by the implementation used for Enzo/Enzo-E
        # - those other implementations use the lower-level hdf5 interface. Unclear
        #   whether that affords any advantages...
        mapper = self.ds.index._dataset_mapping.field_mapping

        if len(fields) == 0:
            yield from ()
            return
        # peek ahead at the field type
        common_ftype, _ = fields[0]

        with _CachedH5Openner(mode="r") as h5_context_manager:
            for chunk in chunks:
                for obj in chunk.objs:
                    if obj.filename is None:  # unclear when this case arises...
                        continue

                    # ensure the file containing data for obj is open
                    fh = h5_context_manager.open_fh(obj.filename)

                    # access the HDF5 group containing the datasets of field values
                    grp = fh[mapper.h5_group_map[common_ftype]]
                    # get the indices in a generic dataset that correspond to obj.id
                    idx = mapper.idx_map[obj.id]

                    for field in fields:
                        ftype, fname = field
                        assert ftype == common_ftype  # sanity check!
                        yield field, obj, grp[fname][idx].astype("=f8")

    def _read_chunk_data(self, chunk, fields):
        raise NotImplementedError
