"""
Testsuite for PlotWindow class



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
import itertools
import os
import tempfile
import shutil
import unittest
from yt.extern.parameterized import parameterized, param
from yt.testing import \
    fake_random_pf, assert_equal, assert_rel_equal, assert_array_equal
from yt.utilities.answer_testing.framework import \
    requires_pf, data_dir_load, PlotWindowAttributeTest
from yt.visualization.api import \
    SlicePlot, ProjectionPlot, OffAxisSlicePlot, OffAxisProjectionPlot
from yt.data_objects.yt_array import YTArray, YTQuantity

def setup():
    """Test specific setup."""
    from yt.config import ytcfg
    ytcfg["yt", "__withintesting"] = "True"


def assert_fname(fname):
    """Function that checks file type using libmagic"""
    if fname is None:
        return

    with open(fname, 'rb') as fimg:
        data = fimg.read()
    data = str(data)
    image_type = ''

    # see http://www.w3.org/TR/PNG/#5PNG-file-signature
    if data.startswith('\211PNG\r\n\032\n'):
        image_type = '.png'
    # see http://www.mathguide.de/info/tools/media-types/image/jpeg
    elif data.startswith('\377\330'):
        image_type = '.jpeg'
    elif data.startswith('%!PS-Adobe'):
        if 'EPSF' in data[:data.index('\n')]:
            image_type = '.eps'
        else:
            image_type = '.ps'
    elif data.startswith('%PDF'):
        image_type = '.pdf'

    return image_type == os.path.splitext(fname)[1]


TEST_FLNMS = [None, 'test.png', 'test.eps',
              'test.ps', 'test.pdf']
M7 = "DD0010/moving7_0010"
WT = "WindTunnel/windtunnel_4lev_hdf5_plt_cnt_0030"

ATTR_ARGS = {"pan": [(((0.1, 0.1), ), {})],
             "pan_rel": [(((0.1, 0.1), ), {})],
             "set_axes_unit": [(("kpc", ), {}),
                               (("Mpc", ), {}),
                               ((("kpc", "kpc"),), {}),
                               ((("kpc", "Mpc"),), {})],
             "set_buff_size": [((1600, ), {}),
                               (((600, 800), ), {})],
             "set_center": [(((0.4, 0.3), ), {})],
             "set_cmap": [(('density', 'RdBu'), {}),
                          (('density', 'kamae'), {})],
             "set_font": [(({'family': 'sans-serif', 'style': 'italic',
                             'weight': 'bold', 'size': 24}, ), {})],
             "set_log": [(('density', False), {})],
             "set_window_size": [((7.0, ), {})],
             "set_zlim": [(('density', 1e-25, 1e-23), {}),
                          (('density', 1e-25, None), {'dynamic_range': 4})],
             "zoom": [((10, ), {})]}


CENTER_SPECS = ("m",
                "M",
                "max",
                "Max",
                "c",
                "C",
                "center",
                "Center",
                [0.5, 0.5, 0.5],
                [[0.2, 0.3, 0.4], "cm"],
                YTArray([0.3, 0.4, 0.7], "cm"))

@requires_pf(M7)
def test_attributes():
    """Test plot member functions that aren't callbacks"""
    plot_field = 'density'
    decimals = 3

    pf = data_dir_load(M7)
    for ax in 'xyz':
        for attr_name in ATTR_ARGS.keys():
            for args in ATTR_ARGS[attr_name]:
                test = PlotWindowAttributeTest(pf, plot_field, ax, attr_name,
                                               args, decimals)
                test_attributes.__name__ = test.description
                yield test


@requires_pf(WT)
def test_attributes_wt():
    plot_field = 'density'
    decimals = 3

    pf = data_dir_load(WT)
    ax = 'z'
    for attr_name in ATTR_ARGS.keys():
        for args in ATTR_ARGS[attr_name]:
            yield PlotWindowAttributeTest(pf, plot_field, ax, attr_name,
                                          args, decimals)


class TestSetWidth(unittest.TestCase):

    pf = None

    def setUp(self):
        if self.pf is None:
            self.pf = fake_random_pf(64)
            self.slc = SlicePlot(self.pf, 0, "density")

    def _assert_05cm(self):
        assert_array_equal([self.slc.xlim, self.slc.ylim, self.slc.width],
                         [(YTQuantity(0.25, 'cm'), YTQuantity(0.75, 'cm')),
                          (YTQuantity(0.25, 'cm'), YTQuantity(0.75, 'cm')),
                          (YTQuantity(0.5,  'cm'), YTQuantity(0.5,  'cm'))])

    def _assert_05_075cm(self):
        assert_array_equal([self.slc.xlim, self.slc.ylim, self.slc.width],
                         [(YTQuantity(0.25,  'cm'), YTQuantity(0.75,  'cm')),
                          (YTQuantity(0.125, 'cm'), YTQuantity(0.875, 'cm')),
                          (YTQuantity(0.5,   'cm'), YTQuantity(0.75,  'cm'))])

    def test_set_width_one(self):
        assert_equal([self.slc.xlim, self.slc.ylim, self.slc.width],
                     [(0.0, 1.0), (0.0, 1.0), (1.0, 1.0)])

    def test_set_width_nonequal(self):
        self.slc.set_width((0.5, 0.8))
        assert_rel_equal([self.slc.xlim, self.slc.ylim, self.slc.width],
                         [(0.25, 0.75), (0.1, 0.9), (0.5, 0.8)], 15)

    def test_twoargs_eq(self):
        self.slc.set_width(0.5, 'cm')
        self._assert_05cm()

    def test_tuple_eq(self):
        self.slc.set_width((0.5, 'cm'))
        self._assert_05cm()

    def test_tuple_of_tuples_neq(self):
        self.slc.set_width(((0.5, 'cm'), (0.75, 'cm')))
        self._assert_05_075cm()

    def test_tuple_of_tuples_neq(self):
        self.slc.set_width(((0.5, 'cm'), (0.0075, 'm')))
        self._assert_05_075cm()


class TestPlotWindowSave(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        test_pf = fake_random_pf(64)
        normal = [1, 1, 1]
        ds_region = test_pf.h.region([0.5] * 3, [0.4] * 3, [0.6] * 3)
        projections = []
        projections_ds = []
        projections_c = []
        for dim in range(3):
            projections.append(ProjectionPlot(test_pf, dim, "density"))
            projections_ds.append(ProjectionPlot(test_pf, dim, "density",
                                                 data_source=ds_region))
        for center in CENTER_SPECS:
            projections_c.append(ProjectionPlot(test_pf, dim, "density",
                                                center=center))

        cls.slices = [SlicePlot(test_pf, dim, "density") for dim in range(3)]
        cls.projections = projections
        cls.projections_ds = projections_ds
        cls.projections_c = projections_c
        cls.offaxis_slice = OffAxisSlicePlot(test_pf, normal, "density")
        cls.offaxis_proj = OffAxisProjectionPlot(test_pf, normal, "density")

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.curdir = os.getcwd()
        os.chdir(self.tmpdir)

    def tearDown(self):
        os.chdir(self.curdir)
        shutil.rmtree(self.tmpdir)

    @parameterized.expand(
        param.explicit(item)
        for item in itertools.product(range(3), TEST_FLNMS))
    def test_slice_plot(self, dim, fname):
        assert assert_fname(self.slices[dim].save(fname)[0])

    @parameterized.expand(
        param.explicit(item)
        for item in itertools.product(range(3), TEST_FLNMS))
    def test_projection_plot(self, dim, fname):
        assert assert_fname(self.projections[dim].save(fname)[0])

    @parameterized.expand([(0, ), (1, ), (2, )])
    def test_projection_plot_ds(self, dim):
        self.projections_ds[dim].save()

    @parameterized.expand([(i, ) for i in range(len(CENTER_SPECS))])
    def test_projection_plot_c(self, dim):
        self.projections_c[dim].save()

    @parameterized.expand(
        param.explicit((fname, ))
        for fname in TEST_FLNMS)
    def test_offaxis_slice_plot(self, fname):
        assert assert_fname(self.offaxis_slice.save(fname)[0])

    @parameterized.expand(
        param.explicit((fname, ))
        for fname in TEST_FLNMS)
    def test_offaxis_projection_plot(self, fname):
        assert assert_fname(self.offaxis_proj.save(fname)[0])
