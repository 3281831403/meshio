"""Contract tests: model-layer validation and exact roundtrips.

These tests pin down the contract between the Mesh model and the file
formats:

- The Mesh constructor rejects inconsistent cell data and names the
  offending field.
- Writers refuse formats that cannot represent the mesh as-is, and name
  the data that cannot be stored.
- Formats that can represent the mesh must roundtrip it exactly: cell
  blocks stay aligned with their per-block data, integer ids stay
  integers, and all arrays come back in native byte order.
"""

import numpy as np
import pytest

import meshio
from meshio._exceptions import WriteError


def mixed_mesh():
    """Multi-block triangle/quad mesh with int material ids and a float
    point field."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    cells = [
        ("triangle", np.array([[0, 1, 4], [0, 4, 3]])),
        ("quad", np.array([[1, 2, 5, 4]])),
        ("triangle", np.array([[2, 6, 5]])),
    ]
    cell_data = {
        "material": [
            np.array([1, 1], dtype=np.int32),
            np.array([2], dtype=np.int32),
            np.array([3], dtype=np.int32),
        ]
    }
    point_data = {"temperature": np.linspace(0.0, 1.0, 7)}
    return meshio.Mesh(points, cells, point_data=point_data, cell_data=cell_data)


def assert_exact_roundtrip(in_mesh, out_mesh, point_atol=1.0e-15):
    # blocks and per-block data stay aligned
    assert [c.type for c in out_mesh.cells] == [c.type for c in in_mesh.cells]
    for c0, c1 in zip(in_mesh.cells, out_mesh.cells):
        assert c0.data.shape == c1.data.shape
        assert np.array_equal(c0.data, c1.data)

    # integer ids stay integers, and stay per-block aligned
    for name, blocks in in_mesh.cell_data.items():
        out_blocks = out_mesh.cell_data[name]
        assert len(out_blocks) == len(blocks)
        for b0, b1 in zip(blocks, out_blocks):
            assert b0.dtype == b1.dtype
            assert np.array_equal(b0, b1)

    for name, values in in_mesh.point_data.items():
        assert out_mesh.point_data[name].dtype == values.dtype
        assert np.allclose(out_mesh.point_data[name], values, atol=point_atol, rtol=0.0)

    # all arrays come back in native byte order
    assert out_mesh.points.dtype.isnative
    for c in out_mesh.cells:
        assert c.data.dtype.isnative
    for values in out_mesh.point_data.values():
        assert values.dtype.isnative
    for blocks in out_mesh.cell_data.values():
        for b in blocks:
            assert b.dtype.isnative


def test_mesh_constructor_inconsistent_cell_data_dtype():
    mesh = mixed_mesh()
    bad = {
        "material": [
            np.array([1, 1], dtype=np.int32),
            np.array([2.5], dtype=np.float64),
            np.array([3], dtype=np.int32),
        ]
    }
    with pytest.raises(ValueError, match="material"):
        meshio.Mesh(mesh.points, mesh.cells, cell_data=bad)


def test_mesh_constructor_inconsistent_cell_data_shape():
    mesh = mixed_mesh()
    bad = {
        "material": [
            np.ones((2, 2)),
            np.ones((1, 3)),
            np.ones((1, 2)),
        ]
    }
    with pytest.raises(ValueError, match="material"):
        meshio.Mesh(mesh.points, mesh.cells, cell_data=bad)


def test_mesh_constructor_inconsistent_cell_data_block_count():
    mesh = mixed_mesh()
    bad = {"material": [np.array([1, 1]), np.array([2])]}
    with pytest.raises(ValueError, match="material"):
        meshio.Mesh(mesh.points, mesh.cells, cell_data=bad)


@pytest.mark.parametrize("file_format", ["vtk42", "vtk", "vtk51"])
@pytest.mark.parametrize("binary", [True, False])
def test_roundtrip_vtk(tmp_path, file_format, binary):
    mesh = mixed_mesh()
    filepath = tmp_path / "test.vtk"
    meshio.write(filepath, mesh, file_format=file_format, binary=binary)
    out_mesh = meshio.read(filepath)
    assert_exact_roundtrip(mesh, out_mesh)


@pytest.mark.parametrize("binary", [True, False])
def test_roundtrip_vtu(tmp_path, binary):
    mesh = mixed_mesh()
    filepath = tmp_path / "test.vtu"
    meshio.write(filepath, mesh, binary=binary)
    out_mesh = meshio.read(filepath)
    # VTU ASCII writes floats with only 11 significant digits
    assert_exact_roundtrip(mesh, out_mesh, point_atol=1.0e-15 if binary else 1.0e-10)


def test_roundtrip_xdmf(tmp_path):
    mesh = mixed_mesh()
    filepath = tmp_path / "test.xdmf"
    meshio.write(filepath, mesh)
    out_mesh = meshio.read(filepath)
    assert_exact_roundtrip(mesh, out_mesh)


def test_roundtrip_abaqus_geometry(tmp_path):
    # Abaqus cannot store point/cell data; the geometry and the block
    # structure must still roundtrip exactly.
    mesh = mixed_mesh()
    mesh = meshio.Mesh(mesh.points, mesh.cells)
    filepath = tmp_path / "test.inp"
    meshio.write(filepath, mesh)
    out_mesh = meshio.read(filepath)
    assert [c.type for c in out_mesh.cells] == [c.type for c in mesh.cells]
    for c0, c1 in zip(mesh.cells, out_mesh.cells):
        assert np.array_equal(c0.data, c1.data)
    assert np.allclose(out_mesh.points, mesh.points, atol=1.0e-15, rtol=0.0)


def test_abaqus_refuses_data(tmp_path):
    mesh = mixed_mesh()
    filepath = tmp_path / "test.inp"
    with pytest.raises(WriteError, match="temperature"):
        meshio.write(filepath, mesh)

    mesh2 = meshio.Mesh(mesh.points, mesh.cells, cell_data=mesh.cell_data)
    with pytest.raises(WriteError, match="material"):
        meshio.write(filepath, mesh2)


@pytest.mark.parametrize("binary", [True, False])
def test_roundtrip_gmsh22_block_alignment(tmp_path, binary):
    # gmsh 2.2 cannot store integer data, use float fields here. The
    # per-block data must stay aligned with the blocks.
    mesh = mixed_mesh()
    mesh = meshio.Mesh(
        mesh.points,
        mesh.cells,
        point_data=mesh.point_data,
        cell_data={
            "material": [b.astype(float) for b in mesh.cell_data["material"]],
        },
    )
    filepath = tmp_path / "test.msh"
    meshio.write(filepath, mesh, file_format="gmsh22", binary=binary)
    out_mesh = meshio.read(filepath)

    assert [c.type for c in out_mesh.cells] == [c.type for c in mesh.cells]
    for c0, c1 in zip(mesh.cells, out_mesh.cells):
        assert np.array_equal(c0.data, c1.data)
    out_blocks = out_mesh.cell_data["material"]
    assert len(out_blocks) == 3
    for b0, b1 in zip(mesh.cell_data["material"], out_blocks):
        assert np.allclose(b0, b1, atol=1.0e-15, rtol=0.0)


@pytest.mark.parametrize("fmt_version", ["2.2", "4.1"])
def test_gmsh_refuses_int_data(tmp_path, fmt_version):
    # gmsh stores all data as doubles; integer ids would silently come
    # back as floats, so the writer must refuse and name the field.
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    cells = [("triangle", np.array([[0, 1, 2]]))]
    cell_data = {"material": [np.array([7], dtype=np.int32)]}
    mesh = meshio.Mesh(points, cells, cell_data=cell_data)
    filepath = tmp_path / "test.msh"
    with pytest.raises(WriteError, match="material"):
        meshio.gmsh.write(filepath, mesh, fmt_version=fmt_version)


def test_vtk_binary_native_byteorder(tmp_path):
    # Binary VTK files store big-endian data; reading must hand back
    # native-byteorder arrays.
    mesh = mixed_mesh()
    filepath = tmp_path / "test.vtk"
    meshio.write(filepath, mesh, file_format="vtk42", binary=True)
    out_mesh = meshio.read(filepath)
    for dt in [out_mesh.points.dtype] + [c.data.dtype for c in out_mesh.cells]:
        assert dt.isnative
        assert dt.byteorder != ">"
